import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn


class VGG19Truncated(nn.Module):
    def __init__(self):
        super().__init__()
        original_vgg = vgg19_bn()
        self.truncated_vgg = nn.Sequential(*list(original_vgg.children())[:-2])

    def forward(self, x):
        return self.truncated_vgg(x)


class ImageEncoder(nn.Module):
    def __init__(self, image_height, hidden_size):
        super().__init__()

        out_feature_maps = 512 # vgg outputs 512 feature maps
        feature_height = image_height // 2**5
        input_size = feature_height * out_feature_maps
        self.hidden_size = hidden_size
        self.vgg = VGG19Truncated()
        self.gru1 = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        output = self.vgg(x)
        batch_size, f_maps, h, w = output.size()
        x = output.transpose(1, 3).resize(batch_size, w, h * f_maps)

        x, hidden = self.gru1(x)

        x = self.dropout(x)

        x, hidden = self.gru2(x)

        return x, hidden

    def run_inference(self, x):
        return self.forward(x)


class HybridAttention(nn.Module):
    # adapted from paper https://arxiv.org/pdf/1506.07503.pdf, equation (9)

    def __init__(self, embedding_size, decoder_state_dim, hidden_size, filters, kernel_size):
        """

        :param embedding_size: size of encoder embedding vectors
        :param decoder_state_dim: size of decoder hidden state
        :param hidden_size: number of units in a hidden layer (of a scoring network)
        :param filters: number of 1d convolutional filters
        :param kernel_size: kernel size of each convolutional filter
        """
        super().__init__()

        # corresponds to a matrix F used to convolve attention weights from a previous step
        self.conv1d = nn.Conv1d(1, filters, kernel_size=kernel_size, padding='same')

        # pack W, V, U, b into a single linear layer operating on vectors of size
        # embedding_size + hidden_size + filters
        self.linear1 = nn.Linear(
            embedding_size + decoder_state_dim + filters, hidden_size, bias=True
        )

        # w transpose from equation (9)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, attention_weights):
        """Computes attention context vectors as a weighted sum of encoder_outputs
        :param decoder_hidden: hidden state of the decoder
        :type decoder_hidden: tensor of shape (1, batch_size, num_cells)
        :param encoder_outputs: outputs of the encoder
        :type encoder_outputs: tensor of shape (batch_size, num_steps, embedding_size)
        :param attention_weights: attention weights computed on the previous time step
        :type attention_weights: tensor of shape (batch_size, num_embedding_vectors)
        :return: new attention weights
        :rtype: tensor of shape (batch_size, num_embedding_vectors)
        """

        embeddings = encoder_outputs
        batch_size, num_steps, embedding_size = embeddings.shape
        
        s = self.repeat_state(decoder_hidden, num_steps)
        
        features = self.convolve(self.conv1d, attention_weights)
        f = features.transpose(1, 2)

        x = self.concat_features(embeddings, s, f)

        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        scores = x.squeeze(2)

        new_attention_weights = self.smoothing(scores)

        return new_attention_weights

    @staticmethod
    def convolve(conv1d, attention_weights):
        attention_weights = attention_weights.unsqueeze(1)
        return conv1d(attention_weights)

    @staticmethod
    def repeat_state(t, num_repeats):
        """Maps tensor of shape (1, B, n) to shape (B, num_repeats, n) where each step is repeated"""
        t = t.squeeze(0)
        t = t.unsqueeze(1)

        return t.repeat(1, num_repeats, 1)

    @staticmethod
    def concat_features(*args):
        return torch.cat(args, dim=2)

    @staticmethod
    def smoothing(scores):
        sigmoid_scores = torch.sigmoid(scores)
        _, n = sigmoid_scores.shape
        feature_sum = sigmoid_scores.sum(dim=1)
        feature_sum = feature_sum.unsqueeze(1).repeat(1, n)
        return sigmoid_scores / feature_sum

    @staticmethod
    def compute_context_vectors(new_attention_weights, embeddings):
        return torch.bmm(new_attention_weights.unsqueeze(1), embeddings).squeeze(1)

    @staticmethod
    def initial_attention_weights(batch_size, num_embeddings, device):
        attention_weights = list(
            reversed([i / num_embeddings for i in range(num_embeddings)])
        )
        attention_weights = torch.tensor(attention_weights, device=device)
        attention_weights = attention_weights.unsqueeze(0).repeat(batch_size, 1)

        return HybridAttention.smoothing(attention_weights)


class AttendingDecoder(nn.Module):
    def __init__(self, sos_token, context_size, y_size,
                 hidden_size=256, filters=10, kernel_size=5):
        super().__init__()

        # start of sequence token
        self.sos_token = sos_token

        self.y_size = y_size

        self.hidden_size = hidden_size

        self.attention = HybridAttention(
            embedding_size=context_size, decoder_state_dim=hidden_size,
            hidden_size=hidden_size, filters=filters, kernel_size=kernel_size
        )
        self.decoder_gru = nn.GRU(context_size + y_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, y_size)

    def forward(self, encodings, y_shifted=None):
        decoder_hidden = torch.zeros(1, len(encodings), self.hidden_size, device=encodings.device)

        if y_shifted is None:
            # close loop inference
            sos = torch.zeros(len(encodings), self.y_size, dtype=encodings.dtype, device=encodings.device)
            sos[:, self.sos_token] = 1
            return self.close_loop_inference(encodings, decoder_hidden, sos)

        # teacher forcing

        batch_size, num_steps, num_classes = y_shifted.size()
        num_embeddings = encodings.shape[1]

        outputs = []

        # initializing attention weights (higher values on the left end, lower ones on the right end)
        attention_weights = self.attention.initial_attention_weights(batch_size, num_embeddings, device=encodings.device)

        for t in range(0, num_steps):
            y_hat_prev = y_shifted[:, t, :]
            log_pmf, decoder_hidden, attention_weights = self.predict_next(
                decoder_hidden, encodings, attention_weights, y_hat_prev
            )
            outputs.append(log_pmf)

        y_hat = torch.stack(outputs, dim=1)
        return [y_hat]

    def close_loop_inference(self, encodings, decoder_hidden, sos):
        outputs, _ = self._do_inference(encodings, decoder_hidden, sos)
        return [outputs]

    def debug_attention(self, encodings):
        decoder_hidden = torch.zeros(1, len(encodings), self.hidden_size, device=encodings.device)
        sos = torch.zeros(len(encodings), self.y_size, dtype=encodings.dtype, device=encodings.device)
        sos[:, self.sos_token] = 1
        return self._do_inference(encodings, decoder_hidden, sos)

    def _do_inference(self, encodings, decoder_hidden, sos):
        """Generates sequence of tokens in autoregressive way
        Returns both predicted sequence of tokens and list of attention weight tensors,
        one tensor for each predicted token
        """
        outputs = [sos]

        y_hat_prev = sos

        batch_size = len(encodings)
        batch_indices = range(batch_size)

        _, num_embeddings, _ = encodings.shape
        attention_weights = self.attention.initial_attention_weights(
            batch_size, num_embeddings, device=encodings.device
        )

        attention_per_step = [attention_weights]
        for t in range(20):
            scores, decoder_hidden, attention_weights = self.predict_next(
                decoder_hidden, encodings, attention_weights, y_hat_prev
            )

            top = scores.argmax(dim=1)

            y_hat_prev = torch.zeros(batch_size, self.y_size, device=encodings.device)
            y_hat_prev[batch_indices, top] = 1.0
            outputs.append(scores)

            attention_per_step.append(attention_weights)

        return torch.stack(outputs, dim=1), attention_per_step

    def run_inference(self, encodings):
        decoder_hidden = torch.zeros(1, len(encodings), self.hidden_size, device=encodings.device)
        sos = torch.zeros(len(encodings), self.y_size, dtype=encodings.dtype, device=encodings.device)
        sos[:, self.sos_token] = 1
        return self.close_loop_inference(encodings, decoder_hidden, sos)

    def predict_next(self, decoder_hidden, encoder_outputs, prev_attention_weights, y_hat_prev):
        new_attention_weights = self.attention(decoder_hidden, encoder_outputs, prev_attention_weights)
        c = self.attention.compute_context_vectors(new_attention_weights, encoder_outputs)

        v = torch.cat([y_hat_prev, c], dim=1).unsqueeze(1)

        h, hidden = self.decoder_gru(v, decoder_hidden)
        h = h.squeeze(1)
        return self.linear(h), hidden, new_attention_weights


def build_encoder(session, *args, **kwargs):
    return ImageEncoder(*args, **kwargs)


def build_decoder(session):
    encoder_model = next(iter(session.models.values()))
    context_size = encoder_model.hidden_size * 2
    decoder_hidden_size = encoder_model.hidden_size

    tokenizer = session.preprocessors["tokenize"]
    sos_token = tokenizer.char2index[tokenizer.start]
    return AttendingDecoder(sos_token, context_size, y_size=tokenizer.charset_size,
                            hidden_size=decoder_hidden_size)
