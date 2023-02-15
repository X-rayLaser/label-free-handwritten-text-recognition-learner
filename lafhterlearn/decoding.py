from .tokenizer import CharacterTokenizer
from torch.nn.functional import softmax


def decode_output_batch(tensor, tokenizer):
    """Convert tensor of predictions into a list of strings.

    :param tensor: tensor of shape (batch_size, max_steps, num_classes) containing
    raw (unnormalized) scores representing how likely is a given character at a given step

    :param tokenizer: instance of CharacterTokenizer class
    :return: textual transcripts extracted using predictions tensor
    """

    transcripts, _ = decode_and_score(tensor, tokenizer)
    return transcripts


def decode_and_score(tensor, tokenizer):
    """Convert tensor of predictions into a list of strings and compute confidence scores.

    :param tensor: tensor of shape (batch_size, max_steps, num_classes) containing
    raw (unnormalized) scores representing how likely is a given character at a given step

    :param tokenizer: instance of CharacterTokenizer class
    :return: 2-tuples with transcripts and corresponding confidence score
    """

    pmf = softmax(tensor, dim=2)
    values, indices = pmf.max(dim=2)

    end_token = tokenizer.encode(tokenizer.end)

    all_transcripts = []
    all_scores = []

    for i in range(len(indices)):
        tokens = indices[i].tolist()

        try:
            first_n = tokens.index(end_token)
        except ValueError:
            first_n = len(tokens)

        confidence_score = values[i, :first_n].mean()

        transcript = tokenizer.decode_to_string(tokens, clean_output=True)

        all_transcripts.append(transcript)
        all_scores.append(confidence_score)

    return all_transcripts, all_scores


class DecodeBatchTransform:
    def __init__(self, charset):
        self.tokenizer = CharacterTokenizer(charset)

    def __call__(self, y_hat, y):
        return decode_output_batch(y_hat, self.tokenizer), y
