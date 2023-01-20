import torch
from torchvision import transforms
from hwr_self_train.augmentation import fit_height

from hwr_self_train.utils import make_tf_batch


class WordRecognitionPipeline:
    def __init__(self, neural_pipeline, tokenizer, image_preprocessor, show_attention=False):
        self.neural_pipeline = neural_pipeline
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        self.show_attention = show_attention

    def __call__(self, images, transcripts=None):
        """Given a list of PIL images and (optionally) corresponding list of text transcripts"""
        image_batch = self.image_preprocessor(images)

        if transcripts is not None:
            transcripts, _ = make_tf_batch(transcripts, self.tokenizer)

        if self.show_attention:
            return self.neural_pipeline.debug_attention(image_batch)
        else:
            return self.neural_pipeline(image_batch, transcripts)


class EncoderDecoder:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def debug_attention(self, image_batch):
        image_batch = image_batch.to(self.device)
        encodings = self.encoder(image_batch)
        return self.decoder.debug_attention(encodings)

    def predict(self, image_batch, transcripts=None):
        image_batch = image_batch.to(self.device)
        if transcripts is not None:
            transcripts = transcripts.to(self.device)

        encodings = self.encoder(image_batch)
        return self.decoder(encodings, transcripts)

    def __call__(self, image_batch, transcripts=None):
        return self.predict(image_batch, transcripts)


class TrainableEncoderDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, device, encoder_optimizer, decoder_optimizer):
        super().__init__(encoder, decoder, device)

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()


class ImageBatchPreprocessor:
    def __init__(self, augment_strategy=None, pad_strategy=None,
                 max_height=None, target_height=None, pad_fill=255):
        self.augment_strategy = augment_strategy or identity
        self.pad_strategy = pad_strategy or one_sided_padding
        self.max_height = max_height
        self.target_height = target_height
        self.fill = pad_fill

    def __call__(self, images):
        to_tensor = transforms.ToTensor()

        images = self.augment_strategy(images)

        if self.max_height:
            images = self.clip_height(images, self.max_height)
        elif self.target_height:
            images = self.to_same_height(images, self.target_height)

        images = self.pad_images(images)
        tensors = [self.to_rgb(to_tensor(im)) for im in images]
        return torch.stack(tensors)

    def clip_height(self, images, max_value):
        """Ensure every image has at most max_value height"""
        res = []
        for im in images:
            image = im if im.height <= max_value else fit_height(im, max_value)
            res.append(image)
        return res

    def to_same_height(self, images, height):
        return [fit_height(im, height) for im in images]

    def to_rgb(self, image):
        return image.repeat(3, 1, 1)

    def pad_images(self, images):
        max_height = self.max_height or max([im.height for im in images])
        max_width = max([im.width for im in images])

        if max_width % 32 != 0:
            max_width = (max_width // 32 + 1) * 32

        padded = []
        for im in images:
            padding_top, padding_bottom = self.pad_strategy(max_height, im.height)
            padding_left, padding_right = self.pad_strategy(max_width, im.width)

            pad = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom),
                                 fill=self.fill)
            padded.append(pad(im))

        return padded


def identity(images):
    return images


def equal_padding(max_length, length):
    len_diff = max_length - length
    padding1 = len_diff // 2
    padding2 = len_diff - padding1
    return padding1, padding2


def one_sided_padding(max_length, length):
    len_diff = max_length - length
    return 0, len_diff
