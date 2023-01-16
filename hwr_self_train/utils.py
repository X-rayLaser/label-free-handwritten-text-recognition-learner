import torch
from torchvision import transforms


class Mask:
    def __init__(self, lengths, max_length):
        self.mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
        self.lengths = lengths

        for i, length in enumerate(lengths):
            self.mask[i, :length] = True

    @property
    def device(self):
        return self.mask.device

    def to(self, device):
        self.mask = self.mask.to(device)
        return self


def add_padding(seq, size, filler):
    seq = list(seq)
    while len(seq) < size:
        seq.append(filler)
    return seq


def pad_sequences(seqs, filler):
    lengths = [len(seq) for seq in seqs]
    max_length = max(lengths)

    mask = Mask(lengths, max_length)

    padded = [add_padding(seq, max_length, filler) for seq in seqs]
    return padded, mask


def prepare_tf_seqs(transcripts, tokenizer):
    """Prepare a sequence of tokens used for training a decoder with teacher forcing.

    It tokenizes each string, adds a special <start of word> token.

    :param transcripts: strings representing transcripts of word images
    :param tokenizer: tokenizer
    :return: list of integers
    """
    return [tokenizer(t)[:-1] for t in transcripts]


def prepare_targets(transcripts, tokenizer):
    """Converts raw strings into sequences of tokens to be used in loss calculation.

    It tokenizes each string, adds a special <end of word> token.

    :param transcripts: strings representing transcripts of word images
    :param tokenizer: tokenizer
    :return: list of integers
    """
    return [tokenizer(t)[1:] for t in transcripts]


class ImageBatchPreprocessor:
    def __init__(self, pad_strategy=None):
        self.pad_strategy = pad_strategy or one_sided_padding

    def __call__(self, images):
        to_tensor = transforms.ToTensor()
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        tensors = [self.to_rgb(to_tensor(im)) for im in images]
        return torch.stack(tensors)

    def to_rgb(self, image):
        return image.repeat(3, 1, 1)

    def pad_images(self, images):
        max_height = max([im.height for im in images])
        max_width = max([im.width for im in images])

        if max_width % 32 != 0:
            max_width = (max_width // 32 + 1) * 32

        padded = []
        for im in images:
            padding_top, padding_bottom = self.pad_strategy(max_height, im.height)
            padding_left, padding_right = self.pad_strategy(max_width, im.width)

            pad = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=255)
            padded.append(pad(im))

        return padded


def equal_padding(max_length, length):
    len_diff = max_length - length
    padding1 = len_diff // 2
    padding2 = len_diff - padding1
    return padding1, padding2


def one_sided_padding(max_length, length):
    len_diff = max_length - length
    return 0, len_diff


def make_tf_batch(transcripts, tokenizer):
    transcripts = prepare_tf_seqs(transcripts, tokenizer)
    return pad_sequences(transcripts, tokenizer.end_of_word)
