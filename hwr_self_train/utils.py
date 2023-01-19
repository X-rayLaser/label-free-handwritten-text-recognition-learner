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

        images = self.pad_images(images)
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
    """Convert raw transcripts into tensor used by decoder network for teacher forcing"""
    transcripts = prepare_tf_seqs(transcripts, tokenizer)
    filler = tokenizer._encode(tokenizer.end)
    padded, mask = pad_sequences(transcripts, filler)
    tf_batch = one_hot_tensor(padded, tokenizer.charset_size)
    return tf_batch, mask


def one_hot_tensor(classes, num_classes):
    """Form a 1-hot tensor from a list of class sequences

    :param classes: list of class sequences (list of lists)
    :param num_classes: total number of available classes
    :return: torch.tensor of shape (batch_size, max_seq_len, num_classes)
    :raise ValueError: if there is any class value that is negative or >= num_classes
    """

    if not hasattr(one_hot_tensor, 'eye'):
        one_hot_tensor.eye = {}

    if num_classes not in one_hot_tensor.eye:
        one_hot_tensor.eye[num_classes] = torch.eye(num_classes, dtype=torch.float32)
    eye = one_hot_tensor.eye[num_classes]
    try:
        tensors = [eye[class_seq] for class_seq in classes]
    except IndexError:
        msg = f'Every class must be a non-negative number less than num_classes={num_classes}. ' \
              f'Got classes {classes}'
        raise ValueError(msg)

    return torch.stack(tensors)


def collate(batch):
    """Split batch into a tuple of lists"""

    num_vars = len(batch[0])
    res = [[] for _ in range(num_vars)]

    for example in batch:
        for i, inp in enumerate(example):
            res[i].append(inp)

    return res
