import torch
from torch import nn

from .utils import prepare_targets, pad_sequences, truncate_sequences
from .tokenizer import CharacterTokenizer


class MaskedCrossEntropy:
    def __init__(self, reduction='mean', label_smoothing=0.0):
        """

        :param reduction: defines which reduction to apply to elementwise losses:
            if set to "mean", mean will be taken across all dimensions,
            if set to "sum", all individual losses across all dimensions will be summed
            if set to 'none', no reduction will be performed
        """
        self.loss_function = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        self.reduction = reduction

    def __call__(self, y_hat, ground_true, mask):
        losses = self.loss_function(self.swap_axes(y_hat), ground_true)

        masked_loss = losses[mask.mask]
        if self.reduction == 'mean':
            return masked_loss.mean()
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return masked_loss

    def swap_axes(self, t):
        return t.transpose(1, 2)


class LossTargetTransform:
    """Transform function (callable) invoked by Metric instance wrapping loss function.

    It takes 2 arguments: predictions and transcripts.
    Prediction is PyTorch tensor of shape (batch_size, max_steps, num_classes)
    containing raw unnormalized probabilities of different classes.
    Transcripts is a list of corresponding transcripts for each prediction in a batch.
    Each transcript is a Python string (str).

    The callable converts all transcripts into token sequences, pads them and
    wraps in PyTorch LongTensor class. It also creates a mask which is
    another PyTorch 2d Tensor which specifies the original (unpadded) length
    of each token sequence.

    The callable keeps predictions tensor unchanged.

    It returns 3-tuple (prediction, targets, mask)
    """
    def __init__(self, charset):
        self.tokenizer = CharacterTokenizer(charset)

    def __call__(self, *args):
        y_hat, transcripts = args

        tokens = prepare_targets(transcripts, self.tokenizer)

        filler = tokens[0][-1]

        prediction_num_steps = y_hat.shape[1]

        max_transcript_len = max(len(token_seq) for token_seq in tokens)

        # todo: consider truncating the tensor along steps dim instead
        #  (to the length of longest transcript)
        if max_transcript_len > prediction_num_steps:
            tokens = truncate_sequences(tokens, prediction_num_steps)

        seqs, mask = pad_sequences(tokens, filler, max_length=prediction_num_steps)
        target = torch.LongTensor(seqs)
        return [y_hat] + [target] + [mask]
