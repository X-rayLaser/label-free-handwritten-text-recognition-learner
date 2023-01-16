from torch import nn


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
