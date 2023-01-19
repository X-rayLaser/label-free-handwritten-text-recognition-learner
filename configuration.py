import torch
from torch.optim import Adam
from torchmetrics import CharErrorRate

from hwr_self_train.preprocessors import CharacterTokenizer, decode_output_batch
from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.loss_functions import LossTargetTransform
from hwr_self_train.metrics import Metric

tokenizer = CharacterTokenizer()


def decode(y_hat, y):
    y_hat = decode_output_batch(y_hat, tokenizer)
    return y_hat, y


loss_conf = {
    'class': MaskedCrossEntropy,
    'kwargs': dict(reduction='sum', label_smoothing=0.6),
    'transform': LossTargetTransform(tokenizer)
}


cer_conf = {
    'class': CharErrorRate,
    'transform': decode
}


optimizer_conf = {
    'class': Adam,
    'kwargs': dict(lr=0.0001)
}


class Configuration:
    image_height = 64
    hidden_size = 32

    fonts_dir = './fonts'
    dictionary_file = 'words.txt'
    training_set_size = 4
    validation_set_size = 2

    batch_size = 2
    num_workers = 2

    loss_function = loss_conf

    encoder_optimizer = optimizer_conf
    decoder_optimizer = optimizer_conf

    training_metrics = {
        'loss': loss_conf,
        'CER': cer_conf
    }

    validation_metrics = {
        'val loss': loss_conf,
        'val CER': cer_conf
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_save_dir = 'checkpoints'
    history_path = 'pretrain_history.csv'

    evaluation_steps = {
        'training_set': 0.1,
        'validation_set': 1.0
    }
    epochs = 5


def create_metric(name, metric_fn, transform_fn):
    return Metric(
        name, metric_fn=metric_fn, metric_args=["y_hat", "y"], transform_fn=transform_fn
    )


def create_optimizer(model, optimizer_conf):
    optimizer_class = optimizer_conf['class']
    kwargs = optimizer_conf['kwargs']
    return optimizer_class(model.parameters(), **kwargs)


def prepare_loss(loss_conf):
    loss_class = loss_conf["class"]
    loss_kwargs = loss_conf["kwargs"]
    loss_transform = loss_conf["transform"]
    loss_function = loss_class(**loss_kwargs)
    return create_metric('loss', loss_function, loss_transform)


def prepare_metrics(metrics_conf):
    metric_fns = {}
    for spec in metrics_conf:
        name = spec['name']
        metric_class = spec['class']
        metric_args = spec.get('args', [])
        metric_kwargs = spec.get('kwargs', {})
        transform_fn = spec['transform']
        metric_fn = metric_class(*metric_args, **metric_kwargs)

        metric_fns[name] = create_metric(name, metric_fn, transform_fn)

    return metric_fns
