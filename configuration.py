import string
import os

import torch
from torch.optim import Adam
from torchmetrics import CharErrorRate

from hwr_self_train.preprocessors import CharacterTokenizer, decode_output_batch
from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.loss_functions import LossTargetTransform
from hwr_self_train.training import get_simple_trainer, get_consistency_trainer
from hwr_self_train.word_samplers import UniformSampler, FrequencyBasedSampler


def get_transform(tokenizer):
    def decode(y_hat, y):
        y_hat = decode_output_batch(y_hat, tokenizer)
        return y_hat, y
    return decode


optimizer_conf = {
    'class': Adam,
    'kwargs': dict(lr=0.0001)
}

letters = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
punctuation = ".,?!:;-()'\""
letters.upper()


class Configuration:
    image_height = 64
    hidden_size = 128

    # todo: add decoder args here (hidden_size, attention_filters, attention_kernel_size)

    charset = letters + letters.upper() + digits + punctuation
    #charset = string.ascii_letters

    iam_pseudo_labels = 'iam/pseudo_labels.txt'
    iam_train_path = 'iam/iam_train.txt'
    iam_dataset_path = 'iam/iam_val.txt'
    confidence_threshold = 0.4

    fonts_dir = './fonts'

    word_sampler = FrequencyBasedSampler.from_file("word_frequencies.csv")

    data_generator_options = dict(
        bg_range=(200, 255),
        color_range=(0, 100),
        font_size_range=(50, 100),
        rotation_range=(0, 0)
    )

    training_set_size = 50000
    validation_set_size = 2500

    batch_size = 32
    num_workers = 0

    tokenizer = CharacterTokenizer(charset)

    loss_conf = {
        'class': MaskedCrossEntropy,
        'kwargs': dict(reduction='sum', label_smoothing=0.6),
        'transform': LossTargetTransform(tokenizer)
    }

    cer_conf = {
        'class': CharErrorRate,
        'transform': get_transform(tokenizer)
    }

    loss_function = loss_conf

    encoder_optimizer = optimizer_conf
    decoder_optimizer = optimizer_conf

    training_metrics = {
        'loss': loss_conf,
        'CER': cer_conf
    }

    # evaluated using the same augmentation used in training dataset
    train_val_metrics = {
        'train-val loss': loss_conf,
        'train-val CER': cer_conf
    }

    # evaluated without using augmentation
    validation_metrics = {
        'val loss': loss_conf,
        'val CER': cer_conf
    }

    # evaluated on IAM dataset (without augmentation)
    test_metrics = {
        'iam loss': loss_conf,
        'iam CER': cer_conf
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    session_dir = 'session'

    evaluation_steps = {
        'training_set': 0.1,
        'train_validation_set': 1.0,
        'validation_set': 1.0,
        'test_set': 0.5
    }
    epochs = 50
    tuning_epochs = 50

    tuning_trainer_factory = get_simple_trainer
    weak_augment_options = dict(
        p_augment=0.4,
        target_height=64,
        fill=255,
        rotation_degrees_range=(-5, 5),
        blur_size=3,
        blur_sigma=[1, 1],
        noise_sigma=10,
        should_fit_height=False
    )
