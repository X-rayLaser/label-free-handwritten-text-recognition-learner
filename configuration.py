import string

import torch


optimizer_conf = {
    'class': 'torch.optim.Adam',
    'kwargs': dict(lr=0.0001)
}


class Configuration:
    image_height = 64
    hidden_size = 128

    decoder_params = dict(
        hidden_size=128,
        filters=10,
        kernel_size=5
    )

    charset = string.ascii_letters

    fonts_dir = './fonts'

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

    loss_conf = {
        'class': 'hwr_self_train.loss_functions.MaskedCrossEntropy',
        'kwargs': dict(reduction='sum', label_smoothing=0.6),
        'transform': {
            'class': 'hwr_self_train.loss_functions.LossTargetTransform',
            'kwargs': dict(charset=charset)
        }
    }

    cer_conf = {
        'class': 'torchmetrics.CharErrorRate',
        'transform': {
            'class': 'hwr_self_train.decoding.DecodeBatchTransform',
            'kwargs': dict(charset=charset)
        }
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

    # evaluated on test dataset with distribution close to one seen in production (without augmentation)
    test_metrics = {
        'test loss': loss_conf,
        'test CER': cer_conf
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

    word_sampler = 'hwr_self_train.word_samplers.FrequencyBasedSampler'
    word_frequencies = "word_frequencies.csv"

    tuning_data_dir = 'tuning_data'
    confidence_threshold = 0.4

    tuning_trainer_factory = "simple_trainer"
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
