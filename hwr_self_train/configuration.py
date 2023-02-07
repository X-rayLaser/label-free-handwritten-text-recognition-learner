import string
import json
import torch
from hwr_self_train.utils import full_class_name


optimizer_conf = {
    'class': 'torch.optim.Adam',
    'kwargs': dict(lr=0.0001)
}

charset = string.ascii_letters

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Configuration:
    def __init__(self):
        self.image_height = 64
        self.hidden_size = 128

        self.decoder_params = dict(
            decoder_hidden_size=self.hidden_size,
            filters=10,
            kernel_size=5
        )

        self.charset = string.ascii_letters

        self.fonts_dir = './fonts'

        self.data_generator_options = dict(
            bg_range=(200, 255),
            color_range=(0, 100),
            font_size_range=(50, 100),
            rotation_range=(0, 0)
        )

        self.training_set_size = 50000
        self.validation_set_size = 2500

        self.batch_size = 32
        self.num_workers = 0

        self.loss_conf = loss_conf

        self.cer_conf = cer_conf

        self.loss_function = loss_conf

        self.encoder_optimizer = optimizer_conf
        self.decoder_optimizer = optimizer_conf

        self.training_metrics = {
            'loss': loss_conf,
            'CER': cer_conf
        }

        # evaluated using the same augmentation used in training dataset
        self.train_val_metrics = {
            'train-val loss': loss_conf,
            'train-val CER': cer_conf
        }

        # evaluated without using augmentation
        self.validation_metrics = {
            'val loss': loss_conf,
            'val CER': cer_conf
        }

        # evaluated on test dataset with distribution close to one seen in production (without augmentation)
        self.test_metrics = {
            'test loss': loss_conf,
            'test CER': cer_conf
        }

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.session_dir = 'session'

        self.evaluation_steps = {
            'training_set': 0.1,
            'train_validation_set': 1.0,
            'validation_set': 1.0,
            'test_set': 0.5
        }
        self.epochs = 50
        self.tuning_epochs = 50

        self.word_sampler = 'hwr_self_train.word_samplers.FrequencyBasedSampler'
        self.word_frequencies = "word_frequencies.csv"

        self.tuning_data_dir = 'tuning_data'
        self.confidence_threshold = 0.4

        self.tuning_trainer_factory = "simple_trainer"

        self.weak_augment_options = dict(
            p_augment=0.4,
            target_height=64,
            fill=255,
            rotation_degrees_range=(-5, 5),
            blur_size=3,
            blur_sigma=(1, 1),
            noise_sigma=10,
            should_fit_height=False
        )

    def to_json(self):
        d = {
            'class': full_class_name(self),
            'fields': self.__dict__
        }
        return json.dumps(d)
