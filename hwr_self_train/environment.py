import os

from torch.utils.data import DataLoader

from hwr_self_train.utils import collate
from .models import ImageEncoder, AttendingDecoder
from .recognition import WordRecognitionPipeline, TrainableEncoderDecoder
from .datasets import SyntheticOnlineDataset, SyntheticOnlineDatasetCached
from .checkpoints import (
    load_latest_checkpoint,
    make_new_checkpoint,
    save_checkpoint,
    CheckpointsNotFound
)
from .training import TrainingLoop, Trainer
from .history_saver import HistoryCsvSaver
from .evaluation import EvaluationTask

from configuration import tokenizer, Configuration, prepare_metrics, prepare_loss, create_optimizer


def create_neural_pipeline(device):
    encoder = ImageEncoder(image_height=Configuration.image_height,
                           hidden_size=Configuration.hidden_size)

    context_size = encoder.hidden_size * 2
    decoder_hidden_size = encoder.hidden_size

    sos_token = tokenizer.char2index[tokenizer.start]
    decoder = AttendingDecoder(sos_token, context_size, y_size=tokenizer.charset_size,
                               hidden_size=decoder_hidden_size)

    encoder_optimizer = create_optimizer(encoder, Configuration.encoder_optimizer)
    decoder_optimizer = create_optimizer(decoder, Configuration.decoder_optimizer)
    return TrainableEncoderDecoder(
        encoder, decoder, device, encoder_optimizer, decoder_optimizer
    )


def load_or_create_neural_pipeline():
    neural_pipeline = create_neural_pipeline(Configuration.device)

    try:
        return load_latest_checkpoint(neural_pipeline,
                                      Configuration.checkpoints_save_dir,
                                      Configuration.device)
    except CheckpointsNotFound:
        save_dir = make_new_checkpoint(Configuration.checkpoints_save_dir)
        save_checkpoint(neural_pipeline, save_dir, Configuration.device)
        return neural_pipeline


class Environment:
    def __init__(self):
        self.make_checkpoints_dir()

        self.neural_pipeline = load_or_create_neural_pipeline()

        recognizer = WordRecognitionPipeline(self.neural_pipeline, tokenizer)

        loss_fn = prepare_loss(Configuration.loss_function)

        train_metric_fns = prepare_metrics(Configuration.training_metrics)

        val_metric_fns = prepare_metrics(Configuration.validation_metrics)

        training_loader = self.create_data_loader(SyntheticOnlineDataset,
                                                  Configuration.training_set_size)

        val_loader = self.create_data_loader(SyntheticOnlineDatasetCached,
                                             Configuration.validation_set_size)

        trainer = Trainer(recognizer, training_loader, loss_fn, tokenizer)

        self.training_loop = TrainingLoop(trainer, metric_fns=train_metric_fns,
                                          epochs=Configuration.epochs)

        self.history_saver = HistoryCsvSaver(Configuration.history_path)

        eval_on_train = EvaluationTask(recognizer, training_loader, train_metric_fns,
                                       Configuration.evaluation_steps['training_set'])
        eval_on_val = EvaluationTask(recognizer, val_loader, val_metric_fns,
                                     Configuration.evaluation_steps['validation_set'])

        self.eval_tasks = [eval_on_train, eval_on_val]

    def make_checkpoints_dir(self):
        os.makedirs(Configuration.checkpoints_save_dir, exist_ok=True)

    def create_data_loader(self, dataset_class, dataset_size):
        ds = dataset_class(
            Configuration.fonts_dir, dataset_size,
            dict_file=Configuration.dictionary_file, image_height=Configuration.image_height
        )

        return DataLoader(ds, batch_size=Configuration.batch_size,
                          num_workers=Configuration.num_workers, collate_fn=collate)
