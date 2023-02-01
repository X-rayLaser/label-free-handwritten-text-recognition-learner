import os
import json
import shutil

from torch.utils.data import DataLoader

from hwr_self_train.utils import collate
from .models import ImageEncoder, AttendingDecoder
from .recognition import (
    WordRecognitionPipeline,
    TrainableEncoderDecoder
)
from .image_pipelines import make_pretraining_pipeline, make_validation_pipeline
from .datasets import (
    SyntheticOnlineDataset,
    SyntheticOnlineDatasetCached,
    UnlabeledDataset,
    LabeledDataset
)

from .checkpoints import (
    CheckpointKeeper,
    SessionDirectoryLayout,
    CheckpointsNotFound
)
from .training import TrainingLoop, Trainer
from .history_saver import HistoryCsvSaver
from .evaluation import EvaluationTask

from configuration import Configuration

from .config_utils import (
    prepare_metrics,
    prepare_loss,
    create_optimizer
)


session_layout = SessionDirectoryLayout(Configuration.session_dir)


def create_neural_pipeline(device, tokenizer):
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


def load_or_create_neural_pipeline(tokenizer):
    """Instantiate encoder-decoder model and restore it from checkpoint if it exists,
    otherwise create checkpoint.

    Returns encoder-decoder model
    """
    neural_pipeline = create_neural_pipeline(Configuration.device, tokenizer)

    keeper = CheckpointKeeper(session_layout.checkpoints)

    try:
        keeper.load_latest_checkpoint(neural_pipeline, Configuration.device)
        return neural_pipeline
    except CheckpointsNotFound:
        # since checkpoints do not exist, assume that we start from scratch,
        # therefore we remove existing history file
        session_layout.remove_history()

        neural_pipeline.encoder.to(neural_pipeline.device)
        neural_pipeline.decoder.to(neural_pipeline.device)
        keeper.make_new_checkpoint(neural_pipeline, Configuration.device, 0, metrics={})
        return neural_pipeline


class Environment:
    def __init__(self):
        if not os.path.exists(session_layout.session):
            os.makedirs(session_layout.session)

        self._make_checkpoints_dir()

        tokenizer = Configuration.tokenizer
        self.neural_pipeline = load_or_create_neural_pipeline(tokenizer)

        image_pipeline = make_pretraining_pipeline(
            augmentation_options=Configuration.weak_augment_options,
            max_heights=Configuration.image_height
        )
        recognizer = WordRecognitionPipeline(self.neural_pipeline, tokenizer, image_pipeline)

        loss_fn = prepare_loss(Configuration.loss_function)

        train_metric_fns = prepare_metrics(Configuration.training_metrics)
        train_val_metric_fns = prepare_metrics(Configuration.train_val_metrics)
        val_metric_fns = prepare_metrics(Configuration.validation_metrics)
        test_metric_fns = prepare_metrics(Configuration.test_metrics)

        training_loader = self._create_data_loader(SyntheticOnlineDataset,
                                                   Configuration.training_set_size)

        val_loader = self._create_data_loader(SyntheticOnlineDatasetCached,
                                              Configuration.validation_set_size)

        test_ds = LabeledDataset(Configuration.iam_dataset_path)
        test_loader = DataLoader(test_ds, batch_size=Configuration.batch_size,
                                 num_workers=Configuration.num_workers, collate_fn=collate)

        trainer = Trainer(recognizer, training_loader, loss_fn, tokenizer)

        self.epochs_trained = self._get_trained_epochs()

        self.training_loop = TrainingLoop(trainer, metric_fns=train_metric_fns,
                                          epochs=Configuration.epochs,
                                          starting_epoch=self.epochs_trained + 1)

        self.history_saver = HistoryCsvSaver(session_layout.history)

        eval_on_train = EvaluationTask(recognizer, training_loader, train_metric_fns,
                                       Configuration.evaluation_steps['training_set'],
                                       close_loop_prediction=False)

        eval_on_train_val = EvaluationTask(recognizer, val_loader, train_val_metric_fns,
                                           Configuration.evaluation_steps['train_validation_set'],
                                           close_loop_prediction=False)

        val_preprocessor = make_validation_pipeline(max_heights=Configuration.image_height)
        val_recognizer = WordRecognitionPipeline(self.neural_pipeline, tokenizer, val_preprocessor)
        eval_on_val = EvaluationTask(val_recognizer, val_loader, val_metric_fns,
                                     Configuration.evaluation_steps['validation_set'],
                                     close_loop_prediction=True)

        eval_on_test = EvaluationTask(val_recognizer, test_loader, test_metric_fns,
                                      num_batches=Configuration.evaluation_steps['test_set'],
                                      close_loop_prediction=True)
        self.eval_tasks = [eval_on_train, eval_on_train_val, eval_on_val, eval_on_test]

    def save_checkpoint(self, epoch, metrics):
        keeper = CheckpointKeeper(session_layout.checkpoints)
        keeper.make_new_checkpoint(self.neural_pipeline, Configuration.device, epoch, metrics)

    def _get_trained_epochs(self):
        try:
            keeper = CheckpointKeeper(session_layout.checkpoints)
            meta_data = keeper.get_latest_meta_data()
            return meta_data["epoch"]
        except CheckpointsNotFound:
            return 0

    def _make_checkpoints_dir(self):
        os.makedirs(session_layout.checkpoints, exist_ok=True)

    def _create_data_loader(self, dataset_class, dataset_size):
        ds = dataset_class(
            Configuration.fonts_dir, dataset_size,
            word_sampler=Configuration.word_sampler,
            **Configuration.data_generator_options
        )

        return DataLoader(ds, batch_size=Configuration.batch_size,
                          num_workers=Configuration.num_workers, collate_fn=collate)


class TuningEnvironment:
    def __init__(self):
        pseudo_labeled_dataset = LabeledDataset(Configuration.iam_pseudo_labels)
        pseudo_labeled_loader = self._create_loader(pseudo_labeled_dataset)

        training_dataset = LabeledDataset(Configuration.iam_train_path)
        training_loader = self._create_loader(training_dataset)

        test_ds = LabeledDataset(Configuration.iam_dataset_path)
        test_loader = self._create_loader(test_ds)

        unlabeled_ds = UnlabeledDataset(Configuration.iam_train_path)
        unlabeled_loader = self._create_loader(unlabeled_ds)

        session_layout.create_tuning_checkpoint()

        tokenizer = Configuration.tokenizer
        encoder_decoder = create_neural_pipeline(Configuration.device, tokenizer)
        keeper = CheckpointKeeper(session_layout.tuning_checkpoints)
        keeper.load_latest_checkpoint(encoder_decoder, Configuration.device)

        image_preprocessor = make_validation_pipeline(max_heights=Configuration.image_height)
        recognizer = WordRecognitionPipeline(encoder_decoder, tokenizer, image_preprocessor)

        metric_fns = prepare_metrics(Configuration.training_metrics)

        test_metrics = prepare_metrics(Configuration.test_metrics)

        eval_steps = Configuration.evaluation_steps

        eval_on_train_ds = EvaluationTask(recognizer, training_loader, metric_fns,
                                          num_batches=eval_steps["training_set"],
                                          close_loop_prediction=True)
        eval_on_test_ds = EvaluationTask(recognizer, test_loader, test_metrics,
                                         num_batches=eval_steps["test_set"],
                                         close_loop_prediction=True)

        self.pseudo_labeled_dataset = pseudo_labeled_dataset
        self.pseudo_labeled_loader = pseudo_labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.tokenizer = tokenizer
        self.pseudo_labels_path = Configuration.iam_pseudo_labels
        self.threshold = Configuration.confidence_threshold
        self.tuning_epochs = Configuration.tuning_epochs
        self.neural_pipeline = encoder_decoder
        self.recognizer = recognizer
        self.metric_fns = metric_fns

        self.tasks = [eval_on_train_ds, eval_on_test_ds]

        self.history_saver = HistoryCsvSaver(session_layout.tuning_history)

    def _create_loader(self, ds):
        return DataLoader(ds,
                          batch_size=Configuration.batch_size,
                          num_workers=Configuration.num_workers,
                          collate_fn=collate)

    def get_trainer(self):
        self.pseudo_labeled_dataset.re_build()
        loader = self._create_loader(self.pseudo_labeled_dataset)

        loss_fn = prepare_loss(Configuration.loss_function)

        tokenizer = Configuration.tokenizer

        return Configuration.tuning_trainer_factory(
            self.recognizer, loader, loss_fn, tokenizer,
            **Configuration.weak_augment_options)

    def save_checkpoint(self, epoch, metrics):
        keeper = CheckpointKeeper(session_layout.tuning_checkpoints)
        keeper.make_new_checkpoint(self.neural_pipeline, Configuration.device, epoch, metrics)

    def get_trained_epochs(self):
        keeper = CheckpointKeeper(session_layout.tuning_checkpoints)
        meta_data = keeper.get_latest_meta_data()
        return meta_data["epoch"]
