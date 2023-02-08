import json
import os
import importlib

from torch.utils.data import DataLoader

from hwr_self_train.utils import collate, split_import_path

from .models import build_from_spec

from .recognition import (
    WordRecognitionPipeline,
    TrainableEncoderDecoder
)
from .image_pipelines import make_pretraining_pipeline, make_validation_pipeline
from .datasets import (
    SyntheticOnlineDataset,
    SyntheticOnlineDatasetCached,
    RealUnlabeledDataset,
    RealLabeledDataset
)

from .session import (
    CheckpointKeeper,
    SessionDirectoryLayout,
    CheckpointsNotFound
)
from .training import (
    TrainingLoop,
    Trainer,
    get_simple_trainer,
    get_consistency_trainer
)
from .history_saver import HistoryCsvSaver
from .evaluation import EvaluationTask

from .config_utils import (
    prepare_metrics,
    prepare_loss,
    create_optimizer
)

from .configuration import Configuration

from .tokenizer import CharacterTokenizer


def create_neural_pipeline(device, model_spec, config):
    encoder, decoder = build_from_spec(model_spec)

    encoder_optimizer = create_optimizer(encoder, config.encoder_optimizer)
    decoder_optimizer = create_optimizer(decoder, config.decoder_optimizer)
    return TrainableEncoderDecoder(
        encoder, decoder, device, encoder_optimizer, decoder_optimizer
    )


def load_neural_pipeline(config, session_layout):
    """Instantiate encoder-decoder model and restore it from checkpoint if it exists,
    otherwise create checkpoint.

    Returns encoder-decoder model
    """

    spec = load_model_spec(session_layout)

    neural_pipeline = create_neural_pipeline(config.device, spec, config)

    keeper = CheckpointKeeper(session_layout.checkpoints)

    keeper.load_latest_checkpoint(neural_pipeline, config.device)
    return neural_pipeline


def load_model_spec(session_layout):
    with open(session_layout.model_spec) as f:
        s = f.read()
        return json.loads(s)


class Environment:
    def __init__(self, config: Configuration):
        self.config = config
        session_layout = SessionDirectoryLayout(config.session_dir)

        self.session_layout = session_layout

        tokenizer = CharacterTokenizer(config.charset)

        neural_pipeline = load_neural_pipeline(config, session_layout)

        image_pipeline = make_pretraining_pipeline(
            augmentation_options=config.weak_augment_options,
            max_width=config.max_image_width,
            max_height=config.image_height
        )
        recognizer = WordRecognitionPipeline(neural_pipeline, tokenizer, image_pipeline)

        loss_fn = prepare_loss(config.loss_function)

        train_metric_fns = prepare_metrics(config.training_metrics)
        train_val_metric_fns = prepare_metrics(config.train_val_metrics)
        val_metric_fns = prepare_metrics(config.validation_metrics)
        test_metric_fns = prepare_metrics(config.test_metrics)

        training_loader = self._create_data_loader(SyntheticOnlineDataset,
                                                   config.training_set_size)

        val_loader = self._create_data_loader(SyntheticOnlineDatasetCached,
                                              config.validation_set_size)

        test_ds_path = os.path.join(config.tuning_data_dir, "labeled")
        test_ds = RealLabeledDataset(test_ds_path)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                                 num_workers=config.num_workers, collate_fn=collate)

        trainer = Trainer(recognizer, training_loader, loss_fn, tokenizer)

        epochs_trained = self._get_trained_epochs()

        training_loop = TrainingLoop(trainer, metric_fns=train_metric_fns,
                                     epochs=config.epochs,
                                     starting_epoch=epochs_trained + 1)

        eval_on_train = EvaluationTask(recognizer, training_loader, train_metric_fns,
                                       config.evaluation_steps['training_set'],
                                       close_loop_prediction=False)

        eval_on_train_val = EvaluationTask(recognizer, val_loader, train_val_metric_fns,
                                           config.evaluation_steps['train_validation_set'],
                                           close_loop_prediction=False)

        val_preprocessor = make_validation_pipeline(max_width=config.max_image_width,
                                                    max_height=config.image_height)
        val_recognizer = WordRecognitionPipeline(neural_pipeline, tokenizer, val_preprocessor)
        eval_on_val = EvaluationTask(val_recognizer, val_loader, val_metric_fns,
                                     config.evaluation_steps['validation_set'],
                                     close_loop_prediction=True)

        eval_on_test = EvaluationTask(val_recognizer, test_loader, test_metric_fns,
                                      num_batches=config.evaluation_steps['test_set'],
                                      close_loop_prediction=True)

        self.neural_pipeline = neural_pipeline
        self.epochs_trained = epochs_trained
        self.training_loop = training_loop
        self.history_saver = HistoryCsvSaver(session_layout.history)
        self.eval_tasks = [eval_on_train, eval_on_train_val, eval_on_val, eval_on_test]

    def save_checkpoint(self, epoch, metrics):
        keeper = CheckpointKeeper(self.session_layout.checkpoints)
        keeper.make_new_checkpoint(self.neural_pipeline, self.config.device, epoch, metrics)

    def _get_trained_epochs(self):
        try:
            keeper = CheckpointKeeper(self.session_layout.checkpoints)
            meta_data = keeper.get_latest_meta_data()
            return meta_data["epoch"]
        except CheckpointsNotFound:
            return 0

    def _create_data_loader(self, dataset_class, dataset_size):
        module_path, class_name = split_import_path(self.config.word_sampler)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        sampler = cls.from_file(self.config.sampler_data_file)
        ds = dataset_class(
            self.config.fonts_dir, dataset_size,
            word_sampler=sampler,
            **self.config.data_generator_options
        )

        return DataLoader(ds, batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers, collate_fn=collate)


class TuningEnvironment:
    def __init__(self, config: Configuration):
        self.config = config
        test_ds_path = os.path.join(config.tuning_data_dir, "labeled")
        test_ds = RealLabeledDataset(test_ds_path)
        test_loader = self._create_loader(test_ds)

        unlabeled_ds_path = os.path.join(config.tuning_data_dir, "unlabeled")
        unlabeled_ds = RealUnlabeledDataset(unlabeled_ds_path)
        unlabeled_loader = self._create_loader(unlabeled_ds)

        session_layout = SessionDirectoryLayout(config.session_dir)
        session_layout.create_tuning_checkpoint()
        self.session_layout = session_layout

        tokenizer = CharacterTokenizer(config.charset)

        model_spec = load_model_spec(session_layout)

        encoder_decoder = create_neural_pipeline(config.device, model_spec, config)
        keeper = CheckpointKeeper(session_layout.tuning_checkpoints)
        keeper.load_latest_checkpoint(encoder_decoder, config.device)

        image_preprocessor = make_validation_pipeline(max_width=config.max_image_width,
                                                      max_height=config.image_height)
        recognizer = WordRecognitionPipeline(encoder_decoder, tokenizer, image_preprocessor)

        metric_fns = prepare_metrics(config.training_metrics)

        test_metrics = prepare_metrics(config.test_metrics)

        eval_steps = config.evaluation_steps

        eval_on_test_ds = EvaluationTask(recognizer, test_loader, test_metrics,
                                         num_batches=eval_steps["test_set"],
                                         close_loop_prediction=True)

        self.unlabeled_loader = unlabeled_loader
        self.tokenizer = tokenizer
        self.threshold = config.confidence_threshold
        self.tuning_epochs = config.tuning_epochs
        self.neural_pipeline = encoder_decoder
        self.recognizer = recognizer
        self.metric_fns = metric_fns

        self.tasks = [eval_on_test_ds]

        self.history_saver = HistoryCsvSaver(session_layout.tuning_history)

    def _create_loader(self, ds):
        return DataLoader(ds,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=collate)

    def get_trainer(self, pseudo_labeled_ds):
        loader = self._create_loader(pseudo_labeled_ds)

        loss_fn = prepare_loss(self.config.loss_function)

        tokenizer = CharacterTokenizer(self.config.charset)

        if self.config.tuning_trainer_factory == "simple_trainer":
            trainer_factory = get_simple_trainer
        else:
            trainer_factory = get_consistency_trainer

        return trainer_factory(
            self.recognizer, loader, loss_fn, tokenizer,
            **self.config.weak_augment_options)

    def save_checkpoint(self, epoch, metrics):
        keeper = CheckpointKeeper(self.session_layout.tuning_checkpoints)
        keeper.make_new_checkpoint(self.neural_pipeline, self.config.device, epoch, metrics)

    def get_trained_epochs(self):
        keeper = CheckpointKeeper(self.session_layout.tuning_checkpoints)
        meta_data = keeper.get_latest_meta_data()
        return meta_data["epoch"]
