from dataclasses import dataclass
from typing import Any, List, AnyStr

import torch
from torch.utils.data import DataLoader

from hwr_self_train.formatters import Formatter
from hwr_self_train.metrics import MetricsSetCalculator, Metric
from hwr_self_train.recognition import WordRecognitionPipeline
from hwr_self_train.preprocessors import CharacterTokenizer
from hwr_self_train.augmentation import WeakAugmentation, StrongAugmentation


def print_metrics(computed_metrics, epoch):
    formatter = Formatter()

    final_metrics_string = formatter.format_metrics(computed_metrics)

    epoch_str = formatter.format_epoch(epoch)

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}\r{epoch_str} {final_metrics_string}')


@dataclass
class IterationLogEntry:
    iteration: int
    num_iterations: int
    outputs: torch.Tensor
    targets: List[AnyStr]
    loss: Any


@dataclass
class BaseTrainer:
    recognizer: WordRecognitionPipeline
    data_loader: DataLoader
    loss_fn: Metric
    tokenizer: CharacterTokenizer
    supress_errors: bool = True

    def __iter__(self):
        num_iterations = len(self.data_loader)
        self.recognizer.neural_pipeline.train_mode()

        for i, (images, transcripts) in enumerate(self.data_loader):
            try:
                loss, result = self.train_one_iteration(images, transcripts)
            except torch.cuda.OutOfMemoryError:
                if not self.supress_errors:
                    raise
                continue

            outputs = result
            targets = transcripts
            yield IterationLogEntry(i, num_iterations, outputs, targets, loss)

    def train_one_iteration(self, images, transcripts):
        loss, y_hat = self.compute_loss(images, transcripts)

        # invoke zero_grad for each neural network
        self.recognizer.neural_pipeline.zero_grad()
        loss.backward()

        # todo: clip gradients

        # invoke optimizer.step() for every neural network if there is one
        self.recognizer.neural_pipeline.step()

        return loss, y_hat

    def compute_loss(self, images, transcripts):
        raise NotImplemented


class Trainer(BaseTrainer):
    def compute_loss(self, images, transcripts):
        y_hat = self.recognizer(images, transcripts)
        loss = self.loss_fn(y_hat=y_hat, y=transcripts)
        return loss, y_hat


class ConsistencyTrainer(BaseTrainer):
    def __init__(self, recognizer, data_loader, loss_fn, tokenizer, weak_augment, strong_augment):
        super().__init__(recognizer, data_loader, loss_fn, tokenizer)
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment

    def compute_loss(self, images, transcripts):
        y_weak = self.recognizer(self.weak_augment(images), transcripts)
        y_strong = self.recognizer(self.strong_augment(images), transcripts)

        loss_weak = self.loss_fn(y_hat=y_weak, y=transcripts)
        loss_strong = self.loss_fn(y_hat=y_strong, y=transcripts)
        loss = loss_weak + loss_strong
        return loss, y_weak


@dataclass
class TrainingLoop:
    trainer: BaseTrainer
    metric_fns: dict
    epochs: int
    starting_epoch: int

    def __iter__(self):
        for epoch in range(self.starting_epoch, self.epochs + 1):
            calculator = MetricsSetCalculator(self.metric_fns, interval=100)

            for iteration_log in self.trainer:
                running_metrics = calculator(iteration_log)
                self.print_metrics(epoch, iteration_log, running_metrics)

            yield epoch

    def print_metrics(self, epoch, iteration_log: IterationLogEntry, running_metrics):
        formatter = Formatter()

        stats = formatter(
            epoch, iteration_log.iteration + 1,
            iteration_log.num_iterations, running_metrics
        )
        print(f'\r{stats}', end='')


def get_simple_trainer(recognizer, loader, loss_fn, tokenizer, **kwargs):
    return Trainer(recognizer, loader, loss_fn, tokenizer)


def get_consistency_trainer(recognizer, loader, loss_fn, tokenizer,
                               **weak_augment_options):
    weak_augment = WeakAugmentation(**weak_augment_options)
    strong_augment = StrongAugmentation()
    return ConsistencyTrainer(recognizer, loader, loss_fn, tokenizer,
                              weak_augment, strong_augment)
