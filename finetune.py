from dataclasses import dataclass
import torch

from PIL import Image

from hwr_self_train.training import TrainingLoop
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.formatters import show_progress_bar
from hwr_self_train.preprocessors import decode_and_score
from hwr_self_train.datasets import PseudoLabeledDataset


def predict_labels(data_loader, recognizer, tokenizer):
    recognizer.neural_pipeline.eval_mode()

    for image_paths in show_progress_bar(data_loader, desc='Predicting pseudo labels: '):
        images = [Image.open(path) for path in image_paths]
        y_hat = recognizer(images)
        transcripts, scores = decode_and_score(y_hat, tokenizer)

        yield PseudoLabeledBatch(image_paths=image_paths, transcripts=transcripts, scores=scores)


@dataclass
class PseudoLabeledBatch:
    image_paths: list
    transcripts: list
    scores: list

    def above_threshold(self, threshold):
        for path, transcript, score in zip(self.image_paths, self.transcripts, self.scores):
            if score > threshold:
                yield path, transcript


def train_on_pseudo_labels(env, epoch, pseudo_labeled):
    pseudo_labeled_ds = PseudoLabeledDataset(pseudo_labeled)
    trainer = env.get_trainer(pseudo_labeled_ds)
    training_loop = TrainingLoop(trainer, env.metric_fns, epochs=epoch + 1, starting_epoch=epoch)
    next(iter(training_loop))


def compute_metrics(env):
    metrics = {}
    for task in env.tasks:
        metrics.update(evaluate(task))
    return metrics


def fine_tune():
    env = TuningEnvironment()

    starting_epoch = env.get_trained_epochs() + 1
    for epoch in range(starting_epoch, env.tuning_epochs):
        pseudo_labeled = []
        with torch.no_grad():
            for pseudo_labeled_batch in predict_labels(env.unlabeled_loader,
                                                       env.recognizer,
                                                       env.tokenizer):
                paths_with_transcripts = pseudo_labeled_batch.above_threshold(env.threshold)
                pseudo_labeled.extend(paths_with_transcripts)

        train_on_pseudo_labels(env, epoch, pseudo_labeled)

        metrics = compute_metrics(env)
        print_metrics(metrics, epoch)
        env.history_saver.add_entry(epoch, metrics)
        env.save_checkpoint(epoch, metrics)


if __name__ == '__main__':
    fine_tune()
