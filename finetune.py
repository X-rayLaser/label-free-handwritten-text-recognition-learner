from dataclasses import dataclass
import torch

from hwr_self_train.training import TrainingLoop
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.formatters import show_progress_bar
from hwr_self_train.preprocessors import decode_and_score


def clear_pseudo_labels(pseudo_labels_path):
    with open(pseudo_labels_path, "w") as _:
        pass


def re_build_index(dataset):
    dataset.re_build()


def predict_labels(data_loader, recognizer, tokenizer):
    recognizer.neural_pipeline.eval_mode()

    for paths, grey_levels, images in show_progress_bar(data_loader,
                                                        desc='Predicting pseudo labels: '):
        y_hat = recognizer(images)
        transcripts, scores = decode_and_score(y_hat, tokenizer)

        yield PseudoLabeledBatch(image_paths=paths, grey_levels=grey_levels,
                                 transcripts=transcripts, scores=scores)


@dataclass
class PseudoLabeledBatch:
    image_paths: list
    grey_levels: list
    transcripts: list
    scores: list

    def save_above_threshold(self, threshold, save_path):
        for path, grey_level, transcript, score in zip(self.image_paths, self.grey_levels,
                                                       self.transcripts, self.scores):
            if score > threshold:
                with open(save_path, 'a') as f:
                    f.write(f'{path}, {grey_level}, {transcript}\n')


def train_on_pseudo_labels(env, epoch):
    trainer = env.get_trainer()
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
        clear_pseudo_labels(env.pseudo_labels_path)

        with torch.no_grad():
            for pseudo_labeled_batch in predict_labels(env.unlabeled_loader,
                                                       env.recognizer,
                                                       env.tokenizer):
                pseudo_labeled_batch.save_above_threshold(
                    env.threshold, env.pseudo_labels_path
                )

        train_on_pseudo_labels(env, epoch)

        metrics = compute_metrics(env)
        print_metrics(metrics, epoch)
        env.history_saver.add_entry(epoch, metrics)
        env.save_checkpoint(epoch, metrics)


if __name__ == '__main__':
    fine_tune()
