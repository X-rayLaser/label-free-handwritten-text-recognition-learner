import torch
from torch.nn.functional import softmax
from hwr_self_train.training import TrainingLoop
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.formatters import ProgressBar


class PseudoLabelPredictor:
    def __init__(self, tokenizer, threshold, pseudo_labels_path):
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.pseudo_labels_path = pseudo_labels_path

    def __call__(self, image_path, gray_level, y_hat):
        pmf = softmax(y_hat, dim=2)
        values, indices = pmf.max(dim=2)

        end_token = self.tokenizer._encode(self.tokenizer.end)
        for i in range(len(indices)):
            tokens = indices[i].tolist()

            try:
                first_n = tokens.index(end_token)
            except ValueError:
                first_n = len(tokens)

            mean_confidence = values[i, :first_n].mean()

            transcript = self.tokenizer.decode_to_string(tokens, clean_output=True)

            if mean_confidence > self.threshold:
                self.save_example(image_path[i], gray_level[i], transcript)

    def save_example(self, image_path, gray_level, transcript):
        with open(self.pseudo_labels_path, 'a') as f:
            f.write(f'{image_path}, {gray_level}, {transcript}\n')


def clear_pseudo_labels(pseudo_labels_path):
    with open(pseudo_labels_path, "w") as _:
        pass


def re_build_index(dataset):
    dataset.re_build()


def predict_labels(data_loader, recognizer, predictor):
    whitespaces = ' ' * 150
    print(f'\r{whitespaces}', end='')

    progress_bar = ProgressBar()
    for i, (path, grey_level, images) in enumerate(data_loader):
        y_hat = recognizer(images)
        predictor(path, grey_level, y_hat)

        step_number = i + 1
        num_batches = len(data_loader)
        progress = progress_bar.updated(step_number, num_batches, cols=50)
        print(f'\rPredicting pseudo labels: {progress} {step_number}/{num_batches}', end='')


def train_on_pseudo_labels(env, epoch):
    trainer = env.get_trainer()
    training_loop = TrainingLoop(trainer, env.metric_fns, epochs=epoch + 1, starting_epoch=epoch)
    next(iter(training_loop))

    metrics = {}
    for task in env.tasks:
        metrics.update(evaluate(task))

    print_metrics(metrics, epoch)
    env.history_saver.add_entry(epoch, metrics)
    env.save_checkpoint(epoch, metrics)


def fine_tune():
    env = TuningEnvironment()

    predictor = PseudoLabelPredictor(env.tokenizer, env.threshold, env.pseudo_labels_path)

    starting_epoch = env.get_trained_epochs() + 1
    for epoch in range(starting_epoch, env.tuning_epochs):
        clear_pseudo_labels(env.pseudo_labels_path)
        with torch.no_grad():
            predict_labels(env.unlabeled_loader, env.recognizer, predictor)

        train_on_pseudo_labels(env, epoch)


if __name__ == '__main__':
    fine_tune()
