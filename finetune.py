from torch.nn.functional import softmax
from hwr_self_train.training import TrainingLoop
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import TuningEnvironment


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
    with open(pseudo_labels_path, "w") as f:
        pass


def re_build_index(dataset):
    dataset.re_build()


def predict_labels(data_loader, recognizer, predictor):
    for path, grey_level, images in data_loader:
        y_hat = recognizer(images)
        predictor(path, grey_level, y_hat)


def train_on_pseudo_labels(trainer, metric_fns, tasks, history_saver, epoch):
    training_loop = TrainingLoop(trainer, metric_fns, epochs=1, starting_epoch=epoch)
    next(iter(training_loop))

    metrics = {}
    for task in tasks:
        metrics.update(evaluate(task))

    print_metrics(metrics, epoch)
    history_saver.add_entry(epoch, metrics)
    #env.save_checkpoint(epoch, metrics)


if __name__ == '__main__':
    env = TuningEnvironment()

    predictor = PseudoLabelPredictor(env.tokenizer, env.threshold, env.pseudo_labels_path)

    for epoch in range(env.tuning_epochs):
        clear_pseudo_labels(env.pseudo_labels_path)
        predict_labels(env.unlabeled_loader, env.recognizer, predictor)
        trainer = env.get_trainer()
        train_on_pseudo_labels(trainer, env.metric_fns, env.tasks, env.history_saver, epoch)
