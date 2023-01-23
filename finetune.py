from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from hwr_self_train.preprocessors import CharacterTokenizer
from hwr_self_train.recognition import WordRecognitionPipeline
from hwr_self_train.environment import load_or_create_neural_pipeline
from hwr_self_train.image_pipelines import make_validation_pipeline
from configuration import Configuration, prepare_loss, prepare_metrics

from hwr_self_train.datasets import UnlabeledDataset, LabeledDataset
from hwr_self_train.utils import collate
from hwr_self_train.training import TrainingLoop, ConsistencyTrainer
from hwr_self_train.augmentation import WeakAugmentation, StrongAugmentation
from hwr_self_train.evaluation import evaluate, EvaluationTask
from hwr_self_train.training import print_metrics
from hwr_self_train.history_saver import HistoryCsvSaver


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


ds = UnlabeledDataset(Configuration.iam_train_path)
train_loader = DataLoader(ds, batch_size=Configuration.batch_size,
                          num_workers=Configuration.num_workers, collate_fn=collate)

test_ds = LabeledDataset(Configuration.iam_pseudo_labels)
test_loader = DataLoader(test_ds,
                         batch_size=Configuration.batch_size,
                         num_workers=Configuration.num_workers, collate_fn=collate)

tokenizer = CharacterTokenizer()
pseudo_labels_path = Configuration.iam_pseudo_labels

encoder_decoder = load_or_create_neural_pipeline()
image_preprocessor = make_validation_pipeline(max_heights=64)
recognizer = WordRecognitionPipeline(encoder_decoder, tokenizer, image_preprocessor)

predictor = PseudoLabelPredictor(tokenizer, Configuration.confidence_threshold, pseudo_labels_path)

loss_fn = prepare_loss(Configuration.loss_function)

metric_fns = prepare_metrics(Configuration.training_metrics)

weak_augment = WeakAugmentation(**Configuration.weak_augment_options)
strong_augment = StrongAugmentation()
trainer = ConsistencyTrainer(recognizer, train_loader, loss_fn, tokenizer, weak_augment, strong_augment)

test_metrics = []
tasks = [EvaluationTask(recognizer, train_loader, metric_fns, num_batches=0.1),
         EvaluationTask(recognizer, test_loader, test_metrics, num_batches=0.5)]

history_saver = HistoryCsvSaver('tuning_history.csv')


def clear_pseudo_labels(pseudo_labels_path):
    with open(pseudo_labels_path, "w") as f:
        pass


def re_build_index(dataset):
    dataset.re_build()


def predict_labels(data_loader, recognizer):
    for path, grey_level, images in data_loader:
        y_hat = recognizer(images)
        predictor(path, grey_level, y_hat)


def train_on_pseudo_labels(trainer, metric_fns, tasks, epoch):
    training_loop = TrainingLoop(trainer, metric_fns, epochs=1, starting_epoch=1)
    next(iter(training_loop))

    metrics = {}
    for task in tasks:
        metrics.update(evaluate(task))

    print_metrics(metrics, epoch)
    history_saver.add_entry(epoch, metrics)
    #env.save_checkpoint(epoch, metrics)


for epoch in range(50):
    clear_pseudo_labels(pseudo_labels_path)
    re_build_index(ds)
    predict_labels(train_loader, recognizer)
    train_on_pseudo_labels(trainer, metric_fns, tasks, epoch)
