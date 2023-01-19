import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate

from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.models import ImageEncoder, AttendingDecoder
from hwr_self_train.preprocessors import CharacterTokenizer, decode_output_batch
from hwr_self_train.history_saver import HistoryCsvSaver
from hwr_self_train.evaluation import evaluate, EvaluationTask
from hwr_self_train.metrics import Metric
from hwr_self_train.training import TrainableEncoderDecoder, WordRecognitionPipeline, Trainer, \
    TrainingLoop, print_metrics
from hwr_self_train.utils import LossTargetTransform, collate
from hwr_self_train.datasets import SyntheticOnlineDataset, SyntheticOnlineDatasetCached
from hwr_self_train.checkpoints import save_checkpoint, load_checkpoint


def create_metric(name, metric_fn, transform_fn):
    return Metric(
        name, metric_fn=metric_fn, metric_args=["y_hat", "y"], transform_fn=transform_fn
    )


def make_new_checkpoint(base_dir):
    try:
        highest = get_highest_checkpoint_number(base_dir)
    except CheckpointsNotFound:
        highest = 0

    checkpoint_name = str(highest + 1)
    checkpoint_path = os.path.join(base_dir, checkpoint_name)
    os.makedirs(checkpoint_path)
    return checkpoint_path


def load_latest_checkpoint(checkpoints_dir, device):
    highest = get_highest_checkpoint_number(checkpoints_dir)
    highest_dir = os.path.join(checkpoints_dir, str(highest))
    return load_checkpoint(highest_dir, device)


def get_highest_checkpoint_number(checkpoints_dir):
    checkpoints = []
    for folder in os.listdir(checkpoints_dir):
        try:
            checkpoints.append(int(folder))
        except ValueError:
            pass

    if not checkpoints:
        raise CheckpointsNotFound()

    return max(checkpoints)


class CheckpointsNotFound(Exception):
    """Raised when trying to load a checkpoint from a folder containing none of them"""


def create_neural_pipeline(tokenizer, device):
    encoder = ImageEncoder(image_height=64, hidden_size=32)

    context_size = encoder.hidden_size * 2
    decoder_hidden_size = encoder.hidden_size

    sos_token = tokenizer.char2index[tokenizer.start]
    decoder = AttendingDecoder(sos_token, context_size, y_size=tokenizer.charset_size,
                               hidden_size=decoder_hidden_size)

    encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
    decoder_optimizer = Adam(decoder.parameters(), lr=0.0001)
    return TrainableEncoderDecoder(encoder, decoder, device, encoder_optimizer, decoder_optimizer)


if __name__ == '__main__':
    save_base_dir = 'checkpoints'
    device = torch.device('cpu')
    os.makedirs(save_base_dir, exist_ok=True)

    tokenizer = CharacterTokenizer()

    try:
        neural_pipeline = load_latest_checkpoint(save_base_dir, device)
    except CheckpointsNotFound:
        neural_pipeline = create_neural_pipeline(tokenizer, device)

    recognizer = WordRecognitionPipeline(neural_pipeline, tokenizer)

    criterion = MaskedCrossEntropy(reduction='sum', label_smoothing=0.6)
    val_criterion = MaskedCrossEntropy(reduction='sum', label_smoothing=0.6)

    loss_transform = LossTargetTransform(tokenizer)

    loss_fn = create_metric('loss', criterion, loss_transform)
    val_loss_fn = create_metric('val loss', val_criterion, loss_transform)

    def decode(y_hat, y):
        y_hat = decode_output_batch(y_hat, tokenizer)
        return y_hat, y

    cer = CharErrorRate()
    cer_metric = create_metric('CER', cer, decode)

    val_cer = CharErrorRate()
    val_cer_metric = create_metric('val CER', val_cer, decode)

    train_metric_fns = {
        'loss': loss_fn,
        'CER': cer_metric
    }

    val_metric_fns = {
        'val loss': val_loss_fn,
        'val CER': val_cer_metric
    }

    training_ds = SyntheticOnlineDataset(
        './fonts', 100, dict_file='words.txt', image_height=64
    )

    val_ds = SyntheticOnlineDatasetCached(
        './fonts', 100, dict_file='words.txt', image_height=64
    )

    training_loader = DataLoader(training_ds, batch_size=8, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, collate_fn=collate)
    trainer = Trainer(recognizer, training_loader, loss_fn, tokenizer)

    training_loop = TrainingLoop(trainer, metric_fns=train_metric_fns, epochs=100)

    history_saver = HistoryCsvSaver("history.csv")

    eval_on_train = EvaluationTask(recognizer, training_loader, train_metric_fns, 0.1)
    eval_on_eval = EvaluationTask(recognizer, val_loader, val_metric_fns, 1.0)

    for epoch in training_loop:
        metrics = evaluate(eval_on_train)
        metrics.update(evaluate(eval_on_eval))

        print_metrics(metrics, epoch)
        history_saver.add_entry(epoch, metrics)

        save_dir = make_new_checkpoint(save_base_dir)
        save_checkpoint(recognizer.neural_pipeline, save_dir, device)
    print("Done")
