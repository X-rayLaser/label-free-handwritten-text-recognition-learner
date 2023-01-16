import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate

from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.models import ImageEncoder, AttendingDecoder
from hwr_self_train.preprocessors import CharacterTokenizer
from hwr_self_train.history_saver import HistoryCsvSaver
from hwr_self_train.evaluation import evaluate, EvaluationTask
from hwr_self_train.metrics import Metric
from hwr_self_train.training import TrainableEncoderDecoder, WordRecognitionPipeline, Trainer, \
    TrainingLoop, print_metrics
from hwr_self_train.utils import pad_sequences
from hwr_self_train.datasets import SyntheticOnlineDataset, SyntheticOnlineDatasetCached


def pad_targets(*args):
    *y_hat, ground_true = args
    filler = ground_true[0][-1]
    seqs, mask = pad_sequences(ground_true, filler)
    target = torch.LongTensor(seqs)
    return y_hat + [target] + [mask]


if __name__ == '__main__':
    encoder = ImageEncoder(image_height=64, hidden_size=128)

    context_size = encoder.hidden_size * 2
    decoder_hidden_size = encoder.hidden_size

    tokenizer = CharacterTokenizer()
    sos_token = tokenizer.char2index[tokenizer.start]
    decoder = AttendingDecoder(sos_token, context_size, y_size=tokenizer.charset_size,
                               hidden_size=decoder_hidden_size)

    encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
    decoder_optimizer = Adam(decoder.parameters(), 0.0001)
    neural_pipeline = TrainableEncoderDecoder(encoder, decoder, encoder_optimizer, decoder_optimizer)

    recognizer = WordRecognitionPipeline(neural_pipeline, tokenizer)

    criterion = MaskedCrossEntropy(reduction='sum', label_smoothing=0.6)

    loss_fn = Metric('loss', metric_fn=criterion, metric_args=["y_hat", "y"], transform_fn=pad_targets)

    val_loss_fn = Metric('val loss', metric_fn=criterion, metric_args=["y_hat", "y"], transform_fn=pad_targets)

    def decode(y_hat, y):
        return y_hat, y

    cer = CharErrorRate()

    cer_metric = Metric('CER', metric_fn=cer, metric_args=["y_hat", "y"], transform_fn=decode)

    val_cer = CharErrorRate()

    val_cer_metric = Metric('CER', metric_fn=val_cer, metric_args=["y_hat", "y"], transform_fn=decode)

    metric_functions = {
        'loss': loss_fn,
        'val loss': val_loss_fn,
        'CER': cer_metric,
        'val CER': val_cer_metric
    }

    training_ds = SyntheticOnlineDataset('./fonts', 100, image_height=64)

    val_ds = SyntheticOnlineDatasetCached('./fonts', 100, image_height=64)

    training_loader = DataLoader(training_ds, batch_size=8, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2)
    trainer = Trainer(recognizer, training_loader, loss_fn, tokenizer)

    training_loop = TrainingLoop(trainer, metric_fns=[], epochs=100)

    history_saver = HistoryCsvSaver("history.csv")
    task = EvaluationTask(recognizer, training_loader, metric_functions)
    for epoch in training_loop:
        metrics = evaluate(task,)
        print_metrics(metrics, epoch)
        history_saver.add_entry(epoch, metrics)

    print("Done")
