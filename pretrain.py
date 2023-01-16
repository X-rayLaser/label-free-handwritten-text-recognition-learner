import argparse
from collections import namedtuple

import torch
from torch.optim import Adam
from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.formatters import Formatter
from hwr_self_train.models import ImageEncoder, AttendingDecoder
from hwr_self_train.preprocessors import CharacterTokenizer

EpochLog = namedtuple('EpochLog', 'epoch metrics')


class WordRecognitionPipeline:
    training_mode = 1
    inference_mode = 2
    debug_mode = 3

    def __init__(self, neural_pipeline, tokenizer,
                 input_adapter, output_adapter, mode=None):
        self.neural_pipeline = neural_pipeline
        self.mode = mode or self.training_mode

    def __call__(self, images, transcripts=None):
        """Given a list of PIL images and (optionally) corresponding list of text transcripts"""
        image_batch = self.make_images_batch(images)

        if transcripts is not None:
            transcripts = self.make_targets_batch(transcripts)

        if self.mode == self.training_mode:

            y_hat = self.neural_pipeline(image_batch, transcripts)
        elif self.mode == self.inference_mode:
            y_hat = self.neural_pipeline(image_batch)
        elif self.mode == self.debug_mode:
            y_hat, attention = self.neural_pipeline.debug_attention(image_batch)
        else:
            raise Exception()

        # different possible forms of output: raw tensor, integer tokens, strings
        self.output_adapter(y_hat)

    def make_images_batch(self, images):
        return []

    def make_targets_batch(self, transcripts):
        transcripts = prepare_tf_seqs(transcripts, self.tokenizer)
        padded, mask = pad_transcripts(transcripts, self.tokenizer.end_of_word)
        return padded


class EncoderDecoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def debug_attention(self, image_batch):
        encodings = self.encoder(image_batch)
        return self.decoder.debug_attention(encodings)

    def predict(self, image_batch, transcripts=None):
        encodings = self.encoder(image_batch)
        return self.decoder(encodings, transcripts)


class TrainableEncoderDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer):
        super().__init__(encoder, decoder)

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()


def pad_transcripts(token_list, filler):
    return token_list, token_list


def prepare_tf_seqs(transcripts, tokenizer):
    """Prepare a sequence of tokens used for training a decoder with teacher forcing.

    It tokenizes each string, adds a special <start of word> token.

    :param transcripts: strings representing transcripts of word images
    :param tokenizer: tokenizer
    :return: list of integers
    """
    return [tokenizer(t)[:-1] for t in transcripts]


def prepare_targets(transcripts, tokenizer):
    """Converts raw strings into sequences of tokens to be used in loss calculation.

    It tokenizes each string, adds a special <end of word> token.

    :param transcripts: strings representing transcripts of word images
    :param tokenizer: tokenizer
    :return: list of integers
    """
    return [tokenizer(t)[1:] for t in transcripts]


class Metric:
    def __init__(self, name, metric_fn, metric_args, transform_fn):
        pass

    def __call__(self, **batch):
        pass


def train_stage(session, stage_number, log_metrics, save_checkpoint, stat_ivl=10):
    stage = session.stages[stage_number]

    start_epoch = session.progress.epochs_done_total + 1

    metric_dicts = [pipeline.metric_fns for pipeline in stage.training_pipelines]

    debuggers = [Debugger(pipeline) for pipeline in stage.debug_pipelines]

    history = []
    for epoch in range(start_epoch, start_epoch + 1000):
        run_callbacks(session, stage.run_before_epoch, epoch)

        calculators = [MetricsSetCalculator(metrics, stat_ivl) for metrics in metric_dicts]

        train_on_data(session, stage.training_pipelines, debuggers, calculators, epoch)

        computed_metrics = compute_and_log_metrics(stage)
        history.append(computed_metrics)

        log_computed_metrics(computed_metrics, stage_number, epoch, log_metrics)

        session.progress.increment_progress()

        save_checkpoint(epoch)


class TrainingLoop:
    def __init__(self, trainer, metric_fns, epochs):
        self.trainer = trainer
        self.epochs = epochs
        self.metric_fns = metric_fns
        self.formatter = Formatter()

    def __iter__(self):
        for epoch in range(self.epochs):
            calculator = MetricsSetCalculator(self.metric_fns, interval=100)

            for iteration_log in self.trainer:
                running_metrics = calculator(iteration_log)
                self.print_metrics(epoch, iteration_log, running_metrics)

            yield epoch

    def print_metrics(self, epoch, iteration_log, running_metrics):
        stats = self.formatter(
            epoch, iteration_log.iteration + 1,
            iteration_log.num_iterations, running_metrics
        )
        print(f'\r{stats}', end='')


class MetricsSetCalculator:
    def __init__(self, metrics, interval):
        """
        :param metrics: {'name': ('graph_leaf', Metric())}
        """
        self.metrics = metrics
        self.interval = interval
        self.running_metrics = {name: MovingAverage() for name in metrics}

    def __call__(self, iteration_log):
        if iteration_log.iteration % self.interval == 0:
            self.reset()

        with torch.no_grad():
            update_running_metrics(self.running_metrics, self.metrics, iteration_log.outputs)

        return self.running_metrics

    def reset(self):
        for metric_avg in self.running_metrics.values():
            metric_avg.reset()


def update_running_metrics(moving_averages, metrics, results_batch):
    for name, (leaf_name, metric_fn) in metrics.items():
        moving_averages[name].update(
            metric_fn(results_batch[leaf_name])
        )


class MovingAverage:
    def __init__(self):
        self.x = 0
        self.num_updates = 0

    def reset(self):
        self.x = 0
        self.num_updates = 0

    def update(self, x):
        self.x += x
        self.num_updates += 1

    @property
    def value(self):
        return self.x / self.num_updates


class IterationLogEntry:
    def __init__(self, iteration, num_iterations, inputs, outputs, targets, loss):
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.loss = loss


class Trainer:
    def __init__(self, recognizer, data_loader, loss_fn, tokenizer, supress_errors=True):
        self.recognizer = recognizer
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.supress_errors = supress_errors

        # calls train() on every model object in the graph
        self.recognizer.train_mode()

    def __iter__(self):
        num_iterations = len(self.data_loader)
        self.recognizer.train_mode()

        inputs = []
        for i, (images, transcripts) in enumerate(self.data_loader):
            try:
                loss, result = self.train_one_iteration(images, transcripts)
            except torch.cuda.OutOfMemoryError:
                if not self.supress_errors:
                    raise
                continue

            outputs = result
            targets = result
            # todo: fix this (may get rid of inputs and targets)
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, loss)

    def train_one_iteration(self, images, transcripts):
        y_hat = self.recognizer(images, transcripts)
        ground_true = prepare_targets(transcripts, self.tokenizer)
        padded_targets, mask = pad_transcripts(ground_true, filler=self.tokenizer.end_of_word)
        loss = self.loss_fn(y_hat, ground_true, mask)

        # invoke zero_grad for each neural network
        self.recognizer.neural_pipeline.zero_grad()
        loss.backward()

        # todo: clip gradients

        # invoke optimizer.step() for every neural network if there is one
        self.recognizer.neural_pipeline.step()

        return loss, y_hat


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
    input_adapter = None
    output_adapter = None
    recognizer = WordRecognitionPipeline(neural_pipeline, tokenizer, input_adapter, output_adapter)

    criterion = MaskedCrossEntropy(reduction='sum', label_smoothing=0.6)
    transform_pad = None
    loss_fn = Metric('loss', metric_fn=criterion, metric_args=["y_hat", "y"], transform_fn=transform_pad)
    data_loader = None
    trainer = Trainer(recognizer, data_loader, loss_fn, tokenizer)

    training_loop = TrainingLoop(trainer, metric_fns=[], epochs=100)
    from hwr_self_train.history_saver import HistoryCsvSaver
    history_saver = HistoryCsvSaver("history.csv")
    for epoch in training_loop:
        # todo: calculate metrics and show them; save them to csv file; save session
        metrics = {}
        history_saver.add_entry(epoch, metrics)

    print("Done")
