import torch
from hwr_self_train.formatters import Formatter
from hwr_self_train.utils import pad_sequences, prepare_targets, \
    make_tf_batch, ImageBatchPreprocessor

from hwr_self_train.metrics import MetricsSetCalculator


class WordRecognitionPipeline:
    def __init__(self, neural_pipeline, tokenizer, show_attention=False):
        self.neural_pipeline = neural_pipeline
        self.tokenizer = tokenizer
        self.show_attention = show_attention

    def __call__(self, images, transcripts=None):
        """Given a list of PIL images and (optionally) corresponding list of text transcripts"""
        batch_preprocessor = ImageBatchPreprocessor()
        image_batch = batch_preprocessor(images)

        if transcripts is not None:
            transcripts, _ = make_tf_batch(transcripts, self.tokenizer)

        if self.show_attention:
            return self.neural_pipeline.debug_attention(image_batch)
        else:
            return self.neural_pipeline(image_batch, transcripts)


class EncoderDecoder:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def debug_attention(self, image_batch):
        image_batch = image_batch.to(self.device)
        encodings = self.encoder(image_batch)
        return self.decoder.debug_attention(encodings)

    def predict(self, image_batch, transcripts=None):
        image_batch = image_batch.to(self.device)
        if transcripts:
            transcripts = transcripts.to(self.device)

        encodings = self.encoder(image_batch)
        return self.decoder(encodings, transcripts)

    def __call__(self, image_batch, transcripts=None):
        return self.predict(image_batch, transcripts)


class TrainableEncoderDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, device, encoder_optimizer, decoder_optimizer):
        super().__init__(encoder, decoder, device)

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


def print_metrics(computed_metrics, epoch):
    formatter = Formatter()

    final_metrics_string = formatter.format_metrics(computed_metrics, validation=False)

    epoch_str = formatter.format_epoch(epoch)

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}\r{epoch_str} {final_metrics_string}')


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

    def __iter__(self):
        num_iterations = len(self.data_loader)
        self.recognizer.neural_pipeline.train_mode()

        inputs = []
        for i, (images, transcripts) in enumerate(self.data_loader):
            try:
                loss, result = self.train_one_iteration(images, transcripts)
            except torch.cuda.OutOfMemoryError:
                if not self.supress_errors:
                    raise
                continue

            outputs = result
            targets = transcripts
            # todo: fix this (may get rid of inputs and targets)
            yield IterationLogEntry(i, num_iterations, inputs, outputs, targets, loss)

    def train_one_iteration(self, images, transcripts):
        y_hat = self.recognizer(images, transcripts)
        loss = self.loss_fn(y_hat=y_hat, y=transcripts)

        # invoke zero_grad for each neural network
        self.recognizer.neural_pipeline.zero_grad()
        loss.backward()

        # todo: clip gradients

        # invoke optimizer.step() for every neural network if there is one
        self.recognizer.neural_pipeline.step()

        return loss, y_hat
