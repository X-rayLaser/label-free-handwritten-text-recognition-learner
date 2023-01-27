import logging
from dataclasses import dataclass
from typing import Any

import torch
from hwr_self_train.utils import make_tf_batch
from .preprocessors import CharacterTokenizer
from .image_pipelines import ImagePipeline
from .models import ImageEncoder, AttendingDecoder


logger = logging.getLogger(__name__)


@dataclass
class EncoderDecoder:
    encoder: ImageEncoder
    decoder: AttendingDecoder
    device: torch.device

    def debug_attention(self, image_batch):
        image_batch = image_batch.to(self.device)
        encodings = self.encoder(image_batch)
        return self.decoder.debug_attention(encodings)

    def predict(self, image_batch, transcripts=None):
        image_batch = image_batch.to(self.device)
        if transcripts is not None:
            transcripts = transcripts.to(self.device)

        encodings = self.encoder(image_batch)
        return self.decoder(encodings, transcripts)

    def __call__(self, image_batch, transcripts=None):
        return self.predict(image_batch, transcripts)


@dataclass
class TrainableEncoderDecoder(EncoderDecoder):
    encoder_optimizer: Any
    decoder_optimizer: Any

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


@dataclass
class WordRecognitionPipeline:
    neural_pipeline: TrainableEncoderDecoder
    tokenizer: CharacterTokenizer
    image_preprocessor: ImagePipeline
    show_attention: bool = False

    def __call__(self, images, transcripts=None):
        """Given a list of PIL images and (optionally) corresponding list of text transcripts"""
        return self._try_recognize(images, transcripts)

    def _try_recognize(self, images, transcripts):
        try:
            return self._do_recognize(images, transcripts)
        except torch.cuda.OutOfMemoryError:
            widths_with_transcripts = [(im.width, t)
                                       for im, t in zip(images, transcripts)]
            largest_pair = max(widths_with_transcripts, key=lambda pair: pair[0])

            logger.exception('Recognition failed: out of memory.'
                             'Longest image width {}. '
                             'image transcript "{}"'.format(*largest_pair))
            raise

    def _do_recognize(self, images, transcripts):
        image_batch = self.image_preprocessor(images)

        if transcripts is not None:
            transcripts, _ = make_tf_batch(transcripts, self.tokenizer)

        if self.show_attention:
            return self.neural_pipeline.debug_attention(image_batch)
        else:
            return self.neural_pipeline(image_batch, transcripts)
