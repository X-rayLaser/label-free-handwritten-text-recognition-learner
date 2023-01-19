from hwr_self_train.utils import pad_sequences, prepare_targets, \
    make_tf_batch, ImageBatchPreprocessor


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
