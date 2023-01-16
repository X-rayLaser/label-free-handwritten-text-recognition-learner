import argparse

from torch.optim import Adam

from hwr_self_train.loss_functions import MaskedCrossEntropy
from hwr_self_train.models import ImageEncoder, AttendingDecoder
from hwr_self_train.preprocessors import CharacterTokenizer
from hwr_self_train.history_saver import HistoryCsvSaver
from hwr_self_train.evaluation import evaluate
from hwr_self_train.metrics import Metric
from hwr_self_train.training import TrainableEncoderDecoder, WordRecognitionPipeline, Trainer, \
    TrainingLoop, print_metrics


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
    transform_pad = None
    loss_fn = Metric('loss', metric_fn=criterion, metric_args=["y_hat", "y"], transform_fn=transform_pad)
    data_loader = None
    trainer = Trainer(recognizer, data_loader, loss_fn, tokenizer)

    training_loop = TrainingLoop(trainer, metric_fns=[], epochs=100)

    history_saver = HistoryCsvSaver("history.csv")
    for epoch in training_loop:
        # todo: calculate metrics and show them; save them to csv file; save session
        metrics = evaluate()
        print_metrics(metrics, epoch)
        history_saver.add_entry(epoch, metrics)

    print("Done")
