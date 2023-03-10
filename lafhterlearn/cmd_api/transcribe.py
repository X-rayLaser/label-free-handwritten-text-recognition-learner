from PIL import Image

from lafhterlearn.environment import TuningEnvironment
from lafhterlearn.decoding import decode_output_batch
from lafhterlearn.session import SessionDirectoryLayout
from lafhterlearn.tokenizer import CharacterTokenizer
from .base import Command


class TranscribeCommand(Command):
    name = 'transcribe'
    help = 'Recognize a handwritten text on a given word image ' \
           'using a model loaded from the latest checkpoint'

    def configure_parser(self, parser):
        parser.add_argument('session_dir', type=str,
                            help='Path to the location of session directory containing model weights')
        parser.add_argument('image_path', type=str, help='Path to the location of image containing word')

    def __call__(self, args):
        config = SessionDirectoryLayout(args.session_dir).load_config()
        env = TuningEnvironment(config)
        recognizer = env.recognizer

        tokenizer = CharacterTokenizer(config.charset)

        image = Image.open(args.image_path)
        label_distr = recognizer([image])
        texts = decode_output_batch(label_distr, tokenizer)
        print('Predicted text:', texts[0])
