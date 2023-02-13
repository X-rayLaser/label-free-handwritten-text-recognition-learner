from PIL import Image

from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.decoding import decode_output_batch
from hwr_self_train.session import SessionDirectoryLayout
from hwr_self_train.tokenizer import CharacterTokenizer
from .base import Command


class TranscribeCommand(Command):
    name = 'transcribe'
    help = 'Recognize a handwritten text on a given word image'

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
