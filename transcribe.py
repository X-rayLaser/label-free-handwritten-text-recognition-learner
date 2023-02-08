import argparse

from PIL import Image
from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.decoding import decode_output_batch
from hwr_self_train.session import SessionDirectoryLayout
from hwr_self_train.tokenizer import CharacterTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recognize text on a word image'
    )

    parser.add_argument('session_dir', type=str,
                        help='Path to the location of session directory containing model weights')
    parser.add_argument('image_path', type=str, help='Path to the location of image containing word')

    args = parser.parse_args()
    config = SessionDirectoryLayout(args.session_dir).load_config()
    env = TuningEnvironment(config)
    recognizer = env.recognizer

    tokenizer = CharacterTokenizer(config.charset)

    image = Image.open(args.image_path)
    label_distr = recognizer([image])
    texts = decode_output_batch(label_distr, tokenizer)
    print('Predicted text:', texts[0])
