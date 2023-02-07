import argparse

from PIL import Image
from hwr_self_train.environment import TuningEnvironment
from hwr_self_train.decoding import decode_output_batch
from configuration import Configuration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recognize text on a word image'
    )
    parser.add_argument('image_path', type=str, help='Path to the location of image containing word')
    args = parser.parse_args()

    env = TuningEnvironment()
    recognizer = env.recognizer
    tokenizer = Configuration.tokenizer

    image = Image.open(args.image_path)
    label_distr = recognizer([image])
    texts = decode_output_batch(label_distr, tokenizer)
    print('Predicted text:', texts[0])
