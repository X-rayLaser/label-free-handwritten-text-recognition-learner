import os
import re
import argparse
import random
import PIL
from PIL import Image


def prepare_iam_dataset(iam_location, output_dir, max_words=None, train_fraction=0.8, only_letters=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    words_dir = os.path.join(iam_location, 'words')

    train_file = os.path.join(output_dir, 'iam_train.txt')
    val_file = os.path.join(output_dir, 'iam_val.txt')

    pseudo_labels_file = os.path.join(output_dir, 'pseudo_labels.txt')
    with open(pseudo_labels_file, 'w') as _:
        pass

    paths_with_transcripts = parse_examples(iam_location, words_dir, max_words, only_letters)

    training_words, val_words = do_random_split(paths_with_transcripts, train_fraction)

    with open(train_file, 'w') as f:
        f.write('\n'.join(training_words))

    with open(val_file, 'w') as f:
        f.write('\n'.join(val_words))


def parse_examples(iam_location, words_dir, max_words, only_letters=False):
    transcripts_file = os.path.join(iam_location, 'ascii', 'words.txt')

    lines_skipped = 0
    paths_with_transcripts = []
    with open(transcripts_file) as f:
        for i, line in enumerate(f):
            if max_words and len(paths_with_transcripts) >= max_words:
                break

            try:
                path, gray_level, transcript = parse_line(line, words_dir, only_letters)
                if i + 1 % 10000:
                    print('Words processed:', i + 1)

                paths_with_transcripts.append(f'{path}, {gray_level}, {transcript}')
            except InvalidLineError as e:
                lines_skipped += 1
                print(f'Skipping a problematic line "{line}": {e.args[0]}')

    print(f'total lines processed {i}, lines skipped {lines_skipped}')

    return paths_with_transcripts


def do_random_split(examples, train_fraction):
    indices = list(range(len(examples)))
    random.shuffle(indices)
    train_size = int(len(indices) * train_fraction)

    training_words = [examples[idx] for idx in indices[:train_size]]
    val_words = [examples[idx] for idx in indices[train_size:]]
    return training_words, val_words


def parse_line(line, words_dir, only_letters=False):
    if line.lstrip().startswith('#'):
        raise InvalidLineError(f'Unexpected character "#" at the start of the line: "{line}"')

    parts = re.findall(r'[\w-]+', line)
    image_id = parts[0]
    status = parts[1]
    gray_level = parts[2]

    if status != 'ok':
        raise InvalidLineError(f'Expected status "ok". Got: "{status}"')

    transcript = parts[-1]

    if only_letters and not transcript.isalpha():
        raise InvalidLineError(f'Ignored transcript with non-letter characters: "{transcript}"')

    path = locate_image(words_dir, image_id)
    if not path:
        raise InvalidLineError(f'Image not found in the path: "{path}"')

    try:
        Image.open(path)
    except PIL.UnidentifiedImageError:
        raise InvalidLineError(f'Failure to read an image in the path: "{path}"')

    return path, gray_level, transcript


class InvalidLineError(Exception):
    """Raised when line content does not satisfy certain criteria"""


def locate_image(words_dir, image_id):
    parts = image_id.split('-')
    dir_name = parts[0]
    sub_dir_name = f'{parts[0]}-{parts[1]}'
    file_name = f'{image_id}.png'
    image_path = os.path.join(words_dir, dir_name, sub_dir_name, file_name)
    if os.path.isfile(image_path):
        return image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare a dataset of handwritten words taken from IAM database'
    )
    parser.add_argument('iam_home', type=str, help='Path to the location of IAM database directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--max-words', type=int, default=None, help='Total # of words to prepare')
    parser.add_argument('--only-letters', default=False, action='store_true',
                        help='Filters out words containing digits and other non-letter characters')

    args = parser.parse_args()
    prepare_iam_dataset(args.iam_home, args.output_dir, max_words=args.max_words,
                        only_letters=args.only_letters)
