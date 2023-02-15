import os
import re
import random
import PIL
from PIL import Image
from .base import DataImporter


def prepare_iam_dataset(iam_location, max_words=None, train_fraction=0.8, only_letters=False):
    words_dir = os.path.join(iam_location, 'words')

    paths_with_transcripts = parse_examples(iam_location, words_dir, max_words, only_letters)

    return do_random_split(paths_with_transcripts, train_fraction)


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

                paths_with_transcripts.append((path, int(gray_level), transcript))
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


class IAMImporter(DataImporter):
    def __init__(self, iam_location, max_words=None,
                 transcribed_fraction=0.2, only_letters=False):
        self.training_examples, self.val_examples = prepare_iam_dataset(
            iam_location, max_words, 1 - transcribed_fraction, only_letters
        )

    def get_images(self):
        for path, grey_level, _ in self.training_examples:
            yield self.get_cleaned_image(path, grey_level)

    def get_transcribed_images(self):
        for path, grey_level, transcript in self.val_examples:
            image = self.get_cleaned_image(path, grey_level)
            yield image, transcript

    def get_cleaned_image(self, path, grey_level):
        image = Image.open(path)
        return clean_image(image, grey_level)


def clean_image(image, gray_level):
    return image.point(lambda p: 255 if p > gray_level else p)
