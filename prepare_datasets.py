import argparse
import json
import os
import importlib
from hwr_self_train.formatters import show_progress_bar


def instantiate_class(dotted_path, *args, **kwargs):
    if '.' not in dotted_path:
        raise ClassImportError(f'Invalid import path: "{dotted_path}"')

    module_path, class_name = split_import_path(dotted_path)
    error_msg = f'Failed to import and instantiate a class "{class_name}" from "{module_path}": '
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)
    except Exception as e:
        raise ClassImportError(error_msg + str(e))


def split_import_path(dotted_path):
    idx = dotted_path.rindex('.')
    module_path = dotted_path[:idx]
    name = dotted_path[idx + 1:]
    return module_path, name


class ClassImportError(Exception):
    """Raised when an error occurs during a class importing"""


def create_unlabeled_dataset(data_importer, destination_dir):
    unlabeled_dir = os.path.join(destination_dir, 'unlabeled')
    os.makedirs(unlabeled_dir, exist_ok=True)

    gen = enumerate(data_importer.get_images())
    for i, image in show_progress_bar(gen, desc='Saving images: '):
        image.save(os.path.join(unlabeled_dir, f'{i}.png'))

    spaces = ' ' * 150
    print(f'\r{spaces}')


def create_labeled_dataset(data_importer, destination_dir):
    labeled_dir = os.path.join(destination_dir, 'labeled')

    os.makedirs(labeled_dir, exist_ok=True)

    if data_importer.get_transcribed_images() is not None:
        gen = enumerate(data_importer.get_transcribed_images())
        for i, (image, transcript) in show_progress_bar(gen, desc='Saving images: '):
            image.save(os.path.join(labeled_dir, f'{i}.png'))

            transcripts_path = os.path.join(labeled_dir, 'transcripts.txt')
            with open(transcripts_path, 'a') as f:
                f.write(transcript + '\n')

    spaces = ' ' * 150
    print(f'\r{spaces}')


def prepare_dataset(args):
    destination_dir = args.dest
    data_importer = instantiate_class(args.data_importer, **args.kwargs)
    create_unlabeled_dataset(data_importer, destination_dir)
    create_labeled_dataset(data_importer, destination_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare real handwriting data to be used for adaptation stage'
    )

    parser.add_argument('data_importer', type=str,
                        help='Fully qualified path (dotted) to Data Importer class')

    parser.add_argument('--kwargs', type=json.loads, default={},
                        help='Optional keyword arguments in JSON format passed to data importer constructor')

    parser.add_argument('--dest', type=str, default='tuning_data',
                        help='Output directory that will store prepared datasets')

    cmd_args = parser.parse_args()

    prepare_dataset(cmd_args)
