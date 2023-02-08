import os
from PIL import Image
from torch.utils.data import Dataset
from .data_generator import SimpleRandomWordGenerator


class SyntheticOnlineDataset(Dataset):
    def __init__(self, fonts_dir, size, word_sampler,
                 bg_range=(255, 255),
                 color_range=(0, 100),
                 font_size_range=(50, 100),
                 rotation_range=(0, 0)):
        super().__init__()
        self.size = size
        self.fonts_dir = fonts_dir

        simple_generator = SimpleRandomWordGenerator(word_sampler, self.fonts_dir,
                                                     bg_range=bg_range,
                                                     color_range=color_range,
                                                     font_size_range=font_size_range,
                                                     rotation_range=rotation_range)
        self.iterator = iter(simple_generator)

    def __getitem__(self, idx):
        if 0 <= idx < len(self):
            return self.generate_example()
        else:
            raise IndexError()

    def generate_example(self):
        im, word = next(self.iterator)
        return im, word

    def __len__(self):
        return self.size


class SyntheticOnlineDatasetCached(SyntheticOnlineDataset):
    def __init__(self, fonts_dir, size, word_sampler, bg_range=(255, 255),
                 color_range=(0, 100), font_size_range=(50, 100), rotation_range=(0, 0)):
        super().__init__(fonts_dir, size, word_sampler,
                         bg_range=bg_range,
                         color_range=color_range,
                         font_size_range=font_size_range,
                         rotation_range=rotation_range)

        self.cache = {}

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = super().__getitem__(idx)

        return self.cache[idx]


class RealLabeledDataset(Dataset):
    def __init__(self, ds_dir):
        self.examples = self.load_data(ds_dir)

    def load_data(self, ds_dir):
        transcripts_path = os.path.join(ds_dir, "transcripts.txt")
        transcripts_exist = os.path.isfile(transcripts_path)
        loader = PreparedDatasetLoader() if transcripts_exist else ImageFolderLoader()
        return loader(ds_dir)

    def __getitem__(self, idx):
        image_path, transcript = self.examples[idx]
        image = Image.open(image_path)
        return image, transcript

    def __len__(self):
        return len(self.examples)


class ImageFolderLoader:
    """Assumes that dataset is simply set of images in a folder.
    Transcript of each image is contained in its file name.
    For example, an image with a file name "apple_84" will have
    its label "apple".
    """
    def __call__(self, ds_dir):
        examples = []
        for f_name in os.listdir(ds_dir):
            if f_name.endswith('.png') or f_name.endswith('.jpg'):
                path = os.path.join(ds_dir, f_name)
                s, _ = os.path.splitext(f_name)
                transcript = s.split('_')[0]
                examples.append((path, transcript))
        return examples


class PreparedDatasetLoader:
    """Assumes that dataset is prepared by the toolkit"""
    def __call__(self, ds_dir):
        transcripts_path = os.path.join(ds_dir, "transcripts.txt")
        with open(transcripts_path) as f:
            content = f.read()
            transcripts = content.split("\n")

        def sort_key(file_name):
            name, _ = os.path.splitext(file_name)
            return int(name)

        file_names = [f_name for f_name in os.listdir(ds_dir)
                      if not f_name.endswith('.txt')]

        file_names.sort(key=sort_key)
        image_paths = [os.path.join(ds_dir, f_name) for f_name in file_names]
        return list(zip(image_paths, transcripts))


class RealUnlabeledDataset(Dataset):
    def __init__(self, ds_dir):
        self.image_paths = [os.path.join(ds_dir, f_name) for f_name in os.listdir(ds_dir)]

    def __getitem__(self, idx):
        return self.image_paths[idx],

    def __len__(self):
        return len(self.image_paths)


class PseudoLabeledDataset(Dataset):
    def __init__(self, paths_with_transcripts):
        self.paths_with_transcripts = paths_with_transcripts

    def __getitem__(self, idx):
        path, transcript = self.paths_with_transcripts[idx]
        image = Image.open(path)
        return image, transcript

    def __len__(self):
        return len(self.paths_with_transcripts)


def clean_image(image, gray_level):
    return image.point(lambda p: 255 if p > gray_level else p)
