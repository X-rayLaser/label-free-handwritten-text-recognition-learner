from .augmentation import WeakAugmentation
from .image_utils import clip_all_heights, pad_images, make_rgb_batch


def clip_to_64(images):
    return clip_all_heights(images, max_height=64)


def pad(images):
    return pad_images(images, max_height=64)


class ImagePipeline:
    def __init__(self, transform_functions):
        self.transform_functions = transform_functions

    def __call__(self, images):
        for func in self.transform_functions:
            images = func(images)
        return images


pretraining_pipeline = ImagePipeline([WeakAugmentation(), clip_to_64, pad, make_rgb_batch])
