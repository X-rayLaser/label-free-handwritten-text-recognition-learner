from .augmentation import WeakAugmentation
from .image_utils import clip_all_heights, pad_images, make_rgb_batch


class ImagePipeline:
    def __init__(self, transform_functions):
        self.transform_functions = transform_functions

    def __call__(self, images):
        for func in self.transform_functions:
            images = func(images)
        return images


def make_pretraining_pipeline(augmentation_options, max_heights):
    def clip(images):
        return clip_all_heights(images, max_height=max_heights)

    def pad(images):
        return pad_images(images, max_height=max_heights)

    augment = WeakAugmentation(**augmentation_options)
    return ImagePipeline([augment, clip, pad, make_rgb_batch])


def make_validation_pipeline(max_heights):
    def clip(images):
        return clip_all_heights(images, max_height=max_heights)

    def pad(images):
        return pad_images(images, max_height=max_heights)

    return ImagePipeline([clip, pad, make_rgb_batch])
