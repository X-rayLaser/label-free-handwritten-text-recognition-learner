from .augmentation import WeakAugmentation
from .image_utils import clip_all_heights, pad_images, make_rgb_batch


class ImagePipeline:
    def __init__(self, transform_functions):
        self.transform_functions = transform_functions

    def __call__(self, images):
        for func in self.transform_functions:
            images = func(images)
        return images


def compose_transforms(max_width, max_height):
    def clip(images):
        return clip_all_heights(images, max_height=max_height)

    def pad(images):
        return pad_images(images, max_width=max_width, max_height=max_height, extra_pad=32)
    return [clip, pad, make_rgb_batch]


def make_pretraining_pipeline(augmentation_options, max_width, max_height):
    no_augment_transforms = compose_transforms(max_width, max_height)

    augment = WeakAugmentation(**augmentation_options)
    all_transforms = [augment] + no_augment_transforms

    return ImagePipeline(all_transforms)


def make_validation_pipeline(max_width, max_height):
    return ImagePipeline(compose_transforms(max_width, max_height))
