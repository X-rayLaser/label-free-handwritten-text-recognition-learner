import torch
from torchvision import transforms

from .augmentation import WeakAugmentation
from .image_utils import clip_height, pad_images, to_rgb


def clip_to_64(images):
    return clip_height(images, max_value=64)


def pad(images):
    return pad_images(images, max_height=64)


def to_tensors(images):
    to_tensor = transforms.ToTensor()
    return [to_tensor(im) for im in images]


def to_rgb_tensors(tensors):
    return [to_rgb(t) for t in tensors]


def make_batch(images):
    tensors = to_tensors(images)
    tensors = to_rgb_tensors(tensors)
    return torch.stack(tensors)


class ImagePipeline:
    def __init__(self, transform_functions):
        self.transform_functions = transform_functions

    def __call__(self, images):
        for func in self.transform_functions:
            images = func(images)
        return images


pretraining_pipeline = ImagePipeline([WeakAugmentation(), clip_to_64, pad, make_batch])
