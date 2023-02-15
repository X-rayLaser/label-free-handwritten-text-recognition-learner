from random import randrange, uniform
import math
import random
from dataclasses import dataclass
import torch

from torchvision import transforms
from torchvision.transforms import Compose, PILToTensor, ToPILImage
from .image_utils import fit_height


@dataclass
class WeakAugmentation:
    p_augment: float
    target_height: int
    fill: int
    rotation_degrees_range: tuple
    blur_size: int
    blur_sigma: tuple
    noise_sigma: int
    should_fit_height: bool

    def __call__(self, images):
        rotate = self._rotate_and_scale()
        blur = transforms.GaussianBlur(self.blur_size, sigma=self.blur_sigma)
        add_noise = gaussian_noise(sigma=self.noise_sigma)

        images = self._augment_images(images, rotate)
        images = self._augment_images(images, add_noise)
        images = self._augment_images(images, blur)

        return images

    def _augment_images(self, images, augment_func):
        return [augment_func(im) if self._should_augment() else im for im in images]

    def _should_augment(self):
        return random.random() < self.p_augment

    def _rotate_and_scale(self):
        rotate = transforms.RandomRotation(degrees=self.rotation_degrees_range,
                                           expand=True, fill=self.fill)

        def transform_func(image):
            return fit_height(rotate(image), target_height=self.target_height)

        func = transform_func if self.should_fit_height else rotate
        return func


class StrongAugmentation:
    brightness = transforms.ColorJitter(brightness=(0.05, 0.95))
    contrast = transforms.ColorJitter(contrast=(0.05, 0.95))
    equalize = transforms.RandomEqualize(1)
    rotate = transforms.RandomRotation((-30, 30), fill=255)

    degrees_range = (math.degrees(-0.3), math.degrees(0.3))
    shear_x = transforms.RandomAffine(0, shear=degrees_range, fill=255)
    shear_y = transforms.RandomAffine(0, shear=(0, 0) + degrees_range, fill=255)
    auto_contrast = transforms.RandomAutocontrast(1)
    translate_x = transforms.RandomAffine(0, translate=(0.3, 0), fill=255)
    translate_y = transforms.RandomAffine(0, translate=(0, 0.3), fill=255)

    transforms_per_image = 2

    def __call__(self, images):
        return [self.transform_image(im) for im in images]

    def transform_image(self, image):
        transformations = self.get_random_transformations(self.transforms_per_image)
        for transform_func in transformations:
            image = transform_func(image)
        return image

    def get_random_transformations(self, n):
        return [self.random_transformation() for _ in range(n)]

    def random_transformation(self):
        all_transforms = [self.auto_contrast, self.brightness, self.contrast,
                          self.equalize, identity, posterize, self.rotate,
                          adjust_sharpness, self.shear_x, self.shear_y,
                          solarize, self.translate_x, self.translate_y]
        idx = randrange(0, len(all_transforms))
        return all_transforms[idx]


def identity(image): return image


def posterize(image):
    bits = randrange(4, 8 + 1)
    return transforms.RandomPosterize(bits, p=1)(image)


def adjust_sharpness(image):
    factor = uniform(0.05, 0.95)
    return transforms.RandomAdjustSharpness(factor, p=1)(image)


def solarize(image):
    threshold = int(round(uniform(0, 1) * 255))
    return transforms.RandomSolarize(threshold, p=1)(image)


def gaussian_noise(sigma):
    def add_noise(tensor):
        noisy = tensor + sigma * torch.randn_like(tensor.to(torch.float32))
        noisy = torch.clamp(noisy, 0, 255)
        return noisy.to(tensor.dtype)

    return Compose([PILToTensor(), add_noise, ToPILImage()])
