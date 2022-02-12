import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pdb
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

stl10_mean = (0.4408, 0.4279, 0.3867)
stl10_std = (0.2682, 0.2610, 0.2686)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_stl10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=48,
                              padding=int(48*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])

    train_labeled_dataset = datasets.STL10(
        root, split='train', download=True,
        transform=transform_labeled)

    train_unlabeled_dataset = datasets.STL10(
        root, split='unlabeled',
        transform=TransformFixMatch(mean=stl10_mean, std=stl10_std))

    test_dataset = datasets.STL10(
        root, split='test',
        transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=48,
                                  padding=int(48*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=48,
                                  padding=int(48*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


DATASET_GETTERS = {'stl10': get_stl10}
