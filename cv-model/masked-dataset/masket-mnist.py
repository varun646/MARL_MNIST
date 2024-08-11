import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class MaskedMNIST(Dataset):
    def __init__(self, mask_size, num_masks, transform=None):
        # TODO: apply masks to MNIST dataset
        self._mask_size = mask_size
        self._num_masks = num_masks
        self.transform = transform
        self.masked_images = MNIST

    def _apply_random_masks(img, num_output):
        # TODO: for a given image, return num_output images with various masks applied
        # masks should be of size self._mask_size
        # should apply self._num_masks masks to the image
        raise NotImplemented

    def __len__(self):
        return len(self.masked_images)

    def __getitem__(self, idx):
        return self.masked_images[idx]
