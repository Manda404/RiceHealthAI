"""
Image transformation utilities for RiceHealthAI.

Provides standardized data augmentation and preprocessing pipelines
for training, validation, and test datasets.
"""

from torchvision import transforms
from typing import Tuple


def get_train_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Data augmentation and preprocessing pipeline for training images.

    Parameters
    ----------
    img_size : Tuple[int, int], default=(224, 224)
        Target size (width, height) for resizing the input images.

    Returns
    -------
    torchvision.transforms.Compose
        Transformation pipeline for the training set.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_valid_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Preprocessing pipeline for validation images.

    Parameters
    ----------
    img_size : Tuple[int, int], default=(224, 224)
        Target size (width, height) for resizing the input images.

    Returns
    -------
    torchvision.transforms.Compose
        Transformation pipeline for the validation set.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Preprocessing pipeline for test images.

    Parameters
    ----------
    img_size : Tuple[int, int], default=(224, 224)
        Target size (width, height) for resizing the input images.

    Returns
    -------
    torchvision.transforms.Compose
        Transformation pipeline for the test set.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
