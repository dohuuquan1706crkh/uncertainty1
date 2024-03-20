import albumentations as albu
import torch
# import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import numpy as np

def get_training_augmentation(space: str="rgb", img_size: int=256):
    if space == "rgb":
        train_transform = [

            albu.HorizontalFlip(p=0.5),
            
            # strongly transform
            # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
            # albu.RandomCrop(height=img_size, width=img_size, always_apply=True),
            albu.CenterCrop(height=img_size, width=img_size, always_apply=True),

            # albu.IAAAdditiveGaussianNoise(p=0.2),
            # albu.IAAPerspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    # albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
    else:
        train_transform = [
            albu.HorizontalFlip(p=0.5),            
            albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
            albu.CenterCrop(height=img_size, width=img_size, always_apply=True),
        ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size: int=256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
        albu.CenterCrop(height=img_size, width=img_size, always_apply=True),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    if preprocessing_fn:
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
        _transform = [
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)

def normalize_sample(image):
    image = image/255
    mean = float(image.mean())
    std = float(image.std())
    image -= mean
    image /= std
    image = np.expand_dims(image, -1)
    return image