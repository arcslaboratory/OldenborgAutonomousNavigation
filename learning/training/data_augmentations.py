import albumentations as A
import numpy as np
from fastai.vision.core import PILImage
from fastai.vision.augment import RandTransform


class AlbumentationsTransform(RandTransform):
    """Class that handles albumentations transformations during training."""

    def __init__(self, train_aug, valid_aug=None, split_idx=None):
        """Constructor for AlbumentationsTransform."""
        super().__init__()  # calls base class (RandTransform) constructor
        self.train_aug = train_aug
        self.valid_aug = (
            valid_aug or train_aug
        )  # defaults to training augmentations if no validation augmentations are provided
        self.split_idx = split_idx  # indicates whether the transform is applied to training or validation data
        self.order = 2  # apply after resizing

    def before_call(self, b, split_idx):
        """Called before the transform is applied to set the split index so we know if it's training or validation."""
        self.idx = split_idx

    def encodes(self, img: PILImage):
        """Apply the Albumentations transformations to the input image."""
        aug = (
            self.train_aug if self.idx == 0 else self.valid_aug
        )  # apply the appropriate augmentation
        image = np.array(img)  # albumentations works with numpy arrays
        image = aug(image=image)[
            "image"
        ]  # extract the image from the augmentation result
        return PILImage.create(image)  # convert back to PILImage for compatibility


def get_train_aug():
    """Data augmentations applied to training data."""
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),  # scale by 90%-110% of original size
                translate_percent=0.1,  # shift horizontally or vertically by up to 10% of its width/height
                rotate=(-10, 10),  # rotate between -10 and 10 degrees
                p=0.5,  # 50 chance to apply affine transformations
            ),
            A.RandomBrightnessContrast(
                p=0.2  # 20% chance to adjust brightness and contrast
            ),
            # possible augmentations to add:
            # A.Perspective(
            #    scale=(0.05, 0.1),  # apply perspective transformation with a scale factor between 5% and 10%
            #    p=0.5  # 50% chance to apply perspective transformation
            # ),
            # A.HueSaturationValue(
            #    hue_shift_limit=20,  # shift hue by up to 20 degrees
            #    sat_shift_limit=20,  # shift saturation by up to 20%
            #    val_shift_limit=20,  # shift value by up to 20%
            #    p=0.5  # 50% chance to apply hue, saturation, and value adjustments
            # ),
            # A.RandomGamma(
            #    gamma_limit=(80, 120),  # adjust gamma between 80% and 120%
            #    p=0.5  # 50% chance to apply gamma adjustment
            # ),
            # A.RGBShift(
            #    r_shift_limit=20,  # shift red channel by up to 20
            #    g_shift_limit=20,  # shift green channel by up to 20
            #    b_shift_limit=20,  # shift blue channel by up to 20
            #    p=0.5  # 50% chance to apply RGB shift
            # ),
            # A.MotionBlur(
            #    blur_limit=(3, 7),  # apply motion blur with a kernel size between 3 and 7
            #    p=0.5  # 50% chance to apply motion blur
            # ),
            # A.GaussianNoise(
            #    var_limit=(10, 50),  # add Gaussian noise with a variance between 10 and 50
            #    p=0.5  # 50% chance to apply Gaussian noise
            # ),
            # A.OpticalDistortion(
            #    distort_limit=0.05,  # apply optical distortion with a limit of 5%
            #    shift_limit=0.05,  # shift the image by up to 5%
            #    p=0.5  # 50% chance to apply optical distortion
            # ),
        ]
    )


def get_valid_aug():
    """Data augmentations applied to validation data (none)."""
    return A.Compose([])
