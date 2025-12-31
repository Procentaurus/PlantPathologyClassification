import cv2
import os
import random
import albumentations as A

from .params import (IMG_HEIGHT,
                     IMG_WIDTH,
                     AUGMENTED_COPIES)


class Augmentator:

    def __init__(self, in_dir_path, out_dir_path):
        self.dir_path = in_dir_path
        self.out_dir_path = out_dir_path

    def augment_images(self):
        augmentations = {
            "crop": self._crop,
            "hflip": self._hflip,
            "vflip": self._vflip,
            "brightness_contrast": self._brightness_contrast,
            "shift_scale_rotate": self._shift_scale_rotate,
            "optical_distortion": self._optical_distortion,
            "grid_distortion": self._grid_distortion,
            "coarse_dropout": self._coarse_dropout,
            "gaussian_noise": self._gaussian_noise,
        }
        probabilities = {
            "crop": 0.6,
            "hflip": 0.8,
            "vflip": 0.8,
            "brightness_contrast": 0.6,
            "shift_scale_rotate": 0.5,
            "optical_distortion": 0.4,
            "grid_distortion": 0.5,
            "coarse_dropout": 0.5,
            "gaussian_noise": 0.7,
        }

        for filename in os.listdir(self.dir_path):
            image_path = os.path.join(self.dir_path, filename)
            name, ext = os.path.splitext(filename)

            # =========================
            # Save original image
            # =========================
            out_original_path = os.path.join(self.out_dir_path, filename)
            if not os.path.exists(out_original_path):
                img = cv2.imread(image_path)
                cv2.imwrite(out_original_path, img)
                print("Original saved:", filename)

            # =========================
            # Save augmented copies
            # =========================
            for i in range(AUGMENTED_COPIES):
                aug_names = list(augmentations.keys())

                # Apply selected augmentations sequentially
                augmented_image = self._read_image(image_path)
                for aug_name in aug_names:
                    aug_func = augmentations[aug_name]
                    augmented_image = aug_func(augmented_image,
                                               probabilities[aug_name])

                augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                out_name = f'{name}_{i}{ext}'
                out_path = os.path.join(self.out_dir_path, out_name)
                cv2.imwrite(out_path, augmented_bgr)
                print("Image created:", out_name)

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _crop(self, image, p):
        aug = A.RandomCrop(height=image.shape[0] * 0.7,
                           width=image.shape[1] * 0.7,
                           p=p)
        cropped = aug(image=image)["image"]
        resized = cv2.resize(cropped,
                             (IMG_WIDTH, IMG_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        return resized

    def _hflip(self, image, p):
        aug = A.HorizontalFlip(p=p)
        return aug(image=image)["image"]

    def _vflip(self, image, p):
        aug = A.VerticalFlip(p=p)
        return aug(image=image)["image"]

    def _brightness_contrast(self, image, p):
        aug = A.RandomBrightnessContrast(
            p=p
        )
        return aug(image=image)["image"]

    def _shift_scale_rotate(self, image, p):
        aug = A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=p
        )
        return aug(image=image)["image"]

    def _optical_distortion(self, image, p):
        aug = A.OpticalDistortion(
            distort_limit=(-0.3, -0.2),
            mode="fisheye",
            p=p
        )
        return aug(image=image)["image"]

    def _grid_distortion(self, image, p):
        aug = A.GridDistortion(
            distort_limit=0.4,
            p=p
        )
        return aug(image=image)["image"]

    def _coarse_dropout(self, image, p):
        aug = A.CoarseDropout(
            num_holes_range=(4, 4),
            max_height=90,
            max_width=90,
            fill_value=0,
            p=p
        )
        return aug(image=image)["image"]

    def _gaussian_noise(self, image, p):
        aug = A.GaussNoise(
            std_range=(0.10, 0.12),
            p=p
        )
        return aug(image=image)["image"]


if __name__ == "__main__":
    augmentator = Augmentator(in_dir_path="data/resized_train_images",
                              out_dir_path="data/aug_train_images")
    augmentator.augment_images()
