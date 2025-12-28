import cv2
import os
import albumentations as A


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
            "cutout": self._cutout,
            "coarse_dropout": self._coarse_dropout,
            "gaussian_noise": self._gaussian_noise,
        }

        for filename in os.listdir(self.dir_path):
            image_path = os.path.join(self.dir_path, filename)
            name, ext = os.path.splitext(filename)

            for aug_name, aug_func in augmentations.items():
                try:
                    augmented = aug_func(image_path)
                    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    out_name = f"{name}_{aug_name}{ext}"
                    out_path = os.path.join(self.out_dir_path, out_name)
                    cv2.imwrite(out_path, augmented_bgr)
                except Exception as e:
                    print(f"Skipping {filename} [{aug_name}]: {e}")

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _crop(self, image_path):
        image = self._read_image(image_path)
        aug = A.RandomCrop(height=image.shape[0] // 2,
                           width=image.shape[1] // 2,
                           p=1.0)
        return aug(image=image)["image"]

    def _hflip(self, image_path):
        image = self._read_image(image_path)
        aug = A.HorizontalFlip(p=1.0)
        return aug(image=image)["image"]

    def _vflip(self, image_path):
        image = self._read_image(image_path)
        aug = A.VerticalFlip(p=1.0)
        return aug(image=image)["image"]

    def _brightness_contrast(self, image_path):
        image = self._read_image(image_path)
        aug = A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=1.0
        )
        return aug(image=image)["image"]

    def _shift_scale_rotate(self, image_path):
        image = self._read_image(image_path)
        aug = A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0
        )
        return aug(image=image)["image"]

    def _optical_distortion(self, image_path):
        image = self._read_image(image_path)
        aug = A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.05,
            p=1.0
        )
        return aug(image=image)["image"]

    def _grid_distortion(self, image_path):
        image = self._read_image(image_path)
        aug = A.GridDistortion(
            num_steps=5,
            distort_limit=0.03,
            p=1.0
        )
        return aug(image=image)["image"]

    def _cutout(self, image_path):
        image = self._read_image(image_path)
        aug = A.Cutout(
            num_holes=1,
            max_h_size=64,
            max_w_size=64,
            fill_value=0,
            p=1.0
        )
        return aug(image=image)["image"]

    def _coarse_dropout(self, image_path):
        image = self._read_image(image_path)
        aug = A.CoarseDropout(
            max_holes=4,
            max_height=64,
            max_width=64,
            fill_value=0,
            p=1.0
        )
        return aug(image=image)["image"]

    def _gaussian_noise(self, image_path):
        """
        Apply light Gaussian noise to simulate sensor noise.
        Safe for leaf disease detection.
        """
        image = self._read_image(image_path)

        aug = A.GaussNoise(
            var_limit=(5.0, 20.0),
            mean=0,
            p=1.0
        )

        return aug(image=image)["image"]


if __name__ == "__main__":
    augmentator = Augmentator("data/resized_train_images",
                              "data/aug_train_images")
    augmentator.augment_images()
