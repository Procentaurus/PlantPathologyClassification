import os
import cv2


class Resizer:

    def __init__(self, in_dir_path, out_dir_path, width, height):
        self.dir_path = in_dir_path
        self.out_dir_path = out_dir_path
        self.width = width
        self.height = height

    def resize_images(self):
        """
        Resize all images in `in_dir_path` to the given width and height
        and save them to `out_dir_path`.
        """
        for filename in os.listdir(self.dir_path):
            image_path = os.path.join(self.dir_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {filename}: cannot read image")
                continue

            resized_image = cv2.resize(image,
                                       (self.width, self.height),
                                       interpolation=cv2.INTER_AREA)
            out_path = os.path.join(self.out_dir_path, filename)
            cv2.imwrite(out_path, resized_image)


if __name__ == "__main__":
    train_resizer = Resizer("data/orig_train_images",
                            "data/resized_train_images",
                            width=256,
                            height=256)
    train_resizer.resize_images()

    test_resizer = Resizer("data/orig_test_images",
                           "data/resized_test_images",
                           width=256,
                           height=256)
    test_resizer.resize_images()
