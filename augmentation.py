###########################################################
#
# DeepCancer - Skin cancer detection
# Description:
# - augmentation.py
# - create date: 2018-12-27
#
############################################################


import numpy as np
import argparse
import os

from keras.preprocessing.image import ImageDataGenerator


class Augmentation():
    def __init__(self, batch_size, data_path, image_dir, mask_dir, aug_dict,
                 image_save_prefix='image', mask_save_prefix='mask',
                 save_to_dir=None, target_size=(128, 128), seed=1):
        self.batch_size = batch_size
        self.data_path = data_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.aug_dict = aug_dict
        self.image_save_prefix = image_save_prefix
        self.mask_save_prefix = mask_save_prefix
        self.save_to_dir = save_to_dir
        self.target_size = target_size
        self.seed = seed

    def _norm_data(self, image, mask):
        if np.max(image) > 1:
            image = image / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return image, mask

    def _create_generator(self):
        """
        Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).

        :return:
        """
        image_datagen = ImageDataGenerator(**self.aug_dict)
        mask_datagen = ImageDataGenerator(**self.aug_dict)

        # create the generator for images
        image_generator = image_datagen.flow_from_directory(
            self.data_path,
            classes=[self.image_dir],
            class_mode=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            save_to_dir=self.save_to_dir,
            save_prefix=self.image_save_prefix,
            seed=self.seed)

        # create the generator for masks
        mask_generator = mask_datagen.flow_from_directory(
            self.data_path,
            classes=[self.mask_dir],
            class_mode=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            save_to_dir=self.save_to_dir,
            save_prefix=self.mask_save_prefix,
            seed=self.seed)

        # combine generators into one which yields image and masks
        generator = zip(image_generator, mask_generator)
        for (image, mask) in generator:
            image, mask = self._norm_data(image, mask)
            yield (image, mask)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='datasets/train', type=str,
                        help='The path to the training data')
    parser.add_argument('--image-dir', default='images', type=str,
                        help='The path to the images')
    parser.add_argument('--mask-dir', default='masks', type=str,
                        help='The path to the masks')
    parser.add_argument('--save-dir', default='augmented', type=str,
                        help='The path to the augmented images and masks')
    parser.add_argument('--batch-size', default=10, type=int,
                        help='The batch size of the ')
    parser.add_argument('--rotation-range', default=0.5, type=float,
                        help='Degree range for random rotations')
    parser.add_argument('--width-shift-range', default=0.5, type=float,
                        help='1-D array-like or int')
    parser.add_argument('--height-shift-range', default=0.5, type=float,
                        help='1-D array-like or int')
    parser.add_argument('--shear-range', default=0.5, type=float,
                        help='Shear Intensity (Shear angle in counter-clockwise direction in degrees)')
    parser.add_argument('--zoom-range', default=0.5, type=float,
                        help='Range for random zoom')
    return parser.parse_args()


if __name__ == "__main__":
    # ================= get the arguments ====================
    args = _get_args()
    save_path = os.path.join(args.data_path, args.save_dir)

    if not os.path.exists(save_path):
        print('Creating the {} directory...'.format(save_path))
        os.makedirs(save_path)

    # ========================================================
    # create the dict of data augmentation
    data_gen_args = dict(rotation_range=args.rotation_range,
                         width_shift_range=args.width_shift_range,
                         height_shift_range=args.height_shift_range,
                         shear_range=args.shear_range,
                         zoom_range=args.zoom_range,
                         horizontal_flip=True,
                         fill_mode='nearest')

    # ========================================================
    # create the instance of Augmentation class
    aug = Augmentation(args.batch_size,
                       args.data_path,
                       args.image_dir,
                       args.mask_dir,
                       data_gen_args,
                       save_to_dir=save_path)

    generator = aug._create_generator()

    for idx, batch in enumerate(generator):
        if (idx >= args.batch_size - 1):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
