import os

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image


def load_data_to_mem(data_path, classes, img_height=64, img_width=64):
    X = list()
    y = list()
    for folder in classes:
        path = '{}/{}/'.format(data_path, folder)
        files = [f for f in os.listdir(path) 
                 if os.path.isfile(os.path.join(path, f))]

        for filename in files:
            img = Image.open(os.path.join(path, filename))
            img = img.convert("L")  # to Grayscale
            img = img.resize((img_height, img_width))
            img = np.array(img)
            X.append(img)
            y.append(folder)
    return X, y


def augmentation(images, labels):
    original_images = images.copy()
    result = list()

    transformations = [
        iaa.Fliplr(1),
        iaa.Flipud(1),
        iaa.Affine(rotate=10),
        iaa.Affine(rotate=22),
        iaa.Affine(rotate=45),
        iaa.Affine(rotate=67),
        iaa.Affine(rotate=90),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Dropout(p=(0, 0.2)),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.CropAndPad(percent=(-0.25, 0.25))
    ]
    for transform in transformations:
        aug_images = transform.augment_images(original_images)
        result.extend(aug_images)
        
    result = images + result 
    labels = labels * (len(result) // len(labels))
    return result, labels