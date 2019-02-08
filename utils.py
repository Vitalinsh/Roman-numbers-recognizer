import os
import math

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import matplotlib.pyplot as plt


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


def augmentation(images, labels, n_transform=60):
    original_images = images.copy()
    result = list()

    transformations = []
    for i in range (n_transform):
        # include all possible changes
        transform = iaa.SomeOf((1, 3), [
            iaa.Affine(rotate=(-35, 0), mode=ia.ALL, cval=(0, 255)),
            iaa.Affine(rotate=(0, 35), mode=ia.ALL, cval=(0, 255)),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
                   mode=ia.ALL, cval=(0, 255)),
            iaa.Affine(scale={"x": (0.7, 1.4), "y": (0.7, 1.4)},
                   mode=ia.ALL, cval=(0, 255)),
            iaa.Affine(shear=(-30, 30), mode=ia.ALL, cval=(0, 255)),
            # Blur
            iaa.GaussianBlur(sigma=(0.0, 2.0)),
            iaa.AverageBlur(k=3),
            iaa.MedianBlur(k=3),
            iaa.AdditiveGaussianNoise(scale=0.05*255),
            iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.4), 
            # Others
            iaa.CropAndPad(percent=(-0.15, 0.15)),
            iaa.Dropout(p=(0.01, 0.02)),
            iaa.PiecewiseAffine(scale=(0.03, 0.05)),
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
            ], random_order=True)
        transformations.append(transform)
    
    for transform in transformations:
        aug_images = transform.augment_images(original_images)
        result.extend(aug_images)

    result = images + result 
    labels = labels * (len(result) // len(labels))
    return result, labels

def show_images(images, labels, nb_cols=6, figsize=(15, 15)):
    nb_rows = math.ceil(len(images) / nb_cols)
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            if n < len(images):
                axs[i, j].xaxis.set_ticklabels([])
                axs[i, j].yaxis.set_ticklabels([])
                axs[i, j].set_title(labels[n], fontsize=10)
                axs[i, j].imshow(images[n], cmap="gray")
                n += 1
                
    