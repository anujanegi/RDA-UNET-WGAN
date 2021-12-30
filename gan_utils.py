"""Utility fucntions for gan training.
"""

import numpy as np
from skimage.io import imread
from keras.utils import to_categorical
from random import shuffle, randint
import time


def imgs2discr(real_images, real_labels, fake_labels):
    """
    It gets the input data to the discriminator
    :param real_images: input images
    :param real_labels: ground truth
    :param fake_labels: predicted labels
    :return: input images and labels to the discriminative network
    """
    real = np.concatenate((real_images, real_labels), axis=3)
    fake = np.concatenate((real_images, fake_labels), axis=3)

    img_batch = np.concatenate((real, fake), axis=0)
    lab_batch = np.ones((img_batch.shape[0], 1))
    lab_batch[real.shape[0]:,...] = 0

    return img_batch, lab_batch


def imgs2gan(real_images, real_labels):
    """
    It gets the input data to the segmentation network
    :param real_images: input images
    :param real_labels: ground truth
    :return: input images and labels to the segmentation network
    """
    img_batch = [real_images, real_labels]
    lab_batch = np.ones((real_images.shape[0], 1))

    return img_batch, lab_batch
