from __future__ import print_function

import os
import numpy as np
from PIL import Image
from skimage.io import imread
import cv2

data_path = "./data/"

image_rows = 420
image_cols = 580
dim = (image_cols, image_rows)
#image_rows = 512
#image_cols = 512


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # img = np.array(Image.open(os.path.join(train_data_path, image_name)))
        # img_mask = np.array(Image.open(os.path.join(train_data_path, image_mask_name)))
        # img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        # img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # print (img.shape)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, dim, interpolation = cv2.INTER_AREA)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path + 'file/train.npy', imgs)
    np.save(data_path + 'file/train_mask.npy', imgs_mask)
    print('Saving to .npy files done.')

def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating testing images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        # img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, dim, interpolation = cv2.INTER_AREA)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 50 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path + 'file/test.npy', imgs)
    np.save(data_path + 'file/test_mask.npy', imgs_mask)
    print('Saving to .npy files done.')


def create_valid_data():
    train_data_path = os.path.join(data_path, 'validation')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        # img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, dim, interpolation = cv2.INTER_AREA)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 50 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path + 'file/validation.npy', imgs)
    np.save(data_path + 'file/validation_mask.npy', imgs_mask)
    print('Saving to .npy files done.')

create_train_data()
create_test_data()
create_valid_data()
