"""Script to train a segnet model.
"""

from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF

from skimage.transform import resize
from keras.models import Model
from keras.layers import (
    Input,
    concatenate,
    Conv2D,
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K, models
from skimage.io import imsave


def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)


K.set_image_data_format("channels_last")  # TF dimension ordering in this code
smooth = 1.0

img_rows = 128
img_cols = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_path = "./data/file/"



def load_train_data():
    imgs_train = np.load(data_path + "train.npy")
    imgs_mask_train = np.load(data_path + "train_mask.npy")
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + "validation.npy")
    imgs_mask_valid = np.load(data_path + "validation_mask.npy")
    return imgs_valid, imgs_mask_valid


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(K.round(K.clip((1 - y_true), 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_segnet():
    kernel = 3

    encoding_layers = [
        Conv2D(32, (3, 3), padding="same", input_shape=(img_rows, img_cols, 1)),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(32, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(64, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(64, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(128, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(128, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(128, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        MaxPooling2D(),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(size=(2, 2)),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(256, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(128, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(64, (kernel, kernel), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(64, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(32, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(32, (kernel, kernel), padding="same"),
        BatchNormalization(axis=3),
        Activation("relu"),
        Conv2D(1, (1, 1), padding="valid"),
        BatchNormalization(axis=3),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Activation("hard_sigmoid"))
    autoencoder.compile(
        loss=dice_coef_loss,
        optimizer="adam",
        metrics=[
            "accuracy",
            dice_coef,
            sensitivity,
            specificity,
            f1score,
            precision,
            recall,
            mean_iou,
        ],
    )
    autoencoder.summary()

    return autoencoder


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train():
    print("-" * 30)
    print("Loading and preprocessing train data...")
    print("-" * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_valid, imgs_mask_valid = load_validation_data()

    imgs_train = preprocess(imgs_train)
    print(imgs_train.shape)
    imgs_mask_train = preprocess(imgs_mask_train)
    print(imgs_mask_train.shape)
    imgs_valid = preprocess(imgs_valid)
    print(imgs_valid.shape)
    imgs_mask_valid = preprocess(imgs_mask_valid)
    print(imgs_mask_valid.shape)

    imgs_train = imgs_train.astype("float32")
    imgs_valid = imgs_valid.astype("float32")

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    val_mean = np.mean(imgs_valid)
    val_std = np.std(imgs_valid)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid -= val_mean
    imgs_valid /= val_std

    imgs_mask_train = imgs_mask_train.astype("float32")
    imgs_mask_train /= 255.0  # scale masks to [0, 1]

    imgs_mask_valid = imgs_mask_valid.astype("float32")
    imgs_mask_valid /= 255.0

    print("-" * 30)
    print("Creating and compiling model...")
    print("-" * 30)
    model = get_segnet()
    model_checkpoint = ModelCheckpoint(
        "./data/file/segnet.hdf5",
        monitor="val_loss",
        save_best_only=True,
    )

    print("-" * 30)
    print("Fitting model...")
    print("-" * 30)
    # earlystopper=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    his = model.fit(
        imgs_train,
        imgs_mask_train,
        batch_size=32,
        epochs=300,
        verbose=1,
        shuffle=True,
        validation_data=(imgs_valid, imgs_mask_valid),
        callbacks=[model_checkpoint],
    )

    score_1 = model.evaluate(imgs_train, imgs_mask_train, batch_size=32, verbose=1)
    print(" Train loss:", score_1[0])
    print(" Train accuracy:", score_1[1])
    print(" Train dice_coef:", score_1[2])
    print(" Train sensitivity:", score_1[3])
    print(" Train specificity:", score_1[4])
    print(" Train f1score:", score_1[5])
    print("Train precision:", score_1[6])
    print(" Train recall:", score_1[7])
    print(" Train mean_iou:", score_1[8])
    res_loss_1 = np.array(score_1)
    np.savetxt(data_path + "res_loss_1.txt", res_loss_1)

    score_2 = model.evaluate(imgs_valid, imgs_mask_valid, batch_size=32, verbose=1)
    print(" valid loss:", score_2[0])
    print(" valid  accuracy:", score_2[1])
    print(" valid  dice_coef:", score_2[2])
    print(" valid  sensitivity:", score_2[3])
    print(" valid  specificity:", score_2[4])
    print(" valid f1score:", score_2[5])
    print("valid  precision:", score_2[6])
    print(" valid  recall:", score_2[7])
    print(" valid  mean_iou:", score_2[8])

    res_loss_2 = np.array(score_2)
    np.savetxt(data_path + "res_loss_2.txt", res_loss_2)

    plt.plot()
    plt.plot(his.history["loss"], label="train loss")
    plt.plot(his.history["val_loss"], c="g", label="val loss")
    plt.title("train and val loss")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot()
    plt.plot(his.history["acc"], label="train accuracy")
    plt.plot(his.history["val_acc"], c="g", label="val accuracy")
    plt.title("train  and val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["dice_coef"], label="train dice_coef")
    plt.plot(his.history["val_dice_coef"], c="g", label="val dice_coef")
    plt.title("train  and val dice_coef")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["sensitivity"], label="train sensitivity")
    plt.plot(his.history["val_sensitivity"], c="g", label="val sensitivity")
    plt.title("train  and val sensitivity")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["specificity"], label="train specificity")
    plt.plot(his.history["val_specificity"], c="g", label="val specificity")
    plt.title("train  and val specificity")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["f1score"], label="train f1score")
    plt.plot(his.history["val_f1score"], c="g", label="val f1score")
    plt.title("train  and val f1score")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["precision"], label="train precision")
    plt.plot(his.history["val_precision"], c="g", label="val_precision")
    plt.title("train  and val precision")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["mean_iou"], label="Train mean_iou")
    plt.plot(his.history["val_mean_iou"], c="g", label="val_mean_iou")
    plt.title("train and val mean_iou")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    plt.plot()
    plt.plot(his.history["recall"], label="train recall")
    plt.plot(his.history["val_recall"], c="g", label="val_recall")
    plt.title("train  and val recall")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    train()
