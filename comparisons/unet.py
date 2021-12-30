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
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    Conv2D,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K
from skimage.io import imsave


def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)


K.set_image_data_format("channels_last")  # TF dimension ordering in this code
smooth = 1.0

img_rows = 128
img_cols = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_path = "/home/NanLi/ceshi/unsmceshi/u_net/raw/file/"


# data_path = '/Users/xuchenyang/Documents/sec_exp/file/'


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


def get_unet():
    inputs = Input(shape=(img_rows, img_cols, 1))

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    BatchNormalization(axis=3)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    BatchNormalization(axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    BatchNormalization(axis=3)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    # BatchNormalization(axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    # BatchNormalization(axis=3)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    # BatchNormalization(axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    # BatchNormalization(axis=3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    # BatchNormalization(axis=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    # BatchNormalization(axis=3)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    # BatchNormalization(axis=3)
    # conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    # BatchNormalization(axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    # BatchNormalization(axis=3)
    # conv6 = Dropout(0.5)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    # BatchNormalization(axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    # BatchNormalization(axis=3)
    # conv7 = Dropout(0.5)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    # BatchNormalization(axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    # BatchNormalization(axis=3)
    # conv8 = Dropout(0.5)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    # BatchNormalization(axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    # BatchNormalization(axis=3)
    # conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation="hard_sigmoid", padding="same")(conv9)
    # BatchNormalization(axis=3)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(
        optimizer="adam",
        loss=dice_coef_loss,
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

    return model


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
    model = get_unet()
    model_checkpoint = ModelCheckpoint(
        "/home/NanLi/ceshi/unsmceshi/u_net/raw/file/unet.hdf5",
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
    # model = get_unet()
    # print(model.summary())
    train()
