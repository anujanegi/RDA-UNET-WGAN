from __future__ import print_function

import numpy as np
import os
from keras import layers
import tensorflow as tf
from PIL import Image
from skimage.transform import resize

import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF

from skimage.transform import resize
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    concatenate,
    Conv2D,
    ZeroPadding2D,
    Convolution2D,
    Conv2DTranspose,
    MaxPooling2D,
    add,
    UpSampling2D,
    multiply,
    Dropout,
    BatchNormalization,
    LeakyReLU,
    Dense,
    Flatten,
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


data_path = "./raw/file/"
path_to_save_results = data_path + "predictions/"


# data_path = '/Users/xuchenyang/Documents/sec_exp/file/'


def load_train_data():
    imgs_train = np.load(data_path + "train.npy")
    imgs_mask_train = np.load(data_path + "train_mask.npy")
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + "validation.npy")
    imgs_mask_valid = np.load(data_path + "validation_mask.npy")
    return imgs_valid, imgs_mask_valid


def load_test_data():
    imgs_test = np.load(data_path + "test.npy")
    imgs_mask_test = np.load(data_path + "test_mask.npy")
    return imgs_test, imgs_mask_test


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


def unet4(upsamezhi, chuandi, F_g, F_l, F_int):

    up = Conv2D(F_g, (1, 1), activation="relu", padding="same")(upsamezhi)
    up = BatchNormalization()(up)

    down = Conv2D(F_l, (1, 1), activation="relu", padding="same")(chuandi)
    down = BatchNormalization()(down)
    sumadd = add([up, down])
    sumadd = Activation(activation="relu")(sumadd)

    jihe = Conv2D(F_int, (1, 1), activation="relu", padding="same")(sumadd)
    sumhalf = BatchNormalization()(jihe)

    sum_1 = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(sumhalf)
    sum_1 = BatchNormalization()(sum_1)

    xinchuandi = multiply([chuandi, sum_1])
    return xinchuandi


def bottleneck(
    x,
    filters_bottleneck,
    mode="cascade",
    depth=6,
    kernel_size=(3, 3),
    activation="relu",
):
    dilated_layers = []
    if mode == "cascade":  # used in the competition
        for i in range(depth):
            x = Conv2D(
                filters_bottleneck,
                kernel_size,
                activation=activation,
                padding="same",
                dilation_rate=2 ** i,
            )(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == "parallel":  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(
                    filters_bottleneck,
                    kernel_size,
                    activation=activation,
                    padding="same",
                    dilation_rate=2 ** i,
                )(x)
            )
        return add(dilated_layers)


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation="relu")(res_path)
    res_path = Conv2D(
        filters=nb_filters[0], kernel_size=(3, 3), padding="same", strides=strides[0]
    )(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation="relu")(res_path)
    res_path = Conv2D(
        filters=nb_filters[1], kernel_size=(3, 3), padding="same", strides=strides[1]
    )(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        x
    )
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation="relu")(main_path)

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        main_path
    )

    shortcut = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [512, 512], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    xin_encoder_1 = unet4(main_path, from_encoder[4], 256, 256, 128)
    main_path = concatenate([main_path, xin_encoder_1], axis=3)
    main_path = res_block(main_path, [512, 512], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_2 = unet4(main_path, from_encoder[3], 128, 128, 64)
    main_path = concatenate([main_path, xin_encoder_2], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_3 = unet4(main_path, from_encoder[2], 64, 64, 32)
    main_path = concatenate([main_path, xin_encoder_3], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_4 = unet4(main_path, from_encoder[1], 32, 32, 16)
    main_path = concatenate([main_path, xin_encoder_4], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_5 = unet4(main_path, from_encoder[0], 16, 16, 8)
    main_path = concatenate([main_path, xin_encoder_5], axis=3)
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])

    return main_path


def build_res_unet():
    # inputs = Input(shape=input_shape)
    inputs = Input(shape=(img_rows, img_cols, 1))

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[4], [512, 512], [(2, 2), (1, 1)])

    bottle = bottleneck(path, filters_bottleneck=256, mode="cascade")

    path = decoder(bottle, from_encoder=to_decoder)

    path = Conv2D(
        filters=1, kernel_size=(1, 1), activation="hard_sigmoid", padding="same"
    )(path)
    model = Model(input=inputs, output=path)
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

    # model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])

    return model


def build_discriminator():
    # inputs = Input(shape=(img_rows, img_cols, 1))
    # model = Sequential()
    #
    # model.add(Conv2D(32, kernel_size=(1, 1), strides=2, input_shape=inputs, padding="same"))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, kernel_size=(1, 1), strides=2, padding="same"))
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(128, kernel_size=(1, 1), strides=2, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(256, kernel_size=(1, 1), strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))
    inputs = Input(shape=(img_rows, img_cols, 1))

    # model = []
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        inputs
    )
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))(
        path_1
    )
    path_1 = BatchNormalization()(path_1)
    path_1 = LeakyReLU(alpha=0.2)(path_1)
    path_1 = Flatten()(path_1)
    path_1 = Dense(1, activation="sigmoid")(path_1)

    # model.append(path_1)

    # encoder_ = encoder(inputs)
    # model=Model(input=inputs, output=encoder_)
    # img = Input(shape=self.img_shape)
    # validity = model(img)
    # model = Model(img, validity)
    model_ = Model(input=inputs, output=path_1)
    model_.compile(
        optimizer=Adam(0.0002, 0.5),
        loss="binary_crossentropy",
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
    return model_


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

    # print(imgs_train.shape)

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

    # preparing test data
    imgs_test, imgs_test_mask = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test_mask = preprocess(imgs_test_mask)
    imgs_test_source = imgs_test.astype("float32")
    imgs_test_source -= mean
    imgs_test_source /= std
    imgs_test_mask = imgs_test_mask.astype("float32")
    imgs_test_mask /= 255.0  # scale masks to [0, 1]

    print("-" * 30)
    print("Creating generator model...")
    print("-" * 30)
    # The model is not compiled
    model_generator = build_res_unet()
    # print(model.summary())

    print("-" * 30)
    print("Creating and compiling discriminator model...")
    print("-" * 30)
    model_discriminator = build_discriminator()
    # print(model_discriminator.summary())

    model_checkpoint = ModelCheckpoint(
        "./raw/file/RDAunet.hdf5", monitor="val_loss", save_best_only=True
    )

    # The generator segments the image
    input = Input(shape=(img_rows, img_cols, 1))
    generated_mask = model_generator(input)
    # idx = np.random.randint(0,imgs_train.shape[0])
    # img = imgs_train[idx]
    # img = np.expand_dims(img, axis=0)
    # img = tf.constant(img)
    # print(img.shape)
    # generated_mask = model(img)

    model_discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid_mask = model_discriminator(generated_mask)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    # input = Input(shape=(img_rows, img_cols, 1))
    combined = Model(input, valid_mask)
    combined.compile(
        optimizer=Adam(0.0002, 0.5),
        loss="binary_crossentropy",
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

    print("-" * 30)
    print("Fitting model...")
    print("-" * 30)
    # earlystopper=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    # his = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=2, verbose=1, shuffle=True,
    #                 validation_data=(imgs_valid, imgs_mask_valid), callbacks=[model_checkpoint])
    batch_size = 32
    epochs = 500
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        print("Training...epoch " + str(epoch))
        # Select a random half of images
        idx = np.random.randint(0, imgs_train.shape[0], batch_size)
        imgs = imgs_train[idx]

        masks = imgs_mask_train[idx]
        # Sample noise and generate a batch of new images
        gen_masks = model_generator.predict(imgs)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = model_discriminator.train_on_batch(masks, valid)
        d_loss_fake = model_discriminator.train_on_batch(gen_masks, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = combined.train_on_batch(imgs, valid)

        # generate and save images
        imgs_mask_predict = model_generator.predict(imgs_test_source, verbose=1)
        np.save(data_path + "predict.npy", imgs_mask_predict)
        predicted_masks = np.load(data_path + "predict.npy")
        predicted_masks *= 255

        update = "Epoch " + str(epoch) + " done."
        print(update)

        if epoch % 10 == 0:

            print("Testing...")
            imgs_test, imgs_test_mask = load_test_data()
            for i in range(imgs_test.shape[0]):
                img = resize(imgs_test[i], (128, 128), preserve_range=True)
                img_mask = resize(imgs_test_mask[i], (128, 128), preserve_range=True)
                im_test_source = Image.fromarray(img.astype(np.uint8))
                im_test_masks = Image.fromarray((img_mask.squeeze()).astype(np.uint8))
                im_test_predict = Image.fromarray(
                    (predicted_masks[i].squeeze()).astype(np.uint8)
                )
                if epoch == 0:
                    im_test_source_name = "Test_Image_" + str(i + 1) + ".png"
                    im_test_gt_mask_name = (
                        "Test_Image_" + str(i + 1) + "_OriginalMask.png"
                    )
                    im_test_source.save(
                        os.path.join(path_to_save_results, im_test_source_name)
                    )
                    im_test_masks.save(
                        os.path.join(path_to_save_results, im_test_gt_mask_name)
                    )
                im_test_predict_name = (
                    "Test_Image_" + str(i + 1) + "_epoch_" + str(epoch) + "_Predict.png"
                )
                im_test_predict.save(
                    os.path.join(path_to_save_results, im_test_predict_name)
                )

            print("Testing done")
        # Plot the progress
        # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples

    # score_1=model_generator.evaluate(imgs_train,imgs_mask_train,batch_size=32,verbose=1)
    # print(' Train loss:',score_1[0])
    # print(' Train accuracy:',score_1[1])
    # print(' Train dice_coef:',score_1[2])
    # print(' Train sensitivity:',score_1[3])
    # print(' Train specificity:',score_1[4])
    # print(' Train f1score:',score_1[5])
    # print('Train precision:',score_1[6])
    # print(' Train recall:',score_1[7])
    # print(' Train mean_iou:',score_1[8])
    # res_loss_1 = np.array(score_1)
    # np.savetxt(data_path+ 'res_loss_1.txt', res_loss_1)
    #
    # score_2=model.evaluate(imgs_valid,imgs_mask_valid,batch_size=32,verbose=1)
    # print(' valid loss:',score_2[0])
    # print(' valid  accuracy:',score_2[1])
    # print(' valid  dice_coef:',score_2[2])
    # print(' valid  sensitivity:',score_2[3])
    # print(' valid  specificity:',score_2[4])
    # print(' valid f1score:',score_2[5])
    # print('valid  precision:',score_2[6])
    # print(' valid  recall:',score_2[7])
    # print(' valid  mean_iou:',score_2[8])
    # res_loss_2 = np.array(score_2)
    # np.savetxt(data_path + 'res_loss_2.txt', res_loss_2)
    #
    # plt.plot()
    # plt.plot(his.history['loss'], label='train loss')
    # plt.plot(his.history['val_loss'], c='g', label='val loss')
    # plt.title('train and val loss')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    #
    # plt.plot()
    # plt.plot(his.history['acc'], label='train accuracy')
    # plt.plot(his.history['val_acc'], c='g', label='val accuracy')
    # plt.title('train  and val acc')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt.plot()
    # plt.plot(his.history['dice_coef'], label='train dice_coef')
    # plt.plot(his.history['val_dice_coef'], c='g', label='val dice_coef')
    # plt.title('train  and val dice_coef')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    #
    # plt.plot()
    # plt.plot(his.history['sensitivity'], label='train sensitivity')
    # plt.plot(his.history['val_sensitivity'], c='g', label='val sensitivity')
    # plt.title('train  and val sensitivity')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt.plot()
    # plt.plot(his.history['specificity'], label='train specificity')
    # plt.plot(his.history['val_specificity'], c='g', label='val specificity')
    # plt.title('train  and val specificity')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt.plot()
    # plt.plot(his.history['f1score'], label='train f1score')
    # plt.plot(his.history['val_f1score'], c='g', label='val f1score')
    # plt.title('train  and val f1score')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    #
    # plt.plot()
    # plt.plot(his.history['precision'], label='train precision')
    # plt.plot(his.history['val_precision'], c='g', label='val_precision')
    # plt.title('train  and val precision')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt.plot()
    # plt.plot(his.history['recall'], label='train recall')
    # plt.plot(his.history['val_recall'], c='g', label='val_recall')
    # plt.title('train  and val recall')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    #
    # plt.plot()
    # plt.plot(his.history['mean_iou'], label='Train mean_iou')
    # plt.plot(his.history['val_mean_iou'], c='g', label='val_mean_iou')
    # plt.title('train and val mean_iou')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()
    #


if __name__ == "__main__":
    # model = build_res_unet()
    # print(model.summary())
    train()
