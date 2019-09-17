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
from keras import objectives
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D,ZeroPadding2D,Convolution2D, Conv2DTranspose,MaxPooling2D,add, UpSampling2D, multiply,Dropout, BatchNormalization, LeakyReLU, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input, GaussianNoise
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
from keras import backend as K,models
from skimage.io import imsave
from keras.layers.merge import Concatenate

from gan_utils import imgs2discr, imgs2gan

def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.

img_rows =128
img_cols =128

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


data_path = './data/file/'
path_to_save_results= data_path+"predictions/"
path_to_trained_generator_weights = './data/trained_models/'

# data_path = '/Users/xuchenyang/Documents/sec_exp/file/'


def load_train_data():
    imgs_train = np.load(data_path + 'train.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + 'validation.npy')
    imgs_mask_valid = np.load(data_path + 'validation_mask.npy')
    return imgs_valid, imgs_mask_valid


def load_test_data():
    imgs_test = np.load(data_path + 'test.npy')
    imgs_mask_test = np.load(data_path + 'test_mask.npy')
    return imgs_test, imgs_mask_test


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

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




def sensitivity(y_true,y_pred):
    true_positives=tf.reduce_sum(tf.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positives=tf.reduce_sum(tf.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives+K.epsilon())
def specificity(y_true,y_pred):
    true_negatives=tf.reduce_sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives=tf.reduce_sum(K.round(K.clip((1-y_true), 0, 1)))
    return true_negatives / (possible_negatives+K.epsilon())



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

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def unet4(upsamezhi,chuandi,F_g, F_l,F_int):

    up = Conv2D(F_g, (1, 1), activation='relu', padding='same')(upsamezhi)
    up= BatchNormalization()(up)

    down= Conv2D(F_l, (1, 1), activation='relu', padding='same')(chuandi)
    down= BatchNormalization()(down)
    sumadd=add([up,down])
    sumadd = Activation(activation='relu')(sumadd)



    jihe=Conv2D(F_int, (1, 1), activation='relu', padding='same')(sumadd)
    sumhalf= BatchNormalization()(jihe)


    sum_1=Conv2D(1, (1, 1), activation='sigmoid', padding='same')(sumhalf)
    sum_1= BatchNormalization()(sum_1)

    xinchuandi=multiply([chuandi,sum_1])
    return xinchuandi

def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

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
    xin_encoder_1=unet4(main_path,from_encoder[4],256, 256,128)
    main_path = concatenate([main_path, xin_encoder_1], axis=3)
    main_path = res_block(main_path, [512, 512], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_2=unet4(main_path,from_encoder[3],128, 128,64)
    main_path = concatenate([main_path, xin_encoder_2], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_3=unet4(main_path,from_encoder[2],64, 64,32)
    main_path = concatenate([main_path, xin_encoder_3], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_4=unet4(main_path,from_encoder[1],32, 32,16)
    main_path = concatenate([main_path, xin_encoder_4], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    xin_encoder_5=unet4(main_path,from_encoder[0],16, 16,8)
    main_path = concatenate([main_path, xin_encoder_5], axis=3)
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])


    return main_path


def build_res_unet( ):
    #inputs = Input(shape=input_shape)
    inputs = Input(shape=(img_rows, img_cols, 1))

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[4], [512,512], [(2, 2), (1, 1)])


    bottle = bottleneck(path, filters_bottleneck=256, mode='cascade')

    path = decoder(bottle, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='hard_sigmoid',padding='same')(path)
    model=Model(input=inputs, output=path)

    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])

    return model

def make_trainable(network, value):
    """
    If False, it fixes the network and it is not trainable (the weights are frozen)
    If True, the network is trainable (the weights can be updated)
    :param net: network
    :param val: boolean to make the network trainable or not
    """
    network.trainable = value
    for l in network.layers:
        l.trainable = value


def build_discriminator():

    k = 3  # kernel size
    s = 2  # stride
    n_filters = 32  # number of filters
    inputs = Input(shape=(img_rows, img_cols, 2))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    gap = GlobalAveragePooling2D()(conv5)
    outputs = Dense(1, activation='sigmoid')(gap)

    model = Model(inputs, outputs)

    # loss of the discriminator. it is a binary loss
    def discriminator_loss(y_true, y_pred):
        #L = objectives.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        loss = objectives.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return loss

    model.compile(optimizer=Adam(lr=1e-4), loss=discriminator_loss, metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])

    return model


def build_gan(generator, discriminator):

    image = Input(shape=(img_rows, img_cols, 1 ))
    mask = Input(shape=(img_rows, img_cols, 1 ))

    fake_mask = generator(image)
    fake_pair = Concatenate(axis=3)([image, fake_mask])

    gan = Model([image, mask], discriminator(fake_pair))


    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    # gan.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])
    gan.compile(optimizer=Adam(lr=1e-4), loss=wasserstein_loss, metrics=['accuracy',dice_coef, sensitivity,specificity,f1score,precision,recall,mean_iou])
    return gan

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train():

    # get data
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
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

    imgs_train = imgs_train.astype('float32')
    imgs_valid = imgs_valid.astype('float32')

    # print(imgs_train.shape)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    val_mean = np.mean(imgs_valid)
    val_std = np.std(imgs_valid)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid -= val_mean
    imgs_valid /= val_std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_valid = imgs_mask_valid.astype('float32')
    imgs_mask_valid /= 255.

    # preparing test data
    imgs_test, imgs_test_mask = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test_mask = preprocess(imgs_test_mask)
    imgs_test_source = imgs_test.astype('float32')
    imgs_test_source -= mean
    imgs_test_source /= std
    imgs_test_mask = imgs_test_mask.astype('float32')
    imgs_test_mask /= 255.  # scale masks to [0, 1]


    # Get the generator model i.e UNet
    print('-' * 30)
    print('Creating and compiling the generator model...')
    print('-' * 30)
    model_generator = build_res_unet()
    model_generator.load_weights(path_to_trained_generator_weights + "RDAunet.hdf5")
    print(model_generator.summary())


    # Get the discriminator model
    print('-' * 30)
    print('Creating and compiling discriminator model...')
    print('-' * 30)
    model_discriminator = build_discriminator()
    print(model_discriminator.summary())
    make_trainable(model_discriminator, False)    # make the discriminator untrainable


    # Get the GAN model
    print('-' * 30)
    print('Creating and compiling GAN model...')
    print('-' * 30)
    model_gan = build_gan(model_generator, model_discriminator)
    print(model_gan.summary())


    def get_batch_train():
        idx = np.random.randint(0,imgs_train.shape[0], batch_size)
        imgs = imgs_train[idx]
        masks = imgs_mask_train[idx]
        return imgs, masks

    def get_batch_valid():
        idx = np.random.randint(0,imgs_valid.shape[0], batch_size)
        imgs = imgs_valid[idx]
        masks = imgs_mask_valid[idx]
        return imgs, masks

    n_rounds = 12 # number of rounds to apply adversarial training
    batch_size = 32

    # Getting data and its shape
    print("Getting train and validation data...")

    images_train = imgs_train
    labels_train = imgs_mask_train
    n_files_train = imgs_train.shape[0]
    images_val = imgs_valid
    labels_val = imgs_mask_valid
    n_files_val = imgs_valid.shape[0]
    # steps_per_epoch_d = (n_files_train//batch_size +1)
    # steps_per_epoch_g = (n_files_train//batch_size +1)
    # steps_val_d = (n_files_val//batch_size +1)
    # steps_val_g = (n_files_val//batch_size +1)
    steps_per_epoch_d = 1
    steps_per_epoch_g = 1
    steps_val_d = 1
    steps_val_g = 1

    # to show the progression of the losses
    val_gan_loss = np.zeros(n_rounds)
    val_gan_acc = np.zeros(n_rounds)
    val_gan_dice_coeff = np.zeros(n_rounds)
    val_gan_sensitivity = np.zeros(n_rounds)
    val_gan_specificity = np.zeros(n_rounds)
    val_gan_f1score = np.zeros(n_rounds)
    val_gan_precision = np.zeros(n_rounds)
    val_gan_recall = np.zeros(n_rounds)
    val_gan_mean_iou = np.zeros(n_rounds)

    val_discr_loss = np.zeros(n_rounds)
    val_discr_acc = np.zeros(n_rounds)
    val_discr_dice_coeff = np.zeros(n_rounds)
    val_discr_sensitivity = np.zeros(n_rounds)
    val_discr_specificity = np.zeros(n_rounds)
    val_discr_f1score = np.zeros(n_rounds)
    val_discr_precision = np.zeros(n_rounds)
    val_discr_recall = np.zeros(n_rounds)
    val_discr_mean_iou = np.zeros(n_rounds)

    gan_loss = np.zeros(n_rounds)
    gan_acc = np.zeros(n_rounds)
    gan_dice_coeff = np.zeros(n_rounds)
    gan_sensitivity = np.zeros(n_rounds)
    gan_specificity = np.zeros(n_rounds)
    gan_f1score = np.zeros(n_rounds)
    gan_precision = np.zeros(n_rounds)
    gan_recall = np.zeros(n_rounds)
    gan_mean_iou = np.zeros(n_rounds)

    discr_loss = np.zeros(n_rounds)
    discr_acc = np.zeros(n_rounds)
    discr_dice_coeff = np.zeros(n_rounds)
    discr_sensitivity = np.zeros(n_rounds)
    discr_specificity = np.zeros(n_rounds)
    discr_f1score = np.zeros(n_rounds)
    discr_precision = np.zeros(n_rounds)
    discr_recall = np.zeros(n_rounds)
    discr_mean_iou = np.zeros(n_rounds)

    print('-' * 30)
    print('GAN training...')
    print('-' * 30)
    #earlystopper=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    # his = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=2, verbose=1, shuffle=True,
    #                 validation_data=(imgs_valid, imgs_mask_valid), callbacks=[model_checkpoint])

    for n_round in range(n_rounds):
        print("Training...")
        # train Discriminator
        make_trainable(model_discriminator, True)
        for i in range(steps_per_epoch_d):
            image_batch, labels_batch = get_batch_train()
            pred = model_generator.predict(image_batch)
            img_discr_batch, lab_discr_batch = imgs2discr(image_batch, labels_batch, pred)
            loss, acc, dice_coef, sensitivity, specificity, f1score, precision, recall, mean_iou = model_discriminator.train_on_batch(img_discr_batch, lab_discr_batch)
        discr_loss[n_round] = loss
        discr_acc[n_round] = acc
        discr_dice_coeff[n_round] = dice_coef
        discr_sensitivity[n_round] = sensitivity
        discr_specificity[n_round] = specificity
        discr_f1score[n_round] = f1score
        discr_precision[n_round] = precision
        discr_recall[n_round] = recall
        discr_mean_iou[n_round] = mean_iou
        print("DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss))

        # train GAN
        make_trainable(model_discriminator, False)
        for i in range(steps_per_epoch_g):
            image_batch, labels_batch = get_batch_train()
            img_gan_batch, lab_gan_batch = imgs2gan(image_batch, labels_batch)
            loss, acc, dice_coef, sensitivity, specificity, f1score, precision, recall, mean_iou = model_gan.train_on_batch(img_gan_batch, lab_gan_batch)
        gan_loss[n_round] = loss
        gan_acc[n_round] = acc
        gan_dice_coeff[n_round] = dice_coef
        gan_sensitivity[n_round] = sensitivity
        gan_specificity[n_round] = specificity
        gan_f1score[n_round] = f1score
        gan_precision[n_round] = precision
        gan_recall[n_round] = recall
        gan_mean_iou[n_round] = mean_iou
        print("GAN Round: {0} -> Loss {1}".format((n_round + 1), loss))

        # evalutation on validation dataset
        print("Validating..")

        # Discriminator
        for i in range(steps_val_d):
            image_val_batch, labels_val_batch = get_batch_valid()
            pred = model_generator.predict(image_val_batch)
            img_discr_val, lab_discr_val = imgs2discr(image_val_batch, labels_val_batch, pred)
            loss, acc, dice_coef, sensitivity, specificity, f1score, precision, recall, mean_iou = model_discriminator.test_on_batch(img_discr_val, lab_discr_val)
        val_discr_loss[n_round] = loss
        val_discr_acc[n_round] = acc
        val_discr_dice_coeff[n_round] = dice_coef
        val_discr_sensitivity[n_round] = sensitivity
        val_discr_specificity[n_round] = specificity
        val_discr_f1score[n_round] = f1score
        val_discr_precision[n_round] = precision
        val_discr_recall[n_round] = recall
        val_discr_mean_iou[n_round] = mean_iou
        print("DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss))

        # GAN
        for i in range(steps_val_g):
            image_val_batch, labels_val_batch = get_batch_valid()
            img_gan_val, lab_gan_val = imgs2gan(image_val_batch, labels_val_batch)
            loss, acc, dice_coef, sensitivity, specificity, f1score, precision, recall, mean_iou= model_gan.test_on_batch(img_gan_val, lab_gan_val)
        val_gan_loss[n_round] = loss
        val_gan_acc[n_round] = acc
        val_gan_dice_coeff[n_round] = dice_coef
        val_gan_sensitivity[n_round] = sensitivity
        val_gan_specificity[n_round] = specificity
        val_gan_f1score[n_round] = f1score
        val_gan_precision[n_round] = precision
        val_gan_recall[n_round] = recall
        val_gan_mean_iou[n_round] = mean_iou
        print("GAN Round: {0} -> Loss {1}".format((n_round + 1), loss))

        # save the weights of the unet
        if not os.path.exists(path_to_trained_generator_weights):
            os.makedirs(path_to_trained_generator_weights)
        model_gan.save_weights(os.path.join(path_to_trained_generator_weights + "/gan/", "gan_weights_{}.h5".format(n_round)))
        model_generator.save_weights(os.path.join(path_to_trained_generator_weights + "/unet/", "unet_weights_{}.h5".format(n_round)))

        # saving images generated
        imgs_mask_predict = model_generator.predict(imgs_test_source, verbose=1)
        np.save(data_path+'predict' + str(n_round) + '.npy', imgs_mask_predict)
        predicted_masks=np.load(data_path+'predict' + str(n_round) + '.npy')
        predicted_masks*=255
        if n_rounds%1 == 0:
                    print("Testing...")
                    imgs_test, imgs_test_mask = load_test_data()
                    for i in range(imgs_test.shape[0]):
                        img = resize(imgs_test[i], (128, 128), preserve_range=True)
                        img_mask = resize(imgs_test_mask[i], (128, 128), preserve_range=True)
                        im_test_source = Image.fromarray(img.astype(np.uint8))
                        im_test_masks = Image.fromarray((img_mask.squeeze()).astype(np.uint8))
                        im_test_predict = Image.fromarray((predicted_masks[i].squeeze()).astype(np.uint8))
                        if(n_round==0):
                            im_test_source_name = "Test_Image_"+str(i+1)+".png"
                            im_test_gt_mask_name = "Test_Image_"+str(i+1)+"_OriginalMask.png"
                            im_test_source.save(os.path.join(path_to_save_results,im_test_source_name))
                            im_test_masks.save(os.path.join(path_to_save_results,im_test_gt_mask_name))
                        im_test_predict_name = "Test_Image_"+str(i+1)+"_epoch_"+str(n_round)+"_Predict.png"
                        im_test_predict.save(os.path.join(path_to_save_results,im_test_predict_name))


                    print("Testing done")
        # print(model_generator.train_history)

    # print the evolution of the loss
    print ("DISCR loss {}".format(discr_loss))
    print ("GAN loss {}".format(gan_loss))
    print ("DISCR validation loss {}".format(val_discr_loss))
    print ("GAN validation loss {}".format(val_gan_loss))

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
    plt.plot()
    plt.plot(gan_loss, label='train loss')
    plt.plot(val_gan_loss, c='g', label='val loss')
    plt.title('train and val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val loss.png')

    plt.plot()
    plt.plot(gan_acc, label='train accuracy')
    plt.plot(val_gan_acc, c='g', label='val accuracy')
    plt.title('train  and val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val accuracy.png')

    plt.plot()
    plt.plot(gan_dice_coeff, label='train dice_coef')
    plt.plot(val_gan_dice_coeff, c='g', label='val dice_coef')
    plt.title('train  and val dice_coef')
    plt.xlabel('epoch')
    plt.ylabel('dice coefficient')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val dice_coef.png')


    plt.plot()
    plt.plot(gan_sensitivity, label='train sensitivity')
    plt.plot(val_gan_sensitivity, c='g', label='val sensitivity')
    plt.title('train  and val sensitivity')
    plt.xlabel('epoch')
    plt.ylabel('sensitivity')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val sensitivity.png')

    plt.plot()
    plt.plot(gan_specificity, label='train specificity')
    plt.plot(val_gan_specificity, c='g', label='val specificity')
    plt.title('train  and val specificity')
    plt.xlabel('epoch')
    plt.ylabel('specificity')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val specificity.png')

    plt.plot()
    plt.plot(gan_f1score, label='train f1score')
    plt.plot(val_gan_f1score, c='g', label='val f1score')
    plt.title('train  and val f1score')
    plt.xlabel('epoch')
    plt.ylabel('f1score')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val f1score.png')


    plt.plot()
    plt.plot(gan_precision, label='train precision')
    plt.plot(val_gan_precision, c='g', label='val_precision')
    plt.title('train  and val precision')
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val precision.png')

    plt.plot()
    plt.plot(gan_recall, label='train recall')
    plt.plot(val_gan_recall, c='g', label='val_recall')
    plt.title('train  and val recall')
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val precision.png')


    plt.plot()
    plt.plot(gan_mean_iou, label='Train mean_iou')
    plt.plot(val_gan_mean_iou, c='g', label='val_mean_iou')
    plt.title('train and val mean_iou')
    plt.xlabel('epoch')
    plt.ylabel('mean_iou')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(data_path + 'gan train and val mean_iou.png')


if __name__ == '__main__':
    # model = build_res_unet()
    # print(model.summary())
    train()
