"""Prediction script.
"""

from gan import build_res_unet, preprocess
import numpy as np
from PIL import Image
import os
from skimage.transform import resize


data_path = "./data/file/"


def load_train_data():
    imgs_train = np.load(data_path + "train.npy")
    imgs_mask_train = np.load(data_path + "train_mask.npy")
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load(data_path + "test.npy")
    imgs_mask_test = np.load(data_path + "test_mask.npy")
    return imgs_test, imgs_mask_test


path = "./data/file/"
path_to_trained_generator_weights = "./data/trained_models/unet/"


def predict():
    model = build_res_unet()

    path_to_save_results = path + "test/"
    imgs_test, imgs_test_mask = load_test_data()

    mean = np.mean(imgs_test)
    std = np.std(imgs_test)

    imgs_test = preprocess(imgs_test)
    imgs_test_mask = preprocess(imgs_test_mask)

    imgs_test_source = imgs_test.astype("float32")
    imgs_test_source -= mean
    imgs_test_source /= std

    imgs_test_mask = imgs_test_mask.astype("float32")
    imgs_test_mask /= 255.0  # scale masks to [0, 1]

    print("Loading saved weights...")
    print("-" * 30)
    model.load_weights(path_to_trained_generator_weights + "unet_weights_11.h5")
    print("Predicting masks on test data...")
    print("-" * 30)
    imgs_mask_predict = model.predict(imgs_test_source, verbose=1)
    res = model.evaluate(imgs_test_source, imgs_test_mask, batch_size=1, verbose=1)

    print(" Test loss:", res[0])
    print("Test accuracy:", res[1])
    print(" Test dice_coef:", res[2])
    print(" Test sensitivity:", res[3])
    print(" Test specificity:", res[4])
    print("Test f1score:", res[5])
    print(" Test precision:", res[6])
    print("Test recall:", res[7])
    print(" Test mean_iou:", res[8])

    res_loss = np.array(res)
    np.save(path + "test_predict.npy", imgs_mask_predict)
    np.savetxt(path + "res_loss.txt", res_loss)

    predicted_masks = np.load(path + "test_predict.npy")
    predicted_masks *= 255
    imgs_test, imgs_test_mask = load_test_data()

    for i in range(imgs_test.shape[0]):
        img = resize(imgs_test[i], (128, 128), preserve_range=True)
        img_mask = resize(imgs_test_mask[i], (128, 128), preserve_range=True)
        im_test_source = Image.fromarray(img.astype(np.uint8))
        im_test_masks = Image.fromarray((img_mask.squeeze()).astype(np.uint8))
        im_test_predict = Image.fromarray(
            (predicted_masks[i].squeeze()).astype(np.uint8)
        )
        im_test_source_name = "Test_Image_" + str(i + 1) + ".png"
        im_test_predict_name = "Test_Image_" + str(i + 1) + "_Predict.png"
        im_test_gt_mask_name = "Test_Image_" + str(i + 1) + "_OriginalMask.png"
        im_test_source.save(os.path.join(path_to_save_results, im_test_source_name))
        im_test_predict.save(os.path.join(path_to_save_results, im_test_predict_name))
        im_test_masks.save(os.path.join(path_to_save_results, im_test_gt_mask_name))

    message = "Successfully Saved Results to " + path_to_save_results
    print(message)


if __name__ == "__main__":
    predict()
