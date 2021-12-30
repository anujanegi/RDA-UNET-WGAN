"""Helper script to outline prediction boundaries.
"""

import os
from PIL import Image, ImageFilter

path_to_predictions = "./data/file/test/"
path_to_save = "./data/file/test_outlines"


def overlay(background, mask):
    background_image = Image.open(os.path.join(path_to_predictions, background))
    mask_image = Image.open(os.path.join(path_to_predictions, mask))

    background_image.save(os.path.join(path_to_save, background))

    outline = mask_image.filter(ImageFilter.FIND_EDGES)
    background_image.paste(outline, (0, 0), outline)

    background_image.save(os.path.join(path_to_save, mask))


def draw_outline():
    images = os.listdir(path_to_predictions)
    for image_name in images:
        print(image_name)
        if ("Mask" in image_name) or ("Predict" in image_name):
            continue
        print("hi")
        image_gt = image_name.split(".")[0] + "_OriginalMask.png"
        image_predicted_mask = image_name.split(".")[0] + "_Predict.png"
        overlay(image_name, image_gt)
        overlay(image_name, image_predicted_mask)


draw_outline()
