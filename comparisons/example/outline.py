import os
from PIL import Image, ImageFilter
import numpy as np

path_to_predictions = "./"
path_to_save = "./outline/"


def overlay(background, mask):
    background_image = Image.open(os.path.join(path_to_predictions, background))
    mask_image = Image.open(os.path.join(path_to_predictions, mask))

    outline = mask_image.filter(ImageFilter.FIND_EDGES)

    #
    # outline = outline.convert('RGBA')
    # data = np.array(outline)   # "data" is a height x width x 4 numpy array
    # red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
    # white_areas = (red != 0) & (blue != 0) & (green != 0)
    # data[..., :-1][white_areas.T] = (255, 0, 0)
    # outline = Image.fromarray(data)
    background_image.paste(outline,(0,0),outline)

    background_image.save(os.path.join(path_to_save, mask))

def draw_outline():
	
	image_overlay = 'Test_Image_44_epoch_11_Predict.png'
	image_name = 'Test_Image_44.png'
	overlay(image_name, image_overlay)

draw_outline()
