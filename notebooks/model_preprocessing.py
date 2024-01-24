import numpy as np
from PIL import Image
import tensorflow as tf

def convert_image_to_array(path, target_height, target_width):
    img = np.asarray(Image.open(path), dtype=np.float32)
    img = np.stack((img,)*3, axis=-1)
    img /= 255.
    img = tf.image.resize_with_pad(img, target_height=target_height, target_width=target_width)
    return img

def model_predict(path, model, target_height, target_width):
    x = convert_image_to_array(path=path, target_height=target_height, target_width=target_width)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)