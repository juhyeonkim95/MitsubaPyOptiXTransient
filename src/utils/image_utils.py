import numpy as np
import cv2 as cv
from PIL import Image
import glob
import matplotlib.pyplot as plt
import os


def load_exr_image(path, invert=False):
    image = cv.imread(path, -1).astype(np.float32)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGRA)
    if invert:
        image = cv.flip(image, 0)
    return image


def convert_image_to_uint(image):
    x = np.copy(image)
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


def save_image(image, file_path):
    x = convert_image_to_uint(image)
    new_im = Image.fromarray(x)
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if file_path.endswith(".png"):
        new_im.save("%s" % file_path)
    else:
        new_im.save("%s.png" % file_path)


def save_image_numpy(image, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if file_path.endswith(".npy"):
        np.save("%s" % file_path, image)
    else:
        np.save("%s.npy" % file_path, image)


def load_reference_image(parent_folder, name):
    target_name = "%s/%s_*.png" % (parent_folder, name)
    files = glob.glob(target_name)
    load_file = files[0]
    return load_image(load_file)


def load_image(path):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, 0:3]
    image /= 255.0
    return image
