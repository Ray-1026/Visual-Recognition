import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils


def decode_maskobj(mask_obj):
    return maskUtils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def read_maskfile(filepath):
    mask_array = io.imread(filepath)
    return mask_array
