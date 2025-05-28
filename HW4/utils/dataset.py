import os
import cv2
import random
import numpy as np
import skimage.io as io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from utils.utils import crop_img


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception("Invalid choice of image transformation")
    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


class CustomTrainDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, mode="train"):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode  # discard

        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'."

        self.clean_list = []
        self.degraded_list = []

        self._get_img_pairs()
        # print(f"Total number of images: {len(self.clean_list)}")

        # self.transform = T.Compose(
        #     [
        #         T.ToPILImage(),
        #         T.RandomCrop(patch_size),
        #     ]
        # )
        self.totensor = T.ToTensor()

    def _get_img_pairs(self):
        degraded_path = os.path.join(self.root_dir, "degraded")
        clean_path = os.path.join(self.root_dir, "clean")

        clean_rain_list, clean_snow_list = [], []
        degraded_rain_list, degraded_snow_list = [], []

        for name in os.listdir(degraded_path):
            if "rain" in name:
                clean_name = name.replace("rain", "rain_clean")
                # degraded_rain_list.append(os.path.join(degraded_path, name))
                # clean_rain_list.append(os.path.join(clean_path, clean_name))
            elif "snow" in name:
                clean_name = name.replace("snow", "snow_clean")
                # degraded_snow_list.append(os.path.join(degraded_path, name))
                # clean_snow_list.append(os.path.join(clean_path, clean_name))
            else:
                continue

            self.degraded_list.append(os.path.join(degraded_path, name))
            self.clean_list.append(os.path.join(clean_path, clean_name))

        # rain_len = len(degraded_rain_list)
        # snow_len = len(degraded_snow_list)

        # if self.mode == "train":
        #     self.degraded_list = (
        #         degraded_rain_list[: int(0.9 * rain_len)]
        #         + degraded_snow_list[: int(0.9 * snow_len)]
        #     )
        #     self.clean_list = (
        #         clean_rain_list[: int(0.9 * rain_len)]
        #         + clean_snow_list[: int(0.9 * snow_len)]
        #     )
        # else:  # mode == "val"
        #     self.degraded_list = (
        #         degraded_rain_list[int(0.9 * rain_len) :]
        #         + degraded_snow_list[int(0.9 * snow_len) :]
        #     )
        #     self.clean_list = (
        #         clean_rain_list[int(0.9 * rain_len) :]
        #         + clean_snow_list[int(0.9 * snow_len) :]
        #     )

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[
            ind_H : ind_H + self.patch_size, ind_W : ind_W + self.patch_size
        ]
        patch_2 = img_2[
            ind_H : ind_H + self.patch_size, ind_W : ind_W + self.patch_size
        ]

        return patch_1, patch_2

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        degraded = self.degraded_list[idx]
        clean = self.clean_list[idx]

        degrad_img = crop_img(np.array(Image.open(degraded).convert("RGB")), base=16)
        clean_img = crop_img(np.array(Image.open(clean).convert("RGB")), base=16)
        clean_name = os.path.basename(clean)

        if self.mode == "train":
            degrad_patch, clean_patch = random_augmentation(
                *self._crop_patch(degrad_img, clean_img)
            )
        else:  # mode == "val"
            degrad_patch, clean_patch = degrad_img, clean_img

        degrad_patch = self.totensor(degrad_patch)
        clean_patch = self.totensor(clean_patch)

        return [clean_name, idx], degrad_patch, clean_patch


class CustomTestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.degraded_list = []
        self._get_test_imgs()
        # print(f"Total number of images: {len(self.degraded_list)}")

        self.totensor = T.ToTensor()

    def _get_test_imgs(self):
        extensions = ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG", "bmp", "BMP"]
        for name in os.listdir(self.root_dir):
            if any(name.lower().endswith(ext) for ext in extensions):
                self.degraded_list.append(os.path.join(self.root_dir, name))

    def __len__(self):
        return len(self.degraded_list)

    def __getitem__(self, idx):
        degraded = self.degraded_list[idx]
        degrad_img = crop_img(np.array(Image.open(degraded).convert("RGB")), base=16)
        degrad_name = os.path.basename(degraded)
        degrad_img = self.totensor(degrad_img)

        return [degrad_name], degrad_img


if __name__ == "__main__":
    path = "hw4_dataset/train"
    dataset = CustomTrainDataset(root_dir=path)

    # path = "hw4_dataset/test/degraded"
    # dataset = CustomTestDataset(root_dir=path)

    # for i in range(len(dataset)):
    #     l, d, c = dataset[i]
    #     print(l[0], d.shape)
    #     break
