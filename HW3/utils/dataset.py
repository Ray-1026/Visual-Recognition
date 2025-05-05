import os
import cv2
import json
import torch
import numpy as np
import skimage.io as io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                target["masks"] = F.hflip(target["masks"])

        return image, target


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                target["masks"] = F.vflip(target["masks"])

        return image, target


def get_transforms():
    train_tfm = T.Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            T.ToTensor(),
        ]
    )

    test_tfm = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    return train_tfm, test_tfm


def collate_fn(batch):
    return tuple(zip(*batch))


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False, test_json=None):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if not self.is_test:
            self.image_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
            print(f"Found {len(self.image_dirs)} directories in {root_dir}")
        else:
            self.image_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
            print(f"Found {len(self.image_dirs)} directories in {root_dir}")
            with open(test_json, "r") as f:
                self.test_json = json.load(f)
                self.map_imgname_to_id = {
                    img["file_name"]: img["id"] for img in self.test_json
                }

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        if not self.is_test:
            # --- load image ---
            img_path = os.path.join(self.image_dirs[idx], "image.tif")
            image = Image.open(img_path).convert("RGB")

            # --- load masks ---
            masks = []
            boxes = []
            labels = []
            for mask_id in range(1, 5):
                mask_path = os.path.join(self.image_dirs[idx], f"class{mask_id}.tif")
                if not os.path.exists(mask_path):
                    continue

                mask = io.imread(mask_path)

                # --- create boxes and labels ---
                unique_vals = np.unique(mask)
                for val in unique_vals[1:]:  # Ignore the background (0)
                    mask_binary = mask == val
                    y, x = np.where(mask_binary == 1)
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(mask_id)
                    masks.append(mask_binary[None, ...])  # Shape: (1, H, W)

            masks = np.concatenate(masks, axis=0).astype(np.uint8)  # Shape: (N, H, W)
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            target = {
                "image_id": torch.tensor(idx),
                "boxes": torch.from_numpy(boxes),  # Shape: (N, 4)
                "labels": torch.from_numpy(labels),  # Shape: (N,)
                "masks": torch.from_numpy(masks),  # Shape: (N, H, W)
            }

            if self.transform:
                image, target = self.transform(image, target)

            return image, target

        else:
            image_name = self.image_dirs[idx].split("/")[-1]
            image = Image.open(self.image_dirs[idx]).convert("RGB")
            image_id = self.map_imgname_to_id.get(image_name, None)

            if self.transform:
                image = self.transform(image)

            return image, image_id


if __name__ == "__main__":
    path = "data/test_release"
    json_file = "data/test_image_name_to_ids.json"
    dataset = CustomDataset(root_dir=path, is_test=True, test_json=json_file)

    for i in range(len(dataset)):
        img, ids = dataset[i]
        img = T.ToTensor()(img)
        print(img.shape)
        # break
