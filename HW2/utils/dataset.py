import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from albumentations.pytorch.transforms import ToTensorV2


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]

        return image, target


def get_transforms():
    train_tfm = T.Compose(
        [
            # RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    def __init__(self, root_dir, annotations_file=None, transform=None, test=False):
        self.root_dir = root_dir
        self.json_file = annotations_file
        self.transform = transform
        self.test = test

        if not self.test:
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.images = data["images"]
                # self.annotations = data["annotations"]
                self.categories = data["categories"]

                self.annotations_by_id = defaultdict(list)
                for ann in data["annotations"]:
                    self.annotations_by_id[ann["image_id"]].append(ann)
        else:
            self.images = os.listdir(self.root_dir)
            self.images = sorted(self.images, key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        # return 1000
        return len(self.images)

    def __getitem__(self, idx):
        if not self.test:
            img_id = self.images[idx]["id"]
            img_path = os.path.join(self.root_dir, self.images[idx]["file_name"])
            image = Image.open(img_path).convert("RGB")

            boxes = [
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
                for ann in self.annotations_by_id[img_id]
            ]
            labels = [ann["category_id"] for ann in self.annotations_by_id[img_id]]

            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            target = {
                "boxes": torch.from_numpy(boxes),  # Hint: shape -> (N, 4)
                "labels": torch.from_numpy(labels),  # Hint: shape -> (N,)
            }

            if self.transform:
                image, target = self.transform(image, target)

            return image, target

        else:
            img_path = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, idx


if __name__ == "__main__":
    train_tfm, test_tfm = get_transforms()

    dataset = CustomDataset(
        "nycu-hw2-data/valid",
        annotations_file="nycu-hw2-data/valid.json",
        transform=train_tfm,
        test=False,
    )

    # plot image, boxes, and labels
    for i, (img, target) in enumerate(dataset):
        if i == 5:
            break

        img = img.permute(1, 2, 0).numpy()
        H, W = img.shape[:2]
        img = (img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        for box in boxes:
            cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0),
                1,
            )

        plt.imshow(img)
        plt.savefig(f"img_{i}.png")
        plt.close()

    # sizes = [img.shape[-2:] for img, target in dataset]
    # heights = [h for h, w in sizes]
    # widths = [w for h, w in sizes]
    # plt.hist(heights, bins=20)

    # # save
    # plt.savefig("height.png")
    # plt.close()
    # plt.hist(widths, bins=20)
    # plt.savefig("width.png")
    # plt.close()

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=1,
    #     collate_fn=collate_fn,
    # )

    # for images, targets in dataloader:
    #     print(images[0].shape)
    #     print(targets[0])
    #     break
