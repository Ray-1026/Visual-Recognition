import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


def get_transforms():
    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.TrivialAugmentWide(),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomRotation(45),
            # transforms.RandomApply(
            #     [
            #         transforms.ColorJitter(
            #             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            #         )
            #     ],
            #     p=0.5,
            # ),
            # transforms.RandomApply(
            #     [transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )

    test_tfm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tfm, test_tfm


def CutMix(num_classes=100):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    """
    return transforms.Compose(
        [
            transforms.CutMix(num_classes=num_classes),
        ]
    )


def CutMix_or_MixUp(num_classes=100):
    cutmix = transforms.CutMix(num_classes=num_classes)
    mixup = transforms.MixUp(num_classes=num_classes)
    cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup


class MixUpWrapper(object):
    def __init__(self, dataloader, alpha=0.4, device="cuda", num_classes=100):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device

    def mixup_loader(self, loader):
        def mixup(alpha, num_classes, data, target):
            with torch.no_grad():
                bs = data.size(0)
                c = np.random.beta(alpha, alpha)
                perm = torch.randperm(bs).cuda()

                md = c * data + (1 - c) * data[perm, :]
                mt = c * target + (1 - c) * target[perm, :]
                return md, mt

        for input, target in loader:
            input, target = input.cuda(self.device), target.cuda(self.device)
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)


class data_prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_batch[0].cuda(
                non_blocking=True).float()
            self.next_target = self.next_batch[1].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        data
        ├── train
        |   ├── class1
        |   |   ├── xxxxx.jpg
        |   |   ├── ...
        |   |   └── yyyyy.jpg
        |   ├── class2
        |   |   ├── xxxxx.jpg
        |   |   ├── ...
        |   |   └── yyyyy.jpg
        |   └── ...
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(self.img_dir):
            for image in glob.glob(f"{self.img_dir}/{label}/*"):
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]).convert("RGB"))
        label = int(self.labels[idx])
        return (image, label)


class TestingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        data
        ├── test
        |   ├── xxxxx.jpg
        |   ├── ...
        |   └── yyyyy.jpg
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.names = []

        self.images = sorted(glob.glob(f"{self.img_dir}/*"))
        self.names = [os.path.basename(image)[:-4] for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getnames__(self):
        return self.names

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]).convert("RGB"))
        return image
