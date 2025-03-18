import os
import pandas as pd
import time
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as transforms

from models.resnet import ResNet101, ResNet152, ResNeXt101, ResNeXt50
from models.resnest import ResNest50, ResNest101
from models.cbam import CBAMResNeXt
from models.senet import SENeXt50
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.losses import LabelSmoothing, NLLMultiLabelSmooth
from utils.dataset import (
    data_prefetcher,
    get_transforms,
    TestingDataset,
    TrainDataset,
    MixUpWrapper,
    CutMix,
)


model_dict = {
    "resnest50": ResNest50,
    "resnest101": ResNest101,
    "senext50": SENeXt50,
}


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        self.start_epoch = 0

        # model
        self.model = model_dict[self.opt.model](
            num_classes=100).to(self.device)

        # get transforms
        self.train_tfm, self.test_tfm = get_transforms()

        if not self.opt.eval_only:
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = LabelSmoothing()

            parameters = self.model.named_parameters()
            param_dict = {}
            for k, v in parameters:
                param_dict[k] = v
            new_params = [v for n, v in param_dict.items()
                          if "fc" in n or "cbam" in n]
            rest_params = [
                v for n, v in param_dict.items() if "fc" not in n and "cbam" not in n
            ]

            self.optimizer = torch.optim.AdamW(
                # self.model.parameters(),
                # lr=self.opt.learning_rate,
                [
                    {"params": new_params, "lr": self.opt.learning_rate},
                    {"params": rest_params, "lr": self.opt.learning_rate / 10},
                ],
                weight_decay=self.opt.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.opt.num_epochs,
                eta_min=1e-6,
            )

            self.train_loader = DataLoader(
                TrainDataset(self.opt.train_data_dir, self.train_tfm),
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )

            # # mixup
            # self.train_loader = MixUpWrapper(
            #     self.train_loader, alpha=0.2, device=self.device
            # )

            self.val_loader = DataLoader(
                TrainDataset(self.opt.valid_data_dir, self.test_tfm),
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            # tensorboard
            self.writer = SummaryWriter(
                log_dir=(
                    f"logs/{time.strftime('%Y-%m-%dT%H:%M:%S')}"
                    if self.opt.log_dir is None
                    else f"logs/{self.opt.log_dir}"
                )
            )

            # create ckpt dir
            os.makedirs(self.opt.save_ckpt_dir, exist_ok=True)

            # record best accuracy
            self.best_acc = 0

        else:   # eval only
            self.test_dataset = TestingDataset(
                self.opt.test_data_dir, self.test_tfm)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=8,
            )

    def print_params(self):
        # print network params
        print(
            f"Number of model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M"
        )

    def train(
        self,
        val_every_epochs=1,
    ):
        self.print_params()

        scaler = GradScaler()

        for epoch in range(self.start_epoch, self.opt.num_epochs):
            self.model.train()
            train_losses = []

            prefetcher = data_prefetcher(self.train_loader)  # prefetch data
            imgs, labels = prefetcher.next()
            while imgs is not None:
                # img, label = CutMix()(imgs, labels)

                img = imgs.to(self.device)
                label = labels.to(self.device)

                with autocast(device_type="cuda"):  # mixed precision
                    output = self.model(img)
                    loss = self.criterion(output, label)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # record loss
                train_losses.append(loss.item())

                imgs, labels = prefetcher.next()

            self.writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
            self.writer.add_scalar(
                "Learning_Rate_1", self.optimizer.param_groups[0]["lr"], epoch
            )
            self.writer.add_scalar(
                "Learning_Rate_2", self.optimizer.param_groups[1]["lr"], epoch
            )

            self.scheduler.step()

            if (epoch + 1) % val_every_epochs == 0:
                acc, val_loss = self.validate(epoch + 1)

            print(
                f"Epoch {epoch}, Loss: {np.mean(train_losses):.4f}, val_loss: {val_loss:.4f}, Acc: {acc:.4f}, Best Acc: {self.best_acc:.4f}"
            )

        self.writer.close()

        # save last ckpt
        self.save_weights(epoch)

        print("Training complete")

    def validate(self, epoch):
        self.model.eval()

        acc = 0
        val_loss = []
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(imgs)
                loss = self.criterion(output, labels)
                val_loss.append(loss.item())

                acc += (output.argmax(1) == labels).sum().item()

        acc /= len(self.val_loader.dataset)
        self.writer.add_scalar("Accuracy", acc, epoch)

        if acc > self.best_acc:
            self.best_acc = acc

        return acc, np.mean(val_loss)

    def eval(self):
        self.model.eval()
        self.load_weights()

        preds = []
        with torch.no_grad():
            for imgs in tqdm(self.test_loader):
                imgs = imgs.to(self.device)
                output = self.model(imgs)
                preds.extend(output.argmax(1).cpu().numpy().tolist())

        submission = pd.DataFrame(
            {"image_name": self.test_loader.dataset.names, "pred_label": preds}
        )
        submission.to_csv("prediction.csv", index=False)

        # zip
        os.system("zip solution.zip prediction.csv")

    def save_weights(self, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                # "optimizer": self.optimizer.state_dict(),
                # "epoch": epoch,
            },
            os.path.join(self.opt.save_ckpt_dir, f"{self.opt.model}.pth"),
        )

    def load_weights(self):
        ckpt = torch.load(self.opt.ckpt_name)
        self.model.load_state_dict(ckpt["model"])
