import os
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.faster_rcnn import CustomFasterRCNN
from utils.dataset import get_transforms, collate_fn, CustomDataset


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device

        self.model = CustomFasterRCNN(
            backbone=self.opt.backbone,
            box_score_threshold=self.opt.box_score_thresh,
            num_classes=self.opt.num_classes,
        ).to(self.device)

        self.train_tfm, self.test_tfm = get_transforms()

        if not self.opt.test_only:
            self.optimizer = self.get_optimizer()

            self.scheduler = None
            if self.opt.scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.opt.epochs,
                    eta_min=1e-6,
                )

            # prepare dataset
            self.train_dataset = CustomDataset(
                root_dir=self.opt.train_data_dir,
                annotations_file=self.opt.annotations_file_train,
                transform=self.train_tfm,
                test=False,
            )
            self.valid_dataset = CustomDataset(
                root_dir=self.opt.valid_data_dir,
                annotations_file=self.opt.annotations_file_valid,
                transform=self.test_tfm,
                test=False,
            )

            # dataloader
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=self.opt.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            self.val_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=self.opt.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
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

            # validation mAP record
            self.best_map = 0.0

        else:  # testing only
            self.test_dataset = CustomDataset(
                root_dir=self.opt.test_data_dir,
                annotations_file=None,
                transform=self.test_tfm,
                test=True,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=self.opt.num_workers,
                collate_fn=collate_fn,
            )

    def get_optimizer(self):
        """Get the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """

        # params = [
        #     {"params": self.model.model.rpn.parameters(), "lr": self.opt.lr},
        #     {"params": self.model.model.roi_heads.parameters(), "lr": self.opt.lr},
        #     {"params": self.model.model.backbone.parameters(), "lr": self.opt.lr * 0.1},
        # ]
        # params = [p for p in self.model.parameters() if p.requires_grad]

        if self.opt.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                # params,
                lr=self.opt.lr,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                # params,
                lr=self.opt.lr,
                momentum=0.9,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                # params,
                lr=self.opt.lr,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "nesterov":
            return torch.optim.SGD(
                self.model.parameters(),
                # params,
                lr=self.opt.lr,
                momentum=0.9,
                weight_decay=self.opt.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt.optimizer}")

    def train(
        self,
        val_every_epochs=1,
        # save_every_epochs=5,
    ):
        self.scaler = GradScaler() if self.opt.fp16 else None

        for epoch in range(1, self.opt.epochs + 1):
            self.model.train()
            train_losses = {
                "cls_loss": [],
                "box_loss": [],
                "obj_loss": [],
                "rpn_box_loss": [],
                "total_loss": [],
            }

            # for images, targets in self.train_loader:
            for images, targets in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.opt.epochs}",
                unit="batch",
                leave=False,  # do not leave the progress bar on the screen
            ):
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                self.optimizer.zero_grad()

                with autocast(
                    device_type=self.device.type, enabled=self.scaler is not None
                ):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if self.scaler is not None:  # fp16
                    self.scaler.scale(losses).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:  # fp32
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # record loss
                train_losses["cls_loss"].append(loss_dict["loss_classifier"].item())
                train_losses["box_loss"].append(loss_dict["loss_box_reg"].item())
                train_losses["obj_loss"].append(loss_dict["loss_objectness"].item())
                train_losses["rpn_box_loss"].append(
                    loss_dict["loss_rpn_box_reg"].item()
                )
                train_losses["total_loss"].append(losses.item())

            if self.scheduler is not None:
                self.scheduler.step()

            # validate
            if epoch % val_every_epochs == 0:
                map_result = self.validate(epoch)

                # print log
                print(
                    f"Epoch [{epoch}/{self.opt.epochs}], Loss: {np.mean(train_losses['total_loss']):.4f}, mAP: {map_result['map']:.4f}"
                )

                # save best model
                if map_result["map"] > self.best_map:
                    self.best_map = map_result["map"]
                    self.save_weights(epoch)

            # # save model
            # if epoch % save_every_epochs == 0:
            #     self.save_weights(epoch)

            # write to tensorboard
            self.writer.add_scalar(
                "Loss/cls_loss", np.mean(train_losses["cls_loss"]), epoch
            )
            self.writer.add_scalar(
                "Loss/box_loss", np.mean(train_losses["box_loss"]), epoch
            )
            self.writer.add_scalar(
                "Loss/obj_loss", np.mean(train_losses["obj_loss"]), epoch
            )
            self.writer.add_scalar(
                "Loss/rpn_box_loss", np.mean(train_losses["rpn_box_loss"]), epoch
            )
            self.writer.add_scalar(
                "Loss/total_loss", np.mean(train_losses["total_loss"]), epoch
            )
            self.writer.add_scalar(
                "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
            )
            self.writer.add_scalar("mAP/val", map_result["map"], epoch)

        # close tensorboard writer
        self.writer.close()

        # # save last ckpt
        # self.save_weights(epoch, last=True)
        print("Training complete")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        # mAP metric
        map_metric = MeanAveragePrecision()

        for images, targets in tqdm(
            self.val_loader,
            desc=f"Validation {epoch}/{self.opt.epochs}",
            unit="batch",
            leave=False,
        ):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            pred = self.model(images)

            # update mAP metric
            map_metric.update(pred, targets)

        result = map_metric.compute()
        return result

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.load_weights()

        preds = []  # task 1
        pred_vals = []  # task 2
        for images, idx in tqdm(
            self.test_loader,
            desc="Testing",
            unit="batch",
        ):
            images = list(image.to(self.device) for image in images)
            pred = self.model(images)

            for i, p in zip(idx, pred):
                digits = []
                for j in range(len(p["boxes"])):
                    x_min, y_min, x_max, y_max = p["boxes"][j].cpu().tolist()

                    pred_dict = {
                        "image_id": i + 1,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": p["scores"][j].item(),
                        "category_id": p["labels"][j].item(),
                    }
                    preds.append(pred_dict)

                    digits.append({"x_min": x_min, "val": p["labels"][j].item()})

                if len(digits) > 0:
                    digits.sort(key=lambda d: d["x_min"])  # sort by x_min
                    pred_val = int("".join(str(d["val"] - 1) for d in digits))
                else:
                    pred_val = -1

                pred_vals.append([i + 1, pred_val])

        # save predictions
        with open("pred.json", "w") as f:
            json.dump(preds, f, indent=4)

        df = pd.DataFrame(pred_vals, columns=["image_id", "pred_label"])
        df.to_csv("pred.csv", index=False)

        # zip the predictions
        os.system("zip solution.zip pred.json pred.csv")

    def save_weights(self, epoch, last=False):
        if last:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.opt.save_ckpt_dir, f"last.pth"),
            )
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.opt.save_ckpt_dir, f"best.pth"),
            )
            # torch.save(
            #     self.model.state_dict(),
            #     os.path.join(self.opt.save_ckpt_dir, f"epoch_{epoch}.pth"),
            # )

    def load_weights(self):
        ckpt = torch.load(self.opt.ckpt_name)
        self.model.load_state_dict(ckpt)
