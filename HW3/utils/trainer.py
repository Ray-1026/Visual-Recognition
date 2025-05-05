import os
import time
import json
import tempfile
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from models.mask_rcnn import CustomMaskRCNN
from utils.dataset import get_transforms, collate_fn, CustomDataset
from utils.mask_utils import encode_mask


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device

        self.model = CustomMaskRCNN(
            num_classes=self.opt.num_classes,
            box_score_thresh=self.opt.box_score_thresh,
            backbone=self.opt.backbone,
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
            if self.opt.use_validation:
                dataset = CustomDataset(
                    root_dir=self.opt.train_data_dir,
                )
                val_size = int(len(dataset) * 0.1)
                train_size = len(dataset) - val_size
                self.train_dataset, self.valid_dataset = random_split(
                    dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )
                self.train_dataset.dataset.transform = self.train_tfm
                self.valid_dataset.dataset.transform = self.test_tfm
            else:
                self.train_dataset = CustomDataset(
                    root_dir=self.opt.train_data_dir, transform=self.train_tfm
                )

            # dataloader
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=self.opt.num_workers,
                collate_fn=collate_fn,
            )

            if self.opt.use_validation:
                self.val_loader = DataLoader(
                    self.valid_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.opt.num_workers,
                    collate_fn=collate_fn,
                )
            else:
                self.val_loader = None

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

        else:  # testing only
            self.test_dataset = CustomDataset(
                root_dir=self.opt.test_data_dir,
                transform=self.test_tfm,
                is_test=True,
                test_json=self.opt.json_map_imgname_to_id,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.opt.num_workers,
                collate_fn=collate_fn,
            )

        # validation mAP record
        self.best_mAP = 0.0

    def get_optimizer(self):
        """Get the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.opt.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.opt.lr,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.opt.lr,
                momentum=0.9,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.opt.lr,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "nesterov":
            return torch.optim.SGD(
                params,
                lr=self.opt.lr,
                momentum=0.9,
                weight_decay=self.opt.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt.optimizer}")

    def train(self, val_every_epochs=1):
        self.scaler = GradScaler() if self.opt.fp16 else None

        for epoch in range(1, self.opt.epochs + 1):
            self.model.train()
            train_losses = {
                "cls_loss": 0.0,
                "box_loss": 0.0,
                "obj_loss": 0.0,
                "rpn_box_loss": 0.0,
                "mask_loss": 0.0,
                "total_loss": 0.0,
            }
            total_batches = len(self.train_loader)

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
                train_losses["cls_loss"] += loss_dict["loss_classifier"].item()
                train_losses["box_loss"] += loss_dict["loss_box_reg"].item()
                train_losses["obj_loss"] += loss_dict["loss_objectness"].item()
                train_losses["rpn_box_loss"] += loss_dict["loss_rpn_box_reg"].item()
                train_losses["mask_loss"] += loss_dict["loss_mask"].item()
                train_losses["total_loss"] += losses.item()

            if self.scheduler is not None:
                self.scheduler.step()

            torch.cuda.empty_cache()  # clear cache

            if epoch % val_every_epochs == 0:
                mAP = 0.0
                if self.val_loader is not None:
                    mAP = self.validate(epoch)

                # save best model
                if mAP > self.best_mAP:
                    self.best_mAP = mAP
                    self.save_weights(epoch)

                # print log
                print(
                    f"Epoch [{epoch:>3}/{self.opt.epochs:<3}] | "
                    f"Loss: {train_losses['total_loss'] / total_batches:.4f} | "
                    f"mAP: {mAP:.4f} | "
                    f"Best mAP: {self.best_mAP:.4f}"
                )

            # write to tensorboard
            self.writer.add_scalar(
                "Loss/cls_loss", train_losses["cls_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Loss/box_loss", train_losses["box_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Loss/obj_loss", train_losses["obj_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Loss/rpn_box_loss", train_losses["rpn_box_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Loss/mask_loss", train_losses["mask_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Loss/total_loss", train_losses["total_loss"] / total_batches, epoch
            )
            self.writer.add_scalar(
                "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
            )
            self.writer.add_scalar("mAP/val", mAP, epoch)

            torch.cuda.empty_cache()  # clear cache

        # close tensorboard writer
        self.writer.close()

        # save last ckpt
        self.save_weights(epoch, last=True)
        print("Training complete")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        preds = []
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)],
        }
        ann_id = 1

        for images, targets in tqdm(
            self.val_loader,
            desc=f"Validation {epoch}/{self.opt.epochs}",
            unit="batch",
            leave=False,
        ):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            image_ids = list(t["image_id"].item() for t in targets)

            pred = self.model(images)

            for i, (iid, p) in enumerate(zip(image_ids, pred)):
                coco_gt["images"].append(
                    {
                        "id": iid,
                        "width": images[i].shape[2],
                        "height": images[i].shape[1],
                    }
                )
                gt_masks = targets[i]["masks"].numpy()
                gt_labels = targets[i]["labels"].numpy()
                for j in range(len(gt_masks)):
                    encoded_mask = encode_mask(gt_masks[j])
                    coco_gt["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": iid,
                            "bbox": list(maskUtils.toBbox(encoded_mask)),
                            "category_id": int(gt_labels[j]),
                            "segmentation": encoded_mask,
                            "iscrowd": 0,
                            "area": int(gt_masks[j].sum()),
                        }
                    )
                    ann_id += 1

                for j in range(len(p["masks"])):
                    mask = p["masks"][j].cpu().numpy()[0]
                    binary_mask = (mask >= 0.5).astype(np.uint8)

                    pred_dict = {
                        "image_id": iid,
                        "score": p["scores"][j].item(),
                        "category_id": p["labels"][j].item(),
                        "segmentation": encode_mask(binary_mask),
                    }

                    preds.append(pred_dict)

        if len(preds) == 0:
            return 0.0

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json"
        ) as pred_f, tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as gt_f:
            json.dump(preds, pred_f)
            json.dump(coco_gt, gt_f)

            pred_f.flush()
            gt_f.flush()

            coco_gt = COCO(gt_f.name)
            coco_dt = coco_gt.loadRes(pred_f.name)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
            coco_eval.params.iouThrs = np.array([0.5], dtype=float)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            mAP = coco_eval.stats[0]
            return mAP

    @torch.no_grad()
    def test(self, load_weights=True):
        self.model.eval()

        if load_weights:
            self.load_weights()

        preds = []
        for images, idx in tqdm(
            self.test_loader,
            desc=f"Testing",
            unit="batch",
            leave=False,
        ):
            images = list(image.to(self.device) for image in images)
            pred = self.model(images)

            for i, p in zip(idx, pred):
                for j in range(len(p["masks"])):
                    x_min, y_min, x_max, y_max = p["boxes"][j].cpu().tolist()
                    mask = p["masks"][j].cpu().numpy()[0]
                    H, W = mask.shape
                    binary_mask = (mask >= 0.5).astype(np.uint8)

                    pred_dict = {
                        "image_id": i,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": p["scores"][j].item(),
                        "category_id": p["labels"][j].item(),
                        "segmentation": encode_mask(binary_mask),
                    }

                    preds.append(pred_dict)

        with open("test-results.json", "w") as f:
            json.dump(preds, f, indent=4)

        if self.best_mAP > 0:
            os.system(f"zip solution_{self.best_mAP:.4f}.zip test-results.json")
        else:
            os.system("zip solution.zip test-results.json")

    def save_weights(self, epoch, last=False):
        ckpt_path = os.path.join(self.opt.save_ckpt_dir, f"best.pth")
        if last:
            ckpt_path = os.path.join(self.opt.save_ckpt_dir, f"last.pth")

        torch.save(
            self.model.state_dict(),
            ckpt_path,
        )

    def load_weights(self):
        ckpt = torch.load(self.opt.ckpt_name)
        self.model.load_state_dict(ckpt)
