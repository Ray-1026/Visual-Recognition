import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure

from model.promptir import PromptIR
from utils.losses import L1_SSIM_loss
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.utils import compute_psnr


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

        self.loss_fn = L1_SSIM_loss()

        self.psnr_list = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss, l1_loss, s_loss = self.loss_fn(restored, clean_patch)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_l1_loss", l1_loss)
        self.log("train_ssim_loss", s_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step defines the validation loop."""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)

        # compute psnr
        psnr = compute_psnr(restored, clean_patch)
        self.psnr_list.extend(psnr)

        return np.mean(psnr)

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        avg_psnr = np.mean(self.psnr_list)
        self.log("val_psnr", avg_psnr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.psnr_list.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=15, max_epochs=150
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
