import os
import time
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.options import get_args
from utils.dataset import CustomTrainDataset
from utils.model_utils import PromptIRModel


def main():
    opt = get_args()

    if opt.test_only:
        raise ValueError("Testing is not supported in this script.")

    logger = TensorBoardLogger(
        save_dir=(
            f"logs/{time.strftime('%Y-%m-%dT%H:%M:%S')}"
            if opt.log_dir is None
            else f"logs/{opt.log_dir}"
        )
    )

    train_dataset = CustomTrainDataset(root_dir=opt.train_data_dir, mode="train")
    # val_dataset = CustomTrainDataset(root_dir=opt.train_data_dir, mode="val")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.save_ckpt_dir, every_n_epochs=1, save_top_k=-1
    )

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=1,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed" if opt.fp16 else None,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
