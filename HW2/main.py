import os
import time
import argparse
import numpy as np
import torch

from utils.trainer import Trainer

myseed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


def get_args():
    parser = argparse.ArgumentParser()

    # model options
    parser.add_argument("--backbone", type=str, default="default")
    parser.add_argument("--num_classes", type=int, default=11)

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # training options
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw", "nesterov"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="Scheduler to use for training",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Use mixed precision training (fp16)",
    )

    # dataloader options
    parser.add_argument("--num_workers", type=int, default=8)

    # dataset options
    parser.add_argument("--train_data_dir", type=str, default="nycu-hw2-data/train")
    parser.add_argument("--valid_data_dir", type=str, default="nycu-hw2-data/valid")
    parser.add_argument("--test_data_dir", type=str, default="nycu-hw2-data/test")
    parser.add_argument(
        "--annotations_file_train",
        type=str,
        default="nycu-hw2-data/train.json",
        help="Path to the annotations file for training data",
    )
    parser.add_argument(
        "--annotations_file_valid",
        type=str,
        default="nycu-hw2-data/valid.json",
        help="Path to the annotations file for validation data",
    )

    # logging and checkpointing
    parser.add_argument("--save_ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--log_dir", type=str, default=None)

    # evaluation options
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only run testing, no training",
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="Name of the checkpoint to load for evaluation"
    )
    parser.add_argument(
        "--box_score_thresh",
        type=float,
        default=0.5,
        help="Threshold for test evaluation",
    )

    args = parser.parse_args()

    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def main():
    args = get_args()

    trainer = Trainer(args)

    if not args.test_only:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
