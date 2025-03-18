import os
import time
import argparse
import numpy as np
import torch

from utils.trainer import Trainer

myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model", type=str, default="resnext101")

    # hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=5)

    # data directories
    parser.add_argument("--train_data_dir", type=str, default="data/train")
    parser.add_argument("--valid_data_dir", type=str, default="data/val")
    parser.add_argument("--test_data_dir", type=str, default="data/test")

    # logging and checkpointing
    parser.add_argument("--save_ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--log_dir", type=str, default=None)

    # evaluation options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Only run evaluation, no training",
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="Name of the checkpoint to load for evaluation"
    )

    args = parser.parse_args()

    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def main(args):
    trainer = Trainer(args)

    if not args.eval_only:
        trainer.train()
    else:
        trainer.eval()


if __name__ == "__main__":
    args = get_args()
    main(args)
