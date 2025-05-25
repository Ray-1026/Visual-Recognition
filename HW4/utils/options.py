import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)

    # training options
    # parser.add_argument(
    #     "--optimizer",
    #     type=str,
    #     default="adamw",
    #     choices=["adam", "sgd", "adamw", "nesterov"],
    #     help="Optimizer to use for training",
    # )
    # parser.add_argument(
    #     "--scheduler",
    #     type=str,
    #     default="none",
    #     choices=["none", "cosine"],
    #     help="Scheduler to use for training",
    # )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Use mixed precision training (fp16)",
    )

    # dataloader options
    parser.add_argument("--num_workers", type=int, default=8)

    # dataset options
    parser.add_argument("--train_data_dir", type=str, default="hw4_dataset/train")
    parser.add_argument(
        "--test_data_dir", type=str, default="hw4_dataset/test/degraded"
    )
    parser.add_argument("--use_validation", action="store_true", default=False)

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
        "--ckpt_path", type=str, help="Path of the checkpoint to load for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save test results",
    )

    args = parser.parse_args()

    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
