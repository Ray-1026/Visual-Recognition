import random
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def set_seed(myseed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)


def crop_img(image, base=16):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[
        crop_h // 2 : h - crop_h + crop_h // 2,
        crop_w // 2 : w - crop_w + crop_w // 2,
        :,
    ]


def compute_psnr(pred, gt):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between predicted and ground truth images.

    Args:
        pred (np.ndarray): Predicted image.
        gt (np.ndarray): Ground truth image.

    Returns:
        float: PSNR value.
    """
    if pred.shape != gt.shape:
        raise ValueError("Predicted and ground truth images must have the same shape.")

    pred_np = np.clip(pred.detach().cpu().numpy(), 0, 1)
    gt_np = np.clip(gt.detach().cpu().numpy(), 0, 1)

    pred_np = pred_np.transpose(0, 2, 3, 1)
    gt_np = gt_np.transpose(0, 2, 3, 1)

    psnr_list = []
    for i in range(pred_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(gt_np[i], pred_np[i], data_range=1.0)
        psnr_list.append(psnr_value)

    return psnr_list
