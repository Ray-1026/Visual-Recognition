import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure


class L1_SSIM_loss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.84):
        super(L1_SSIM_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, x, y):
        l1_loss = self.l1_loss(x, y)
        ssim_loss = 1 - self.ssim(x, y)
        return (
            self.l1_weight * l1_loss + self.ssim_weight * ssim_loss,
            l1_loss,
            ssim_loss,
        )


class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        return torch.mean(error)


class Gradient_L1_loss(nn.Module):
    def __init__(self):
        super(Gradient_L1_loss, self).__init__()

    def _gradient(self, img):
        """Compute forward finite differences along x and y.
        img: [B, C, H, W] in any value range (will be used as-is)
        returns 2-tuple (dx, dy) with same dtype
        """
        dx = img[..., :, :-1] - img[..., :, 1:]  # H × (W-1)
        dy = img[..., :-1, :] - img[..., 1:, :]  # (H-1) × W
        return dx, dy

    def forward(self, pred, gt):
        dx_p, dy_p = self._gradient(pred)
        dx_g, dy_g = self._gradient(gt)
        loss = torch.mean(torch.abs(dx_p - dx_g)) + torch.mean(torch.abs(dy_p - dy_g))
        return loss
