import torch
import torch.nn.functional as F
from torch import nn
import timm


class SENeXt50(nn.Module):
    def __init__(self, num_classes=100):
        super(SENeXt50, self).__init__()
        self.model = timm.create_model("seresnext50_32x4d", pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
