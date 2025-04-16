import torch
from torch import nn
from torchvision.models import resnet34, resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN,
    _resnet_fpn_extractor,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CustomFasterRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomFasterRCNNPredictor, self).__init__()
        self.cls_score = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * 4),
        )

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def ResNest50_FPN():
    """
    Load ResNest50 backbone with FPN.
    """
    # Load ResNest50 backbone
    torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
    model = torch.hub.load("zhanghang1989/ResNeSt", "resnest50", pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])

    backbone = BackboneWithFPN(
        model,
        return_layers={"4": "0", "5": "1", "6": "2", "7": "3"},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        norm_layer=nn.BatchNorm2d,
    )

    return backbone


def ResNet50_FPN():
    """
    Load ResNet34 backbone with FPN.
    """
    # Load ResNet34 backbone
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone = _resnet_fpn_extractor(model, 5, norm_layer=nn.BatchNorm2d)

    return backbone


if __name__ == "__main__":
    pass
