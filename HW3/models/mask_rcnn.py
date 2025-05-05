import torch
import torchvision
from torch import nn
from torchvision.models.detection import (
    MaskRCNN,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.backbone import convnext_fpn_backbone, efficientnetv2_fpn_backbone


class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes=5, box_score_thresh=0.5, backbone="default"):
        super(CustomMaskRCNN, self).__init__()

        if backbone == "default":
            self.model = maskrcnn_resnet50_fpn_v2(
                weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                box_score_thresh=box_score_thresh,
            )
        elif backbone == "convnext" or backbone == "c":
            print("Using ConvNeXt backbone")
            self.model = MaskRCNN(
                backbone=convnext_fpn_backbone(),
                num_classes=num_classes,
                box_score_thresh=box_score_thresh,
            )
        elif backbone == "efficientnetv2" or backbone == "e":
            print("Using EfficientNetV2 backbone")
            self.model = MaskRCNN(
                backbone=efficientnetv2_fpn_backbone(),
                num_classes=num_classes,
                box_score_thresh=box_score_thresh,
            )

        # replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        # check model parameters
        self.log_model_parameters()

    def log_model_parameters(self):
        model_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        # >200M => error
        assert model_params < 200, "Model parameters exceed 200M"

        print(f"Model parameters: {model_params:.2f}M")

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        return self.model(images, targets)


if __name__ == "__main__":
    # Example usage
    model = CustomMaskRCNN()
    model.eval()

    x = [torch.rand(3, 300, 400)]
    predictions = model(x)
    print(predictions[0].keys())
    print(predictions[0]["masks"].shape)

    # print unique values in the labels
    labels = predictions[0]["labels"]
    unique_labels = torch.unique(labels)
    print(f"Unique labels: {unique_labels}")
