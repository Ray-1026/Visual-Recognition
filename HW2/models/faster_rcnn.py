import torch
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FastRCNNConvFCHead,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from models.backbone import ResNest50_FPN, ResNet50_FPN, CustomFasterRCNNPredictor


def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class CustomFasterRCNN(nn.Module):
    def __init__(self, backbone="default", box_score_threshold=0.5, num_classes=11):
        super(CustomFasterRCNN, self).__init__()

        if backbone == "default":  # ResNet50-FPN (v2)
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                box_score_thresh=box_score_threshold,  # Set the threshold for the box scores
                min_size=400,
                max_size=1000,
            )

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
        # elif backbone == "resnest":  # ResNest50-FPN ==> discard
        #     backbone = ResNest50_FPN()
        #     rpn_anchor_generator = _default_anchorgen()
        #     rpn_head = RPNHead(
        #         backbone.out_channels,
        #         rpn_anchor_generator.num_anchors_per_location()[0],
        #         conv_depth=2,
        #     )
        #     box_head = FastRCNNConvFCHead(
        #         (backbone.out_channels, 7, 7),
        #         [256, 256, 256, 256],
        #         [1024],
        #         norm_layer=nn.BatchNorm2d,
        #     )

        #     self.model = FasterRCNN(
        #         backbone,
        #         num_classes=num_classes,
        #         rpn_anchor_generator=rpn_anchor_generator,
        #         rpn_head=rpn_head,
        #         box_head=box_head,
        #         box_score_thresh=box_score_threshold,
        #     )
        elif backbone == "2layerpredictor":
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                box_score_thresh=box_score_threshold,  # Set the threshold for the box scores
                min_size=400,
                max_size=1000,
            )

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = CustomFasterRCNNPredictor(
                in_features, num_classes
            )
        # elif backbone == "resnet50_v2":  # ResNet34-FPN ==> discard
        #     backbone = ResNet50_FPN()
        #     rpn_anchor_generator = _default_anchorgen()
        #     rpn_head = RPNHead(
        #         backbone.out_channels,
        #         rpn_anchor_generator.num_anchors_per_location()[0],
        #         conv_depth=2,
        #     )
        #     box_head = FastRCNNConvFCHead(
        #         (backbone.out_channels, 7, 7),
        #         [256, 256, 256, 256],
        #         [1024],
        #         norm_layer=nn.BatchNorm2d,
        #     )

        #     self.model = FasterRCNN(
        #         backbone,
        #         num_classes=num_classes,
        #         rpn_anchor_generator=rpn_anchor_generator,
        #         rpn_head=rpn_head,
        #         box_head=box_head,
        #         box_score_thresh=box_score_threshold,
        #     )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # # fixed backbone
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)


if __name__ == "__main__":
    # model = CustomFasterRCNN_ResNest50()
    pass
