"""
ConvNeXt backbone for Mask R-CNN
    ref: https://github.com/mberkay0/ConvNeXt-MaskRCNN/blob/main/ConvNeXt%20MaskRCNN/utils.py
"""

import torch
import torchvision
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models.feature_extraction.create_feature_extractor
    to extract a submodel that returns the feature maps specified in given backbone
    feature extractor model.
    Parameters
    ----------
    backbone (nn.Module): Feature extractor ConvNeXt pretrained model.
    in_channels_list (List[int]): Number of channels for each feature map
        that is returned, in the order they are present in the OrderedDict
    out_channels (int): number of channels in the FPN.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use.
    extra_blocks (callable, optional): Default None.
        Extra optional FPN blocks.
    Attributes
    ----------
    out_channels : int
        The number of channels in the FPN.
    """

    def __init__(
        self,
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def convnext_fpn_backbone(
    backbone_name="convnext_base",
    trainable_layers=5,
    extra_blocks=None,
    norm_layer=None,
    feature_dict={"1": "0", "3": "1", "5": "2", "7": "3"},
    out_channels=256,
):
    """
    Returns an FPN-extended backbone network using a feature extractor
    based on models developed in the article 'A ConvNet for the 2020s'.
    For detailed information about the feature extractor ConvNeXt, read the article.
    https://arxiv.org/abs/2201.03545
    Parameters
    ----------
    backbone_name : str
        ConvNeXt architecture. Possible values are 'convnext_tiny', 'convnext_small',
        'convnext_base' or 'convnext_large'.
    trainable_layers : int
        Number of trainable (not frozen) layers starting from final block.
        Valid values are between 0 and 8, with 8 meaning all backbone layers
        are trainable.
    extra_blocks (ExtraFPNBlock or None): default a ``LastLevelMaxPool`` is used.
        If provided, extra operations will be performed. It is expected to take
        the fpn features, the original features and the names of the original
        features as input, and returns a new list of feature maps and their
        corresponding names.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use. It is recommended to use
        the default value. For details visit:
        (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
    feature_dict : dictionary
        Contains the names of the 'nn.Sequential' object used in the ConvNeXt
        model configuration if you need more detailed information,
        https://github.com/facebookresearch/ConvNeXt.
    out_channels (int): defaults to 256.
        Number of channels in the FPN.
    Returns
    -------
    BackboneWithFPN : torch.nn.Module
        Returns a specified ConvNeXt backbone with FPN on top.
        Freezes the specified number of layers in the backbone.
    """
    # if backbone_name == "convnext_tiny":
    #     backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
    #     backbone = create_feature_extractor(backbone, feature_dict)
    # elif backbone_name == "convnext_small":
    #     backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
    #     backbone = create_feature_extractor(backbone, feature_dict)
    # elif backbone_name == "convnext_base":

    backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
    backbone = create_feature_extractor(backbone, feature_dict)
    in_channels_list = [128, 256, 512, 1024]

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(
            f"Trainable layers should be in the range [0,8], got {trainable_layers}"
        )
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1"][:trainable_layers]
    if trainable_layers == 8:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )


def efficientnetv2_fpn_backbone(
    trainable_layers=5,
    extra_blocks=None,
    norm_layer=None,
    feature_dict={"1": "0", "3": "1", "5": "2", "7": "3"},
    out_channels=256,
):
    backbone = torchvision.models.efficientnet_v2_m(weights="DEFAULT").features
    backbone = create_feature_extractor(backbone, feature_dict)
    in_channels_list = [24, 80, 176, 512]

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(
            f"Trainable layers should be in the range [0,8], got {trainable_layers}"
        )
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1"][:trainable_layers]
    if trainable_layers == 8:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    return BackboneWithFPN(
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )
