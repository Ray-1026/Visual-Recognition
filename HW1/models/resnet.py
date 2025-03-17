from torch import nn
from torchvision.models import (
    resnet50,
    resnet101,
    resnet152,
    resnext101_64x4d,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
)


class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet101, self).__init__()
        self.model = resnet101(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet152(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet152, self).__init__()
        self.model = resnet152(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNeXt50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNeXt50, self).__init__()
        self.model = resnext50_32x4d(
            weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNeXt101, self).__init__()
        self.model = resnext101_64x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
