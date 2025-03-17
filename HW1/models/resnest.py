import torch
from torch import nn
from models.cbam import CBAM


class ResNest50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNest50, self).__init__()

        # get list of models
        torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)

        self.model = torch.hub.load(
            "zhanghang1989/ResNeSt", "resnest50", pretrained=True
        )
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNest101(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNest101, self).__init__()

        # get list of models
        torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)

        self.model = torch.hub.load(
            "zhanghang1989/ResNeSt", "resnest101", pretrained=True
        )
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
