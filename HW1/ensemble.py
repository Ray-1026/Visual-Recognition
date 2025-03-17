import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.resnest import ResNest50, ResNest101
from models.senet import SENeXt50
from utils.dataset import TestingDataset, get_transforms

myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


class EnsembleModel(nn.Module):
    def __init__(self, model_names, ckpts, device):
        super(EnsembleModel, self).__init__()
        self.models = []
        for model_name, ckpt in zip(model_names, ckpts):
            model = model_name().to(device)
            model = model.eval()
            model.load_state_dict(torch.load(ckpt)["model"])
            self.models.append(model)

    def count_parameters(self):
        self.total_params = 0
        for model in self.models:
            self.total_params += sum(p.numel() for p in model.parameters())

        return self.total_params

    def forward(self, x):
        # softmax
        outputs = None
        for model in self.models:
            if outputs is None:
                outputs = F.softmax(model(x), dim=1)
            else:
                outputs += F.softmax(model(x), dim=1)
        return outputs


def inference(test_loader, model, device):
    # print total number of parameters
    print(f"Total params: {model.count_parameters() / 1e6:.2f}M")

    preds = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())

    submission = pd.DataFrame(
        {"image_name": test_loader.dataset.names, "pred_label": preds}
    )
    submission.to_csv("prediction.csv", index=False)

    # zip
    os.system("zip ensemble.zip prediction.csv")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_tfm = get_transforms()
    test_dataset = TestingDataset("data/test", transform=test_tfm)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=8)

    models = [ResNest101, ResNest50, SENeXt50]
    ckpts = [
        "ckpt/resnest101.pth",
        "ckpt/resnest50.pth",
        "ckpt/senext50.pth",
    ]

    model = EnsembleModel(models, ckpts, device)
    inference(test_loader, model, device)
