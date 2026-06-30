import torch
import torch.nn as nn
import torchvision.models as models
from torchview import draw_graph 


class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        # cater Resnet18 to binary classification on 28x28 images:

        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # reduce input channels to 1
        # self.model.conv1 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False
        # )

        # remove maxpool as images are already small
        self.model.maxpool = nn.Identity()

        # reduce output channels to 1 (binary classification)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ResNet18Binary()
    
    draw_graph(model, to="resnet18_binary_28x28.png")