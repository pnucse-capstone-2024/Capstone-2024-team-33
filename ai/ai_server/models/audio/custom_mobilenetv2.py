import torch
import torch.nn as nn
from torchvision import models

class CustomMobileNetV2(nn.Module):
    def __init__(self):
        super(CustomMobileNetV2, self).__init__()
        mobilenetv2 = models.mobilenet_v2()
        mobilenetv2.features[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        mobilenetv2.classifier[1] = nn.Linear(in_features=1280, out_features=4)

        self.backbone = mobilenetv2
        
    def forward(self, x):
        x = self.backbone(x)
        return x