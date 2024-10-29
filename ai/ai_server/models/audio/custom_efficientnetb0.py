import torch
import torch.nn as nn
from torchvision import models

class CustomEfficientNetB0(nn.Module):
    def __init__(self):
        super(CustomEfficientNetB0, self).__init__()
        self.backbone = models.efficientnet_b0()

        # 첫 번째 Conv2D 레이어의 가중치와 bias를 얻기
        first_conv_layer = self.backbone.features[0][0]

        # 새로운 Conv2D 레이어 생성 (in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        new_first_conv_layer = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=first_conv_layer.out_channels, 
            kernel_size=first_conv_layer.kernel_size, 
            stride=first_conv_layer.stride, 
            padding=first_conv_layer.padding, 
            bias=first_conv_layer.bias is not None
        )

        # EfficientNet 모델의 첫 번째 Conv2D 레이어를 새로운 레이어로 교체
        self.backbone.features[0][0] = new_first_conv_layer
        num_features = self.backbone.classifier[1].in_features  # 기존 출력층의 입력 feature 수
        self.backbone.classifier[1] = torch.nn.Linear(num_features, 2)  # 2개의 클래스로 변경

        
    def forward(self, x):
        x = self.backbone(x)
        return x