import torch
import torch.nn as nn

class Conv3dL(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv3dL, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(1, 8, (1, 3, 3)),  # Change input channels from 3 to 1
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d((1, 2, 2)),  # Use smaller kernel size to prevent excessive reduction
            nn.Conv3d(8, 32, (1, 3, 3)),  # Adjust kernel size
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d((1, 2, 2)),  # Use smaller kernel size to prevent excessive reduction
            nn.Conv3d(32, 64, (1, 3, 3)),  # Adjust kernel size
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d((1, 2, 2)),  # Use smaller kernel size to prevent excessive reduction
            nn.Conv3d(64, 128, (1, 3, 3)),  # Adjust kernel size
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, (1, 3, 3)),  # Adjust kernel size
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Use adaptive pooling to get a fixed output size
        )
        self.classifier = nn.Linear(256, num_classes)  # Adjust input size to match feature map output

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
