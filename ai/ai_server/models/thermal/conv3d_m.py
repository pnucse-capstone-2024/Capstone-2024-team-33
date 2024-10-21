import torch
import torch.nn as nn

class Conv3dM(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv3dM, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(1, 8, (1, 3, 3)), 
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d((1, 2, 2)),  
            nn.Conv3d(8, 32, (1, 3, 3)),  
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d((1, 2, 2)),  
            nn.Conv3d(32, 64, (1, 3, 3)),  
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d((1, 2, 2)),  
            nn.Conv3d(64, 128, (1, 3, 3)),  
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  
        )
        self.classifier = nn.Linear(128, num_classes)  

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x