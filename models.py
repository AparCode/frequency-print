"""Model architectures used by the project."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for real-vs-fake audio spectrograms."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet18(nn.Module):
    """ResNet18 wrapper with a configurable output head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class ResNet34(nn.Module):
    """ResNet34 wrapper with a configurable output head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.model = models.resnet34(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
