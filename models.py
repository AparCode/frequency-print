import torch.nn as nn
from torchvision import models

class CannotParseModelName(ValueError):
    def __init__(self, str_name:str):
        super().__init__(f"Cannot parse model name: {str_name}")

# Defining the Simple CNN architecture with 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, pretrained:bool=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Assuming 10 classes for classification
    
    def forward(self, x):
        return self.model(x)


class ResNet34(nn.Module):
    def __init__(self, pretrained:bool=False):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Assuming 10 classes for classification
    
    def forward(self, x):
        return self.model(x)
