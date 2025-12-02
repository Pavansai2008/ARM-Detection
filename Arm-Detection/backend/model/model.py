"""
Model definitions for ARM Detection System
Contains ARMClassifier and UltrasoundClassifier classes
"""

import torch
import torch.nn as nn
import torchvision.models as models


class UltrasoundClassifier(nn.Module):
    """
    ResNet18-based classifier for detecting ultrasound images.
    Binary classification: ultrasound vs non-ultrasound
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(UltrasoundClassifier, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ARMClassifier(nn.Module):
    """
    ResNet18-based classifier for ARM detection in ultrasound images.
    Binary classification: ARM vs Normal
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(ARMClassifier, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

