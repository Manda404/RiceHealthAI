"""
Custom Convolutional Neural Network for RiceHealthAI.

This model serves as a baseline CNN architecture for classifying
rice leaf diseases. It is simple, modular, and easy to train
compared to pre-trained architectures (ResNet, VGG, EfficientNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from ricehealthai.core.utils import get_device  # auto device detection


class CustomCNN(nn.Module):
    """
    A simple Convolutional Neural Network for rice leaf disease classification.

    Architecture:
        - 3 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
        - Fully connected classifier with dropout
    """

    def __init__(self, num_classes: int = 4):
        """
        Initialize the CNN model.

        Parameters
        ----------
        num_classes : int, default=4
            Number of output classes (e.g., Bacterial Blight, Blast, Brown Spot, Tungro).
        """
        super(CustomCNN, self).__init__()

        # === Feature extractor === #
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
        )

        # === Classifier === #
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        print(f"Input : {x.shape}")
        x = self.features[0:4](x)   # 1st block
        print(f"After block 1 : {x.shape}")
        x = self.features[4:8](x)   # 2nd block
        print(f"After block 2 : {x.shape}")
        x = self.features[8:](x)    # 3rd block
        print(f"After block 3 : {x.shape}")
        x = torch.flatten(x, 1)
        print(f"After flatten : {x.shape}")
        x = self.classifier(x)
        print(f"Final output : {x.shape}")
        return x



    def _init_weights(self):
        """Kaiming initialization for Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


def build_custom_cnn(num_classes: int = 4) -> CustomCNN:
    """
    Factory function to create and move the CustomCNN model
    automatically to the best available device (CUDA, MPS, or CPU).

    Parameters
    ----------
    num_classes : int, default=4
        Number of output classes.

    Returns
    -------
    CustomCNN
        Instantiated and device-ready model.
    """
    from ricehealthai.core.utils import get_device
    device = get_device(verbose=True)
    model = CustomCNN(num_classes=num_classes).to(device)
    return model
