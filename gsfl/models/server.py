import torch.nn as nn

class ServerNet(nn.Module):
    """
    Back part of CNN: classifier on top of 64x7x7 features.
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(x)
