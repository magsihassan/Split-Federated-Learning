import torch.nn as nn

class ClientNet(nn.Module):
    """
    Front part of CNN: conv layers up to the cut layer.
    Input: 1x28x28
    Output: feature map (64x7x7)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x14x14

            nn.Conv2d(32, 64, 3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                  # 64x7x7
        )

    def forward(self, x):
        return self.features(x)
