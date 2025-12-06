import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gsfl.models import ClientNet, ServerNet
from gsfl.sim import uplink_cost, downlink_cost, compute_cost
from gsfl.config import DEVICE, BATCH_SIZE, LR_CLIENT, LR_SERVER


class SplitLearningTrainer:
    """
    Baseline: sequential Split Learning (SL)
    One shared client model and one shared server model.
    Clients are just data partitions.
    """
    def __init__(self, client_datasets, test_loader):
        self.client_datasets = client_datasets
        self.test_loader = test_loader

        self.client = ClientNet().to(DEVICE)
        self.server = ServerNet().to(DEVICE)

        self.opt_client = optim.SGD(self.client.parameters(), lr=LR_CLIENT)
        self.opt_server = optim.SGD(self.server.parameters(), lr=LR_SERVER)

        self.criterion = nn.CrossEntropyLoss()

    def train_round(self):
        """
        One SL round:
        Iterate over all clients once (one batch per client).
        """
        self.client.train()
        self.server.train()

        total_loss = 0.0
        total_uplink = 0.0
        total_downlink = 0.0
        total_compute = 0.0

        for ds in self.client_datasets:
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            images, labels = next(iter(loader))
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Client forward
            smashed = self.client(images)

            # Simulated uplink cost
            total_uplink += uplink_cost(smashed)

            # Server forward
            logits = self.server(smashed)

            # Very rough FLOP estimate: proportional to smashed size
            flops = smashed.nelement() * 100
            total_compute += compute_cost(flops)

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            # Backward
            self.opt_client.zero_grad()
            self.opt_server.zero_grad()

            loss.backward()  # gradients flow into both parts

            # Simulated downlink (server -> client gradients)
            total_downlink += downlink_cost(smashed)

            self.opt_server.step()
            self.opt_client.step()

        return total_loss, total_uplink, total_downlink, total_compute

    def evaluate(self):
        self.client.eval()
        self.server.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                smashed = self.client(images)
                logits = self.server(smashed)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
