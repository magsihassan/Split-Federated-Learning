import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gsfl.models import ClientNet, ServerNet
from gsfl.sim.comm import comm_delay
from gsfl.sim.compute import compute_cost
from gsfl.config import (
    DEVICE,
    BATCH_SIZE,
    LR_CLIENT,
    LR_SERVER,
    NUM_CLIENTS,
    NUM_GROUPS,
)

LOCAL_STEPS = 20  # recommended for strong accuracy


class GSFLTrainer:
    """
    GSFL Trainer using:
    - One global client model
    - One server model per group
    - Wireless bandwidth simulation per client
    """

    def __init__(self, client_datasets, test_loader):
        assert len(client_datasets) == NUM_CLIENTS

        self.client_datasets = client_datasets
        self.test_loader = test_loader

        # Global client model
        self.client = ClientNet().to(DEVICE)

        # Group server models
        self.servers = [ServerNet().to(DEVICE) for _ in range(NUM_GROUPS)]

        # Optimizers
        self.opt_client = optim.SGD(self.client.parameters(), lr=LR_CLIENT)
        self.opt_servers = [optim.SGD(s.parameters(), lr=LR_SERVER) for s in self.servers]

        self.criterion = nn.CrossEntropyLoss()

        # Equal size groups
        self.groups = self._make_groups()

    def _make_groups(self):
        size = NUM_CLIENTS // NUM_GROUPS
        return [list(range(g*size, (g+1)*size)) for g in range(NUM_GROUPS)]

    def train_round(self):
        total_loss = 0
        total_up = 0
        total_down = 0
        total_compute = 0

        self.client.train()

        for g_idx, group in enumerate(self.groups):
            server = self.servers[g_idx]
            opt_server = self.opt_servers[g_idx]

            server.train()

            for client_id in group:
                ds = self.client_datasets[client_id]
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

                for step, (images, labels) in enumerate(loader):
                    if step >= LOCAL_STEPS:
                        break

                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    # Client forward
                    smashed = self.client(images)

                    total_up += comm_delay(smashed, client_id)

                    # Server forward
                    logits = server(smashed)

                    flops = smashed.nelement() * 100
                    total_compute += compute_cost(flops)

                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                    # Backprop
                    self.opt_client.zero_grad()
                    opt_server.zero_grad()
                    loss.backward()

                    total_down += comm_delay(smashed, client_id)

                    self.opt_client.step()
                    opt_server.step()

        self._aggregate_servers()

        return total_loss, total_up, total_down, total_compute

    def _aggregate_servers(self):
        """Federated average server models."""
        with torch.no_grad():
            for params in zip(*[s.parameters() for s in self.servers]):
                avg = sum(p.data for p in params) / len(params)
                for p in params:
                    p.data.copy_(avg)

    def evaluate(self):
        self.client.eval()
        server = self.servers[0]
        server.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                smashed = self.client(images)
                logits = server(smashed)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
