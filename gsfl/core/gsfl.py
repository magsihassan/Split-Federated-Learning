import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gsfl.models import ClientNet, ServerNet
from gsfl.sim import uplink_cost, downlink_cost, compute_cost
from gsfl.config import (
    DEVICE,
    BATCH_SIZE,
    LR_CLIENT,
    LR_SERVER,
    NUM_CLIENTS,
    NUM_GROUPS,
)


# Number of local batches per client per round
LOCAL_STEPS = 20


class GSFLTrainer:
    """
    Simplified, stable GSFL-style trainer:

    - ONE shared client model (front) across all clients
    - ONE server model PER GROUP
    - Split learning done per group (clients in same group share that group's server)
    - After each round, all server models are federated-averaged

    This keeps:
      - grouping
      - split learning
      - per-group server models
      - cross-group aggregation (federated averaging)
    while avoiding the instability of having 16 separate client nets.
    """

    def __init__(self, client_datasets, test_loader):
        assert len(client_datasets) == NUM_CLIENTS, "client_datasets must match NUM_CLIENTS"

        self.client_datasets = client_datasets
        self.test_loader = test_loader

        # One global client-side model
        self.client = ClientNet().to(DEVICE)

        # One server model per group
        self.servers = [ServerNet().to(DEVICE) for _ in range(NUM_GROUPS)]

        # Optimizers
        self.opt_client = optim.SGD(self.client.parameters(), lr=LR_CLIENT)
        self.opt_servers = [optim.SGD(s.parameters(), lr=LR_SERVER) for s in self.servers]

        self.criterion = nn.CrossEntropyLoss()

        # Fixed equal-size groups
        self.groups = self._make_groups()

    def _make_groups(self):
        group_size = NUM_CLIENTS // NUM_GROUPS
        groups = []
        for g in range(NUM_GROUPS):
            start = g * group_size
            end = (g + 1) * group_size
            groups.append(list(range(start, end)))
        return groups

    def train_round(self):
        """
        One GSFL round:
        - Each group trains on its clients with split learning
        - Each client contributes LOCAL_STEPS batches
        - Then server models are federated-averaged across groups
        """
        self.client.train()

        total_loss = 0.0
        total_uplink = 0.0
        total_downlink = 0.0
        total_compute = 0.0

        # For each group
        for g_idx, group in enumerate(self.groups):
            server = self.servers[g_idx]
            opt_server = self.opt_servers[g_idx]
            server.train()

            # For each client in that group
            for c_idx in group:
                ds = self.client_datasets[c_idx]
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

                # Multiple local steps per client
                for step, (images, labels) in enumerate(loader):
                    if step >= LOCAL_STEPS:
                        break

                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    # ----- Client forward -----
                    smashed = self.client(images)

                    # Uplink cost (client -> server)
                    total_uplink += uplink_cost(smashed)

                    # ----- Server forward -----
                    logits = server(smashed)

                    # Rough compute cost
                    flops = smashed.nelement() * 100
                    total_compute += compute_cost(flops)

                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                    # ----- Backward -----
                    self.opt_client.zero_grad()
                    opt_server.zero_grad()

                    loss.backward()

                    # Downlink cost (server -> client grads, same size as smashed)
                    total_downlink += downlink_cost(smashed)

                    # Update models
                    opt_server.step()
                    self.opt_client.step()

        # After all groups/clients trained this round:
        # federated average server models (group-level aggregation)
        self._federated_average_servers()

        return total_loss, total_uplink, total_downlink, total_compute

    def _federated_average_servers(self):
        """Federated averaging over server models only."""
        with torch.no_grad():
            for params in zip(*[s.parameters() for s in self.servers]):
                avg = sum(p.data for p in params) / len(params)
                for p in params:
                    p.data.copy_(avg)

    def evaluate(self):
        """
        Evaluate using shared client and (averaged) server model.
        After federated averaging, all servers are identical, so we use servers[0].
        """
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
