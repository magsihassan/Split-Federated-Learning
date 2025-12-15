import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gsfl.models import ClientNet, ServerNet
from gsfl.sim.comm import comm_delay
from gsfl.sim.compute import compute_cost
from gsfl.config import DEVICE, BATCH_SIZE, LR_CLIENT, LR_SERVER

# Same number of local steps as GSFL for FAIR comparison
LOCAL_STEPS = 4  # Must match GSFL's LOCAL_STEPS


class SplitLearningTrainer:
    """
    Sequential Split Learning Trainer.
    ALL clients train one after another (NO parallelism).
    This is the key difference from GSFL - SL is fully sequential!
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
        total_loss = 0
        total_up = 0
        total_down = 0
        total_compute = 0

        self.client.train()
        self.server.train()

        # In SL, ALL clients train SEQUENTIALLY (no parallelism)
        # This is the key limitation that GSFL addresses!
        for client_id, ds in enumerate(self.client_datasets):
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            
            # Do LOCAL_STEPS batches per client (same as GSFL)
            for step, (images, labels) in enumerate(loader):
                if step >= LOCAL_STEPS:
                    break
                    
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                smashed = self.client(images)
                
                # Use actual client_id for realistic bandwidth simulation
                total_up += comm_delay(smashed, client_id)

                logits = self.server(smashed)

                flops = smashed.nelement() * 100
                total_compute += compute_cost(flops)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                self.opt_client.zero_grad()
                self.opt_server.zero_grad()
                loss.backward()

                # Use actual client_id for realistic bandwidth simulation
                total_down += comm_delay(smashed, client_id)

                self.opt_server.step()
                self.opt_client.step()

        # SL sums ALL client latencies (fully sequential)
        # vs GSFL which takes MAX of parallel groups
        return total_loss, total_up, total_down, total_compute

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
