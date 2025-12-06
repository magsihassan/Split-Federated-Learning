import sys
import os

# Add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from gsfl.data import get_client_datasets, get_test_loader
from gsfl.core import SplitLearningTrainer
from gsfl.config import ROUNDS_SL

import json


def main():
    client_datasets = get_client_datasets()
    test_loader = get_test_loader()

    trainer = SplitLearningTrainer(client_datasets, test_loader)

    metrics = {
        "accuracy": [],
        "loss": [],
        "uplink": [],
        "downlink": [],
        "compute": []
    }

    print("=== Sequential Split Learning (SL) ===")
    for r in range(1, ROUNDS_SL + 1):
        loss, up, down, comp = trainer.train_round()
        acc = trainer.evaluate()

        metrics["accuracy"].append(acc)
        metrics["loss"].append(loss)
        metrics["uplink"].append(up)
        metrics["downlink"].append(down)
        metrics["compute"].append(comp)

        print(
            f"Round {r:02d}: loss={loss:.3f}, acc={acc:.3f}, "
            f"uplink={up:.4f}s, downlink={down:.4f}s, compute={comp:.4f}s"
        )

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/sl_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved: results/sl_results.json")


if __name__ == "__main__":
    main()
