# GSFL Project – Parallel & Distributed Computing

Implementation of:

- Sequential Split Learning (SL) baseline
- Group-based Split Federated Learning (GSFL)

using MNIST, PyTorch, and simulated wireless communication & computation costs.

## Structure

- `gsfl/models`: client-side and server-side neural networks
- `gsfl/data`: MNIST loading and partitioning across clients
- `gsfl/sim`: communication and computation delay models
- `gsfl/core/sl.py`: Sequential Split Learning trainer
- `gsfl/core/gsfl.py`: GSFL trainer
- `experiments/run_sl.py`: run SL baseline
- `experiments/run_gsfl.py`: run GSFL

## Setup

```bash
pip install -r requirements.txt
