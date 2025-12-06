import torch

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data / clients
NUM_CLIENTS = 16
NUM_GROUPS = 4
BATCH_SIZE = 32

# Training
ROUNDS_SL = 10       # Split Learning baseline
ROUNDS_GSFL = 10     # GSFL
LR_CLIENT = 0.05
LR_SERVER = 0.05

# Simulation parameters
# bytes / second (approx)
UPLINK_BW = 5e6      # 5 MB/s
DOWNLINK_BW = 5e6

# FLOPs per second (approx CPU)
DEVICE_FLOPS = 5e9   # 5 GFLOPs/s
