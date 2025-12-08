import random
import numpy as np

# Noise spectral density (approx)
NOISE_DENSITY = -174  # dBm/Hz  (thermal noise)

# Convert dBm to linear scale (mW)
def dbm_to_mw(x):
    return 10 ** (x / 10)

# Convert mW to linear watts
def mw_to_w(x):
    return x / 1000


# Distance of each client from base station (meters)
# You may customize these for experiments
CLIENT_DISTANCES = {
    0: 50,
    1: 100,
    2: 150,
    3: 200,
    4: 300,
    5: 400,
    6: 500,
    7: 600,
    8: 50,
    9: 100,
    10: 200,
    11: 300,
    12: 500,
    13: 600,
    14: 700,
    15: 800,
}

# Technology types (4G, 5G)
CLIENT_TECH = {
    cid: random.choice(["4G", "5G"])
    for cid in CLIENT_DISTANCES
}

# Transmit power in dBm (client → server)
TX_POWER = 23  # typical phone Tx power

# Base station carrier frequency (Hz)
FREQ = 3.5e9  # mid-band 5G

# Bandwidth per technology (Hz)
BW_TECH = {
    "4G": 20e6,    # 20 MHz
    "5G": 100e6    # 100 MHz
}
