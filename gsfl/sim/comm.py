import numpy as np
import random
from math import log10
from gsfl.sim.wireless_config import (
    CLIENT_DISTANCES,
    CLIENT_TECH,
    TX_POWER,
    FREQ,
    BW_TECH,
    NOISE_DENSITY,
    dbm_to_mw,
)

# Speed of light
C = 3e8


def path_loss(distance):
    """ Path loss using free space + shadowing. """

    # Free-space path loss (FSPL)
    wavelength = C / FREQ
    fspl = (4 * np.pi * distance / wavelength) ** 2

    # Convert to dB
    fspl_db = 10 * np.log10(fspl)

    # Shadow fading (log-normal)
    shadowing = np.random.normal(0, 4)  # 4 dB variance

    return fspl_db + shadowing


def rayleigh_fading():
    """ Rayleigh fading coefficient (power gain). """
    h = (np.random.normal(0, 1) + 1j*np.random.normal(0, 1)) / np.sqrt(2)
    return abs(h) ** 2


def compute_bandwidth(client_id):
    """ Compute instantaneous wireless bandwidth for a client. """

    distance = CLIENT_DISTANCES[client_id]
    tech = CLIENT_TECH[client_id]

    # Path loss (dB)
    pl_db = path_loss(distance)

    # Received SNR (in dB)
    rx_power_dbm = TX_POWER - pl_db
    noise_dbm = NOISE_DENSITY + 10*np.log10(BW_TECH[tech])
    snr_db = rx_power_dbm - noise_dbm

    # Apply Rayleigh fading
    snr_linear = 10 ** (snr_db / 10)
    snr_linear *= rayleigh_fading()

    # Shannon capacity (bits/sec)
    bandwidth = BW_TECH[tech] * np.log2(1 + snr_linear)

    return max(bandwidth, 1e3)  # avoid zero bandwidth


def comm_delay(tensor, client_id):
    """ Compute upload/download delay based on wireless model. """
    size_bytes = tensor.nelement() * 4  # float32

    bw = compute_bandwidth(client_id)   # dynamic wireless bandwidth

    return size_bytes / bw
