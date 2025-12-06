from gsfl.config import DEVICE_FLOPS

def compute_cost(flops):
    """
    Simulated compute time (seconds) given an estimated FLOPs amount.
    """
    return flops / DEVICE_FLOPS
