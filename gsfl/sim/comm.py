from gsfl.config import UPLINK_BW, DOWNLINK_BW

def _tensor_num_bytes(tensor):
    # float32 = 4 bytes
    return tensor.nelement() * 4


def uplink_cost(tensor):
    """
    Simulated uplink time (client -> server) in seconds.
    """
    size = _tensor_num_bytes(tensor)
    return size / UPLINK_BW


def downlink_cost(tensor):
    """
    Simulated downlink time (server -> client) in seconds.
    We assume gradient has same size as the activation.
    """
    size = _tensor_num_bytes(tensor)
    return size / DOWNLINK_BW
