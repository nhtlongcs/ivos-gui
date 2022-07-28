import torch
from typing import Any


def detach(obj: Any):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to cpu
    Raises:
        TypeError: Invalid type for detach
    Returns:
        type(obj) -- same object but moved to cpu
    """
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        res = {k: detach(v) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach(list(obj)))
    else:
        raise TypeError("Invalid type for detach")
