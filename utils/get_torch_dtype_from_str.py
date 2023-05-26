import torch


def get_torch_dtype_from_str(s):
    if not s:
        return 'auto'
    elif s != 'auto':
        return getattr(torch, s)
    return s
