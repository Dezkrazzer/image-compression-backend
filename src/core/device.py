"""
device.py

Detect available compute device.

Priority:
1. CUDA (NVIDIA)
2. MPS (Apple Silicon)
3. CPU
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Detect the best available compute device.

    Returns
    -------
    torch.device
        CUDA, MPS or CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def device_name() -> str:
    """
    Human readable device name.
    """

    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)

    if DEVICE.type == "mps":
        return "Apple Metal Performance Shader"

    return "CPU"