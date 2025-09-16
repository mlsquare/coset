"""
Distributed training components for quantized operations
"""

from .hooks import QuantizedGradientHook
from .communicator import QuantizedNCCLCommunicator

__all__ = [
    "QuantizedGradientHook",
    "QuantizedNCCLCommunicator",
]
