"""
Compatibility shim for torch._six which was removed in PyTorch 1.9+
"""
import torch
import types

# Patch torch._six if it doesn't exist (PyTorch >= 1.9)
if not hasattr(torch, '_six'):
    torch._six = types.ModuleType('_six')
    torch._six.string_classes = (str, bytes)
