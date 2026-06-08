"""Decoder registry for semantic parsing."""

from .base import DecoderBase

DECODER_REGISTRY = {}


def register_decoder(name: str):
    """Decorator to register a decoder."""
    def decorator(cls):
        DECODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_decoder(name: str) -> DecoderBase:
    """Get a decoder class by name."""
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(DECODER_REGISTRY.keys())}")
    return DECODER_REGISTRY[name]


# Import decoders to register them
from .mapo import MAPODecoder
