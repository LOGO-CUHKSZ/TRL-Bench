"""
Shared OpenAI/OpenRouter client factory.

Checks environment variables in this order:
1. OPENROUTER_API_KEY → uses OpenRouter (https://openrouter.ai/api/v1)
2. OPENAI_API_KEY → uses OpenAI directly

Model name is auto-prefixed with "openai/" for OpenRouter if not already prefixed.

Model registry:
    text-embedding-3-small  → label "openai",     768 dim, supports dimensions param
    text-embedding-3-large  → label "openai_3l",   768 dim, supports dimensions param
    text-embedding-ada-002  → label "openai_ada", 1536 dim, fixed (no dimensions param)
"""

import os

# Model registry: model_name -> (default_label, native_dim, supports_dimensions)
MODEL_REGISTRY = {
    'text-embedding-3-small': ('openai', 1536, True),
    'text-embedding-3-large': ('openai_3l', 3072, True),
    'text-embedding-ada-002': ('openai_ada', 1536, False),
}


def create_client():
    """Create an OpenAI-compatible client, preferring OpenRouter if configured.

    Returns:
        (client, provider_name) tuple
    """
    from openai import OpenAI

    openrouter_key = os.environ.get('OPENROUTER_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if openrouter_key:
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return client, "openrouter"
    elif openai_key:
        client = OpenAI()
        return client, "openai"
    else:
        raise RuntimeError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY "
            "in the environment or .env file."
        )


def resolve_model_name(model_name: str, provider: str) -> str:
    """Add provider prefix for OpenRouter if needed."""
    if provider == "openrouter" and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def get_model_info(model_name: str):
    """Get (label, native_dim, supports_dimensions) for a model.

    Falls back to sensible defaults for unknown models.
    """
    # Strip openai/ prefix for lookup
    base_name = model_name.split('/')[-1] if '/' in model_name else model_name
    return MODEL_REGISTRY.get(base_name, (base_name.replace('-', '_'), 1536, True))


def supports_dimensions(model_name: str) -> bool:
    """Check if a model supports the dimensions parameter."""
    _, _, supports = get_model_info(model_name)
    return supports


def get_model_label(model_name: str) -> str:
    """Get the default directory label for a model."""
    label, _, _ = get_model_info(model_name)
    return label
