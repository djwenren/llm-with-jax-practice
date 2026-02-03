"""Functions for LLM with JAX Practice."""

import jax
import jax.numpy as jnp

from jaxtyping import Float


def silu(x: Float[jnp.ndarray, "..."]) -> Float[jnp.ndarray, "..."]:
    """Sigmoid-weighted linear unit (SiLU) activation function."""
    return x * jax.nn.sigmoid(x)
