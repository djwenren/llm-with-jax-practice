"""Functions for LLM with JAX Practice."""

import einops
import jax
import jax.numpy as jnp
import optax

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


def silu(x: Float[jnp.ndarray, "..."]) -> Float[jnp.ndarray, "..."]:
    """Sigmoid-weighted linear unit (SiLU) activation function."""
    return jax.nn.silu(x)


def softmax(x: Float[jnp.ndarray, "..."], axis: int) -> Float[jnp.ndarray, "..."]:
    """Softmax activation function."""
    return jax.nn.softmax(x, axis=axis)


def scaled_dot_product_attention(
    q: Float[jnp.ndarray, "...  queries_len d_k"],
    k: Float[jnp.ndarray, "... keys_len d_k"],
    v: Float[jnp.ndarray, "... keys_len d_v"],
    mask: Bool[jnp.ndarray, "... queries_len keys_len"] | None = None,
    attention_normalizer: Float[jnp.ndarray, ""] | None = None,
) -> Float[jnp.ndarray, "... queries_len values_len"]:
    """Scaled dot-product attention.

    Args:
        q: Query tensor of shape (..., queries_len, d_k).
        k: Key tensor of shape (..., keys_len, d_k).
        v: Value tensor of shape (..., values_len, d_v).
        mask: Mask tensor of shape (..., queries_len, keys_len). If not None, the positions where
            the attention should be kept are set to True.
        attention_normalizer: The normalizer for the attention logits.

    Returns:
        Float[jnp.ndarray, "... queries_len values_len"]: Output tensor.
    """
    dtype = q.dtype
    if attention_normalizer is None:
        attention_normalizer = 1.0 / jnp.sqrt(q.shape[-1])
    
    scaled_dot_product = (
        einops.einsum(
            q, k, "... q d, ... k d -> ... q k"
        )
        * attention_normalizer
    )
    if mask is not None:
        # Use a large negative value that is safe for the dtype.
        mask_value = jnp.finfo(dtype).min if dtype != jnp.bfloat16 else -1e30
        scaled_dot_product = jnp.where(mask, scaled_dot_product, mask_value)
    
    return einops.einsum(
        jax.nn.softmax(scaled_dot_product, axis=-1),
        v,
        "... q k, ... k d -> ... q d",
    )


def cross_entropy_loss(
    logits: Float[jnp.ndarray, "... vocab_size"],
    target_seq: Int[jnp.ndarray, "..."],
) -> Float[jnp.ndarray, ""]:
    """Cross-entropy loss."""
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_seq))
