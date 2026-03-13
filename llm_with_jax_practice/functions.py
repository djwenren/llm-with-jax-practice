"""Functions for LLM with JAX Practice."""

import einops
import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


def silu(x: Float[jnp.ndarray, "..."]) -> Float[jnp.ndarray, "..."]:
    """Sigmoid-weighted linear unit (SiLU) activation function."""
    return x * jax.nn.sigmoid(x)


def softmax(x: Float[jnp.ndarray, "..."], axis: int) -> Float[jnp.ndarray, "..."]:
    """Softmax activation function."""
    x_diffed = x - x.max(axis=axis, keepdims=True)
    x_exped = jnp.exp(x_diffed)
    x_exp_sum = x_exped.sum(axis=axis, keepdims=True)
    return x_exped / x_exp_sum


def scaled_dot_product_attention(
    q: Float[jnp.ndarray, "...  queries_len d_k"],
    k: Float[jnp.ndarray, "... keys_len d_k"],
    v: Float[jnp.ndarray, "... keys_len d_v"],
    mask: Bool[jnp.ndarray, "... queries_len keys_len"] | None = None,
    attention_normalizer: float | None = None,
) -> Float[jnp.ndarray, "... queries_len values_len"]:
    """Scaled dot-product attention.

    Args:
        q: Query tensor of shape (..., queries_len, d_k).
        k: Key tensor of shape (..., keys_len, d_k).
        v: Value tensor of shape (..., values_len, d_v).
        mask: Mask tensor of shape (..., queries_len, keys_len). If not None, the positions where
            the attention should be kept are set to True.

    Returns:
        Float[jnp.ndarray, "... queries_len values_len"]: Output tensor.
    """
    attention_normalizer = attention_normalizer or 1.0 / np.sqrt(q.shape[-1])
    scaled_dot_product = (
        einops.einsum(
            q, k, "... queries_len d_k, ... keys_len d_k -> ... queries_len keys_len"
        )
        * attention_normalizer
    )
    if mask is not None:
        scaled_dot_product = jnp.where(mask, scaled_dot_product, -jnp.inf)
    return einops.einsum(
        softmax(scaled_dot_product, axis=-1),
        v,
        "... queries_len keys_len, ... keys_len d_v -> ... queries_len d_v",
    )


def cross_entropy_loss(
    logits: Float[jnp.ndarray, "... vocab_size"],
    target_seq: Int[jnp.ndarray, "..."],
) -> Float[jnp.ndarray, ""]:
    """Cross-entropy loss."""
    logits_shifted: Float[jnp.ndarray, "... vocab_size"] = logits - logits.max(
        axis=-1, keepdims=True
    )
    return jnp.mean(
        -jnp.take_along_axis(logits_shifted, target_seq[..., None], axis=-1).squeeze()
        + jnp.log(jnp.sum(jnp.exp(logits_shifted), axis=-1))
    )
