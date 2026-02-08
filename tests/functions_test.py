"""Functions test."""

import jax
import jax.numpy as jnp

from jaxtyping import Float
from jaxtyping import Int


from llm_with_jax_practice import functions


def _ref_cross_entropy_loss(
    logits: Float[jnp.ndarray, "vocab_size"], target_seq: Int[jnp.ndarray, ""]
) -> Float[jnp.ndarray, ""]:
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    return jnp.mean(
        -logits_shifted[target_seq] + jnp.log(jnp.sum(jnp.exp(logits_shifted), axis=-1))
    )


def test_cross_entropy_loss():
    """Test cross-entropy loss."""
    logits = jax.random.normal(jax.random.key(42), (8, 16, 32))
    target_seq = jax.random.randint(jax.random.key(42), (8, 16), 0, 32)
    loss = functions.cross_entropy_loss(logits, target_seq)
    expected_loss = jnp.mean(
        jax.vmap(jax.vmap(_ref_cross_entropy_loss))(logits, target_seq)
    )
    assert jnp.allclose(loss, expected_loss)
