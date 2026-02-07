"""Train utilities."""

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from jaxtyping import Int
from jaxtyping import Float

from llm_with_jax_practice import data_loader
from llm_with_jax_practice import functions
from llm_with_jax_practice import optimizer
from llm_with_jax_practice import train_config


def loss_fn(
    model: nnx.Module,
    input_seq: Int[jnp.ndarray, "batch_size context_length"],
    target_seq: Int[jnp.ndarray, "batch_size context_length"],
) -> Float[jnp.ndarray, ""]:
    """Computes the loss for the model."""
    logits = model(input_seq)
    return functions.cross_entropy_loss(logits=logits, target_seq=target_seq)
