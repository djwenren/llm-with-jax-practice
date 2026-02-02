"""Layers for LLM with JAX Practice."""

import einops
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jaxtyping import Float


class Linear(nnx.Module):
    """Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = nnx.Param(
            rngs.truncated_normal(
                shape=(out_features, in_features),
                lower=-3.0 * std,
                upper=3.0 * std,
                dtype=dtype,
            )
        )

    def __call__(
        self, x: Float[jnp.ndarray, "... d_in"]
    ) -> Float[jnp.ndarray, "... d_out"]:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
