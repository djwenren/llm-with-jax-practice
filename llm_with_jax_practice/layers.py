"""Layers for LLM with JAX Practice."""

import einops
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jaxtyping import Float
from jaxtyping import Int

from llm_with_jax_practice import functions


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
            * std
        )

    def __call__(
        self, x: Float[jnp.ndarray, "... d_in"]
    ) -> Float[jnp.ndarray, "... d_out"]:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nnx.Module):
    """Embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.weight = nnx.Param(
            rngs.truncated_normal(
                shape=(num_embeddings, embedding_dim),
                lower=-3.0,
                upper=3.0,
                dtype=dtype,
            )
        )

    def __call__(
        self, token_ids: Int[jnp.ndarray, "..."]
    ) -> Float[jnp.ndarray, "... embedding_dim"]:
        return self.weight[token_ids]


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        dtype=jnp.float32,
    ):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((d_model,), dtype=dtype))

    def __call__(
        self, x: Float[jnp.ndarray, "... d_model"]
    ) -> Float[jnp.ndarray, "... d_model"]:
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        result = (
            x
            * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
            * self.weight
        )
        return result.astype(in_dtype)


class SwiGLU(nnx.Module):
    """SwiGLU layer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.in_project_layer_1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            rngs=rngs,
            dtype=dtype,
        )
        self.in_project_layer_3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            rngs=rngs,
            dtype=dtype,
        )
        self.out_project_layer_2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
        self, x: Float[jnp.ndarray, "... d_model"]
    ) -> Float[jnp.ndarray, "... d_model"]:
        out_1 = self.in_project_layer_1(x)
        out_3 = self.in_project_layer_3(x)
        return self.out_project_layer_2(functions.silu(out_1) * out_3)


class RoPE(nnx.Module):
    """RoPE layer."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        theta_tensor = einops.einsum(
            jnp.arange(max_seq_len),
            1.0 / theta ** (2 * jnp.arange(d_k // 2) / d_k),
            "seq_len, half_d_k -> seq_len half_d_k",
        )
        cosine_matrix = einops.einsum(
            jnp.cos(theta_tensor),
            jnp.array([[1.0, 0], [0, 1.0]]),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        sine_matrix = einops.einsum(
            jnp.sin(theta_tensor),
            jnp.array([[0, -1.0], [1.0, 0]]),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        # https://gemini.google.com/app/2aad378109c833fe
        self.rope_matrix = nnx.Variable(cosine_matrix + sine_matrix)

    def __call__(
        self,
        x: Float[jnp.ndarray, "... seq_len d_k"],
        token_positions: Int[jnp.ndarray, "... seq_len"],
    ) -> Float[jnp.ndarray, "... seq_len d_k"]:
        position_embeddings: Float[jnp.ndarray, "... seq_len half_d_k r_out r_in"] = (
            self.rope_matrix[token_positions]
        )
        x_rearranged = einops.rearrange(
            x, "... seq_len (half_d_k r_in) -> ... seq_len half_d_k r_in", r_in=2
        )
        output = einops.einsum(
            x_rearranged,
            position_embeddings,
            (
                "... seq_len half_d_k r_in, ... seq_len half_d_k r_out r_in -> "
                "... seq_len half_d_k r_out"
            ),
        )
        return einops.rearrange(
            output,
            "... seq_len half_d_k r_out -> ... seq_len (half_d_k r_out)",
        )
