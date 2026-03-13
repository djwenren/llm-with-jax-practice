"""Layers for LLM with JAX Practice."""

from typing import Literal

import einops
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jax import Array
from jaxtyping import Float
from jaxtyping import Int

from llm_with_jax_practice import functions
from llm_with_jax_practice import sharding as _sharding

_MultiHeadSelfAttentionSharding = _sharding.MultiHeadSelfAttentionSharding


class Linear(nnx.Module):
    """Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.float32,
        sharding: _sharding.LinearSharding = _sharding.LinearSharding(),
        std: float | None = None,
    ):
        std = std or np.sqrt(2.0 / (in_features + out_features))
        self.weight = nnx.Param(
            rngs.truncated_normal(
                shape=(in_features, out_features),
                lower=-3.0 * std,
                upper=3.0 * std,
                dtype=dtype,
                out_sharding=sharding.weight,
            )
            * jnp.array(std, dtype=dtype)
        )
        self.out_sharding = sharding.out

    def __call__(
        self,
        x: Float[Array, "... d_in"],
    ) -> Float[Array, "... d_out"]:
        output = jnp.einsum(
            "...D, DF -> ... F", x, self.weight, out_sharding=self.out_sharding
        )
        return output


class Embedding(nnx.Module):
    """Embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rngs: nnx.Rngs,
        *,
        dtype=jnp.float32,
        sharding: _sharding.EmbeddingSharding = _sharding.EmbeddingSharding(),
        std: float | None = None,
        alpha_input: float | None = None,
    ):
        std = std or 1.0
        self.weight = nnx.Param(
            rngs.truncated_normal(
                shape=(num_embeddings, embedding_dim),
                lower=-3.0 * std,
                upper=3.0 * std,
                dtype=dtype,
                out_sharding=sharding.embedding_matrix,
            )
            * jnp.array(std, dtype=dtype)
        )
        self.out_sharding = sharding.out
        self.alpha_input = jnp.array(alpha_input or 1.0, dtype=dtype)

    def __call__(
        self, token_ids: Int[Array, "..."]
    ) -> Float[Array, "... embedding_dim"]:
        return (
            self.weight.at[token_ids].get(out_sharding=self.out_sharding)
            * self.alpha_input
        )


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        *,
        dtype=jnp.float32,
        sharding: _sharding.RMSNormSharding = _sharding.RMSNormSharding(),
    ):
        self.eps = eps
        self.weight = nnx.Param(
            jnp.ones((d_model,), dtype=dtype, out_sharding=sharding.weight)
        )

    def __call__(self, x: Float[Array, "... d_model"]) -> Float[Array, "... d_model"]:
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
        *,
        dtype=jnp.float32,
        sharding: _sharding.SwiGLUSharding = _sharding.SwiGLUSharding(),
        std: float | None = None,
    ):
        self.w1_projection = Linear(
            in_features=d_model,
            out_features=d_ff,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.up_projection,
            std=std,
        )
        self.w3_projection = Linear(
            in_features=d_model,
            out_features=d_ff,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.up_projection,
            std=std,
        )
        self.w2_projection = Linear(
            in_features=d_ff,
            out_features=d_model,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.down_projection,
            std=std,
        )

    def __call__(
        self,
        x: Float[Array, "... d_model"],
    ) -> Float[Array, "... d_model"]:
        w1_out = self.w1_projection(x)
        w3_out = self.w3_projection(x)
        return self.w2_projection(functions.silu(w1_out) * w3_out)


class RoPE(nnx.Module):
    """RoPE layer."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        theta_tensor = (
            jnp.arange(max_seq_len)[:, None]
            / (theta ** (2 * jnp.arange(d_k // 2) / d_k)[None, :])
        ).astype(dtype)
        cosine_matrix = jnp.cos(theta_tensor)[:, :, None, None] * (
            jnp.array([[1.0, 0], [0, 1.0]])[None, None, :, :]
        ).astype(dtype)
        sine_matrix = jnp.sin(theta_tensor)[:, :, None, None] * (
            jnp.array([[0, -1.0], [1.0, 0]])[None, None, :, :]
        ).astype(dtype)
        # https://gemini.google.com/app/2aad378109c833fe
        # Shape: (max_seq_len, half_d_k, 2, 2)
        self.rope_matrix = nnx.Variable(cosine_matrix + sine_matrix)

    def __call__(
        self,
        x: Float[Array, "... d_k"],
        token_positions: Int[Array, "..."],
    ) -> Float[Array, "... d_k"]:
        position_embeddings: Float[Array, "... half_d_k r_out r_in"] = self.rope_matrix[
            token_positions
        ]
        x_rearranged = einops.rearrange(
            x, "... (half_d_k r_in) -> ... half_d_k r_in", r_in=2
        )
        output = einops.einsum(
            x_rearranged,
            position_embeddings,
            ("... half_d_k r_in, ... half_d_k r_out r_in -> " "... half_d_k r_out"),
        )
        return einops.rearrange(
            output,
            "... half_d_k r_out -> ... (half_d_k r_out)",
        )


_ALLOWED_ATTENTION_TYPES = {"custom", "xla", "cudnn"}


class MultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rngs: nnx.Rngs,
        *,
        attention_type: Literal["custom", "xla", "cudnn"] = "custom",
        dtype: jnp.dtype = jnp.float32,
        sharding: _MultiHeadSelfAttentionSharding = _MultiHeadSelfAttentionSharding(),
        std: float | None = None,
    ):
        assert (
            d_model % num_heads
        ) == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}."
        assert attention_type in _ALLOWED_ATTENTION_TYPES, (
            f"Invalid attention type: {attention_type}. Allowed types are: "
            f"{_ALLOWED_ATTENTION_TYPES}."
        )
        self.attention_type = attention_type
        d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_head = d_head
        self.combined_in_projection = Linear(
            in_features=d_model,
            out_features=3 * d_model,
            dtype=dtype,
            rngs=rngs,
            sharding=sharding.combined_in_projection,
            # The elements of the combined in-projection matrix should still be initialized with the
            # same variance as the individual Q, K, V projection matrices. So even though the size
            # of the combined in-projection matrix is 3x larger, we still use the same std.
            std=std,
        )
        self.out_projection = Linear(
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            rngs=rngs,
            sharding=sharding.out_projection,
            std=std,
        )

    def __call__(
        self,
        in_features: Float[Array, "... seq_len d_model"],
        token_positions: Int[Array, "... seq_len"] | None = None,
        rope: RoPE | None = None,
    ) -> Float[Array, "... seq_len d_model"]:
        combined_in_projection = self.combined_in_projection(in_features)
        query, key, value = jnp.split(combined_in_projection, 3, axis=-1)
        if self.attention_type == "custom":
            scaled_dot_product_attention_result = self._call_custom_attention(
                query=query,
                key=key,
                value=value,
                token_positions=token_positions,
                rope=rope,
            )
        else:
            scaled_dot_product_attention_result = self._call_jax_attention(
                query=query,
                key=key,
                value=value,
                token_positions=token_positions,
                rope=rope,
            )
        return self.out_projection(scaled_dot_product_attention_result)

    def _call_custom_attention(
        self,
        *,
        query: Float[Array, "... seq_len d_model"],
        key: Float[Array, "... seq_len d_model"],
        value: Float[Array, "... seq_len d_model"],
        token_positions: Int[Array, "... seq_len"] | None = None,
        rope: RoPE | None = None,
    ) -> Float[Array, "... seq_len d_model"]:
        query = einops.rearrange(
            query,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        key = einops.rearrange(
            key,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        value = einops.rearrange(
            value,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        if rope is not None and token_positions is not None:
            query = rope(query, token_positions)
            key = rope(key, token_positions)
        seq_len = query.shape[-2]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
        scaled_dot_product_attention_result = functions.scaled_dot_product_attention(
            q=query, k=key, v=value, mask=mask
        )
        return einops.rearrange(
            scaled_dot_product_attention_result,
            "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)",
        )

    def _call_jax_attention(
        self,
        *,
        query: Float[Array, "... seq_len d_model"],
        key: Float[Array, "... seq_len d_model"],
        value: Float[Array, "... seq_len d_model"],
        token_positions: Int[Array, "... seq_len"] | None = None,
        rope: RoPE | None = None,
    ) -> Float[Array, "... seq_len d_model"]:
        query = einops.rearrange(
            query,
            "... seq_len (num_heads d_head) -> ... seq_len num_heads  d_head",
            num_heads=self.num_heads,
        )
        key = einops.rearrange(
            key,
            "... seq_len (num_heads d_head) -> ... seq_len num_heads d_head",
            num_heads=self.num_heads,
        )
        value = einops.rearrange(
            value,
            "... seq_len (num_heads d_head) -> ... seq_len num_heads d_head",
            num_heads=self.num_heads,
        )
        if rope is not None and token_positions is not None:
            # When using JAX's attention implementation, the head dimension is after the sequence
            # lenght dimension, so we need to add a singleton dimension to the token positions, so
            # that it broadcasts correctly.
            token_positions = token_positions[..., None]
            query = rope(query, token_positions)
            key = rope(key, token_positions)
        scaled_dot_product_attention_result = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=True,
            implementation=self.attention_type,
        )
        return einops.rearrange(
            scaled_dot_product_attention_result,
            "... seq_len num_heads d_head -> ... seq_len (num_heads d_head)",
        )


class TransformerBlock(nnx.Module):
    """Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.float32,
        eps: float = 1e-5,
        sharding: _sharding.TransformerBlockSharding = _sharding.TransformerBlockSharding(),
        attn_std: float | None = None,
        ffn_std: float | None = None,
    ):
        self.rms_norm_pre_attn = RMSNorm(d_model=d_model, eps=eps, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.attn,
            std=attn_std,
        )
        self.rms_norm_pre_ff = RMSNorm(d_model=d_model, eps=eps, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.ffn,
            std=ffn_std,
        )

    def __call__(
        self,
        in_features: Float[Array, "... seq_len d_model"],
        token_positions: Int[Array, "... seq_len"],
        rope: RoPE | None = None,
    ) -> Float[Array, "... seq_len d_model"]:
        activation = self.rms_norm_pre_attn(in_features)
        activation = self.attn(
            in_features=activation,
            token_positions=token_positions,
            rope=rope,
        )
        post_attn_block_activation = in_features + activation
        activation = self.rms_norm_pre_ff(post_attn_block_activation)
        activation = self.ffn(activation)
        return post_attn_block_activation + activation
