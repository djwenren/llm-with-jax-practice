"""Transformer for LLM with JAX Practice."""

import dataclasses
import math

import jax.numpy as jnp
from jaxtyping import Float
from jaxtyping import Int

from flax import nnx

from llm_with_jax_practice import layers as L


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    """Tranformer config."""

    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int

    rope_theta: float

    d_model: int
    d_ff_to_d_model: float | None = None
    d_ff: int | None = None


class TransformerLm(nnx.Module):
    """Transformer language model."""

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.token_embeddings = L.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            rngs=rngs,
            dtype=dtype,
        )
        self.rope = L.RoPE(
            theta=config.rope_theta,
            d_k=config.d_model // config.num_heads,
            max_seq_len=config.context_length,
        )
        self.transformer_blocks = nnx.List(
            [
                L.TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=self._get_d_ff(
                        d_model=config.d_model,
                        d_ff_to_d_model=config.d_ff_to_d_model,
                        d_ff=config.d_ff,
                    ),
                    rngs=rngs,
                    rope=self.rope,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_final = L.RMSNorm(d_model=config.d_model, dtype=dtype)
        self.lm_head = L.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
        self, input_tokens: Int[jnp.ndarray, "... seq_len"]
    ) -> Float[jnp.ndarray, "... seq_len vocab_size"]:
        activation = self.token_embeddings(input_tokens)
        token_positions = jnp.arange(input_tokens.shape[-1])
        for transformer_block in self.transformer_blocks:
            activation = transformer_block(
                in_features=activation,
                token_positions=token_positions,
            )
        activation = self.ln_final(activation)
        return self.lm_head(activation)

    def _get_d_ff(
        self, d_model: int, d_ff_to_d_model: float | None, d_ff: int | None
    ) -> int:
        if d_ff is not None:
            return d_ff
        assert (
            d_ff_to_d_model is not None
        ), "d_ff and d_ff_to_d_model cannot be both None."
        return math.ceil(d_model * d_ff_to_d_model / 64) * 64
