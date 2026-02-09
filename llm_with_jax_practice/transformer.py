"""Transformer for LLM with JAX Practice."""

import dataclasses
import math

import jax
import jax.numpy as jnp

from absl import flags
from jaxtyping import Float
from jaxtyping import Int
from flax import nnx

from llm_with_jax_practice import layers as L

_vocab_size = flags.DEFINE_integer("vocab_size", 1000, "Vocabulary size.")
_context_length = flags.DEFINE_integer("context_length", 16, "Context length.")
_num_layers = flags.DEFINE_integer("num_layers", 2, "Number of layers.")
_num_heads = flags.DEFINE_integer("num_heads", 4, "Number of heads.")
_rope_theta = flags.DEFINE_float("rope_theta", 10000, "RoPE theta.")
_d_model = flags.DEFINE_integer("d_model", 128, "Model dimension.")
_d_ff_to_d_model = flags.DEFINE_float(
    "d_ff_to_d_model", None, "FF dimension to model dimension ratio."
)
_d_ff = flags.DEFINE_integer("d_ff", None, "FF dimension.")


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


def get_transformer_config() -> TransformerConfig:
    """Gets transformer configuration."""
    return TransformerConfig(
        vocab_size=_vocab_size.value,
        context_length=_context_length.value,
        num_layers=_num_layers.value,
        num_heads=_num_heads.value,
        rope_theta=_rope_theta.value,
        d_model=_d_model.value,
        d_ff_to_d_model=_d_ff_to_d_model.value,
        d_ff=_d_ff.value,
    )


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

        @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None}, in_axes=(0,))
        def _create_transformer_block(rngs: nnx.Rngs) -> L.TransformerBlock:
            return L.TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=self._get_d_ff(
                    d_model=config.d_model,
                    d_ff_to_d_model=config.d_ff_to_d_model,
                    d_ff=config.d_ff,
                ),
                rngs=rngs,
                dtype=dtype,
            )

        self.transformer_blocks = _create_transformer_block(
            rngs.fork(split=config.num_layers)
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

        def scan_over_transformer_blocks(activation, transformer_block):
            return (
                transformer_block(
                    in_features=activation,
                    token_positions=token_positions,
                    rope=self.rope,
                ),
                None,
            )

        activation, _ = jax.lax.scan(
            scan_over_transformer_blocks, activation, self.transformer_blocks
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
