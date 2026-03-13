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
from llm_with_jax_practice import sharding as _sharding

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
_use_mu_p = flags.DEFINE_boolean(
    "use_mu_p", False, "Use mu-p parameter initialization."
)
_d_base = flags.DEFINE_integer(
    "d_base", 256, "Base dimension for mu-p parameter initialization."
)
_m_p = flags.DEFINE_integer(
    "m_p", 1, "The width scaling factor for mu-p parameter initialization."
)
_alpha_input = flags.DEFINE_float(
    "alpha_input", 1.0, "Alpha input for embedding layer."
)
_alpha_output = flags.DEFINE_float(
    "alpha_output", 1.0, "Alpha output for output layer."
)
_std_base = flags.DEFINE_float(
    "std_base", 0.001, "Base standard deviation for mu-p parameter initialization."
)


def _get_d_ff(d_model: int, d_ff_to_d_model: float | None, d_ff: int | None) -> int:
    if d_ff is not None:
        return d_ff
    assert d_ff_to_d_model is not None, "d_ff and d_ff_to_d_model cannot be both None."
    return math.ceil(d_model * d_ff_to_d_model / 64) * 64


@dataclasses.dataclass(kw_only=True, frozen=True)
class TransformerConfig:
    """Tranformer config."""

    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int

    rope_theta: float

    d_model: int | None = None
    d_ff_to_d_model: float | None = None
    d_ff: int | None = None

    use_mu_p: bool = False
    alpha_input: float | None = None
    alpha_output: float | None = None
    std_base: float | None = None
    d_base: int | None = None
    m_p: int | None = None


def get_transformer_config() -> TransformerConfig:
    """Gets transformer configuration."""
    if _use_mu_p.value:
        assert (
            _alpha_input.value is not None
        ), "alpha_input must be set when use_mu_p is True."
        assert (
            _alpha_output.value is not None
        ), "alpha_output must be set when use_mu_p is True."
        assert (
            _std_base.value is not None
        ), "std_base must be set when use_mu_p is True."
        assert _d_base.value is not None, "d_base must be set when use_mu_p is True."
        assert _m_p.value is not None, "m_p must be set when use_mu_p is True."
        assert (
            _d_ff_to_d_model.value is not None
        ), "d_ff_to_d_model must be set when use_mu_p is True."
        return TransformerConfig(
            vocab_size=_vocab_size.value,
            context_length=_context_length.value,
            num_layers=_num_layers.value,
            num_heads=_num_heads.value,
            rope_theta=_rope_theta.value,
            d_model=math.ceil(_d_base.value * _m_p.value),
            d_ff_to_d_model=_d_ff_to_d_model.value,
            d_ff=_get_d_ff(
                d_model=math.ceil(_d_base.value * _m_p.value),
                d_ff_to_d_model=_d_ff_to_d_model.value,
                d_ff=None,
            ),
            use_mu_p=_use_mu_p.value,
            alpha_input=_alpha_input.value,
            alpha_output=_alpha_output.value,
            std_base=_std_base.value,
            d_base=_d_base.value,
        )

    assert _d_model.value is not None, "d_model must be set when use_mu_p is False."
    assert (_d_ff_to_d_model.value is not None) or (
        _d_ff.value is not None
    ), "d_ff_to_d_model or d_ff must be set when use_mu_p is False."
    return TransformerConfig(
        vocab_size=_vocab_size.value,
        context_length=_context_length.value,
        num_layers=_num_layers.value,
        num_heads=_num_heads.value,
        rope_theta=_rope_theta.value,
        d_model=_d_model.value,
        d_ff_to_d_model=_d_ff_to_d_model.value,
        d_ff=_d_ff.value,
        use_mu_p=_use_mu_p.value,
        alpha_input=_alpha_input.value,
        alpha_output=_alpha_output.value,
        std_base=_std_base.value,
        d_base=_d_base.value,
        m_p=_m_p.value,
    )


class TransformerLm(nnx.Module):
    """Transformer language model."""

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        *,
        sharding: _sharding.TransformerLmSharding = _sharding.TransformerLmSharding(),
    ):
        if config.use_mu_p:
            self._mu_p_init(config=config, rngs=rngs, dtype=dtype, sharding=sharding)
        else:
            self._s_p_init(config=config, rngs=rngs, dtype=dtype, sharding=sharding)

    def _mu_p_init(
        self,
        *,
        config: TransformerConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        sharding: _sharding.TransformerLmSharding = _sharding.TransformerLmSharding(),
    ) -> None:
        """Initializes the transformer language model with mu-p parameter initialization."""
        self.token_embeddings = L.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.token_embeddings,
            std=config.std_base,
            alpha=config.alpha_input,
        )
        self.rope = L.RoPE(
            theta=config.rope_theta,
            d_k=config.d_model // config.num_heads,
            max_seq_len=config.context_length,
        )
        # Motivated by He/Xavier initialization, where the standard deviation of the weights is
        # sqrt(2.0 / (in_features + out_features)), we set the standard deviation of the weights in
        # the FFN to sqrt(1.0 / (1.0 + d_ff_to_d_model)).
        ffn_std_base = config.std_base * math.sqrt(1.0 / (1.0 + config.d_ff_to_d_model))

        @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None}, in_axes=(0,))
        def _create_transformer_block(rngs: nnx.Rngs) -> L.TransformerBlock:
            return L.TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=_get_d_ff(
                    d_model=config.d_model,
                    d_ff_to_d_model=config.d_ff_to_d_model,
                    d_ff=config.d_ff,
                ),
                rngs=rngs,
                dtype=dtype,
                sharding=sharding.transformer_blocks,
                use_mu_p=True,
                attn_std=config.std_base / math.sqrt(config.m_p),
                ffn_std=ffn_std_base / math.sqrt(config.m_p),
            )

        self.transformer_blocks = _create_transformer_block(
            rngs.fork(split=config.num_layers)
        )
        self.ln_final = L.RMSNorm(
            d_model=config.d_model, dtype=dtype, sharding=sharding.ln_final
        )
        self.lm_head = L.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.lm_head,
            std=config.std_base / math.sqrt(config.m_p),
            alpha=config.alpha_output / config.m_p,
        )

    def _s_p_init(
        self,
        *,
        config: TransformerConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        sharding: _sharding.TransformerLmSharding = _sharding.TransformerLmSharding(),
    ) -> None:
        """Initializes the transformer language model with standard parameter initialization."""
        self.token_embeddings = L.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.token_embeddings,
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
                d_ff=_get_d_ff(
                    d_model=config.d_model,
                    d_ff_to_d_model=config.d_ff_to_d_model,
                    d_ff=config.d_ff,
                ),
                rngs=rngs,
                dtype=dtype,
                sharding=sharding.transformer_blocks,
            )

        self.transformer_blocks = _create_transformer_block(
            rngs.fork(split=config.num_layers)
        )
        self.ln_final = L.RMSNorm(
            d_model=config.d_model, dtype=dtype, sharding=sharding.ln_final
        )
        self.lm_head = L.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            rngs=rngs,
            dtype=dtype,
            sharding=sharding.lm_head,
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
