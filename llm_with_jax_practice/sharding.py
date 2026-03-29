"""Sharding for LLM with JAX Practice."""

import dataclasses

from absl import logging

import jax

from jax.sharding import PartitionSpec as P


@dataclasses.dataclass(kw_only=True, frozen=True)
class LinearSharding:
    """Sharding for linear layer."""

    weight: P | None = None
    out: P | None = None


@dataclasses.dataclass(kw_only=True, frozen=True)
class EmbeddingSharding:
    """Sharding for embedding layer."""

    embedding_matrix: P | None = None
    out: P | None = None


@dataclasses.dataclass(kw_only=True, frozen=True)
class RMSNormSharding:
    """Sharding for RMSNorm layer."""

    weight: P | None = None


@dataclasses.dataclass(kw_only=True, frozen=True)
class SwiGLUSharding:
    """Sharding for SwiGLU layer."""

    up_projection: LinearSharding = LinearSharding()
    down_projection: LinearSharding = LinearSharding()


@dataclasses.dataclass(kw_only=True, frozen=True)
class RoPESharding:
    """Sharding for RoPE layer.

    RoPE layer does not have any sharding.
    """


@dataclasses.dataclass(kw_only=True, frozen=True)
class MultiHeadSelfAttentionSharding:
    """Sharding for multi-head self-attention layer."""

    combined_in_projection: LinearSharding = LinearSharding()
    out_projection: LinearSharding = LinearSharding()


@dataclasses.dataclass(kw_only=True, frozen=True)
class TransformerBlockSharding:
    """Sharding for transformer block layer."""

    rms_norm_pre_attn: RMSNormSharding = RMSNormSharding()
    attn: MultiHeadSelfAttentionSharding = MultiHeadSelfAttentionSharding()
    rms_norm_pre_ff: RMSNormSharding = RMSNormSharding()
    ffn: SwiGLUSharding = SwiGLUSharding()


@dataclasses.dataclass(kw_only=True, frozen=True)
class TransformerLmSharding:
    """Sharding for transformer language model."""

    token_embeddings: EmbeddingSharding = EmbeddingSharding()
    transformer_blocks: TransformerBlockSharding = TransformerBlockSharding()
    ln_final: RMSNormSharding = RMSNormSharding()
    lm_head: LinearSharding = LinearSharding()


FSDP_SHARDING = TransformerLmSharding(
    token_embeddings=EmbeddingSharding(
        embedding_matrix=P(None, None), out=P("data", None, None)
    ),
    transformer_blocks=TransformerBlockSharding(
        rms_norm_pre_attn=RMSNormSharding(
            weight=P(
                None,
            )
        ),
        attn=MultiHeadSelfAttentionSharding(
            combined_in_projection=LinearSharding(
                weight=P("data", None), out=P("data", None, None)
            ),
            out_projection=LinearSharding(
                weight=P(None, "data"), out=P("data", None, None)
            ),
        ),
        rms_norm_pre_ff=RMSNormSharding(
            weight=P(
                None,
            )
        ),
        ffn=SwiGLUSharding(
            up_projection=LinearSharding(
                weight=P("data", None), out=P("data", None, None)
            ),
            down_projection=LinearSharding(
                weight=P(None, "data"), out=P("data", None, None)
            ),
        ),
    ),
    ln_final=RMSNormSharding(
        weight=P(
            None,
        )
    ),
    lm_head=LinearSharding(weight=P(None, None), out=P("data", None, None)),
)

FSDP_TP_SHARDING = TransformerLmSharding(
    token_embeddings=EmbeddingSharding(
        embedding_matrix=P(None, "model"), out=P("data", None, "model")
    ),
    transformer_blocks=TransformerBlockSharding(
        rms_norm_pre_attn=RMSNormSharding(
            weight=P(
                None,
            )
        ),
        attn=MultiHeadSelfAttentionSharding(
            combined_in_projection=LinearSharding(
                weight=P("data", "model"), out=P("data", None, "model")
            ),
            out_projection=LinearSharding(
                weight=P("model", "data"), out=P("data", None, "model")
            ),
        ),
        rms_norm_pre_ff=RMSNormSharding(
            weight=P(
                None,
            )
        ),
        ffn=SwiGLUSharding(
            up_projection=LinearSharding(
                weight=P("data", "model"), out=P("data", None, "model")
            ),
            down_projection=LinearSharding(
                weight=P("model", "data"), out=P("data", None, "model")
            ),
        ),
    ),
    ln_final=RMSNormSharding(
        weight=P(
            None,
        )
    ),
    lm_head=LinearSharding(weight=P("model", None), out=P("data", None, None)),
)


def get_mesh_and_sharding(
    sharding_strategy: str,
) -> tuple[jax.sharding.Mesh | None, TransformerLmSharding]:
    """Gets the mesh and sharding."""
    if sharding_strategy == "fsdp":
        logging.info("Setting mesh for FSDP sharding.")
        mesh = jax.make_mesh((4,), ("data",))
        return mesh, FSDP_SHARDING
    if sharding_strategy == "fsdp_tp":
        logging.info("Setting mesh for FSDP + TP sharding.")
        mesh = jax.make_mesh((4, 2), ("data", "model"))
        return mesh, FSDP_TP_SHARDING
    return None, TransformerLmSharding()
