"""Mem and flops counters."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from llm_with_jax_practice import layers
from llm_with_jax_practice import transformer


@dataclasses.dataclass(kw_only=True, frozen=True)
class CountingResults:
    """Results of counting memory and flops."""

    num_trainable_params: int | np.int64
    num_non_trainable_params: int | np.int64
    state_bytes: int | np.int64
    optimizer_num_params: int | np.int64
    optimizer_param_bytes: int | np.int64
    num_activation_params: int | np.int64
    activation_bytes: int | np.int64
    flops: int | np.int64


class LinearMemAndFlopsCounter:
    """Mem and flops counter for linear layer."""

    def __init__(
        self, in_features: int, out_features: int, *, dtype: jnp.dtype = jnp.float32
    ):
        self._linear = nnx.eval_shape(
            lambda: layers.Linear(
                in_features,
                out_features,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._num_trainable_params = np.prod(self._linear.weight.shape)
        self._num_non_trainable_params = 0
        self._state_bytes = self._num_trainable_params * self._linear.weight.dtype.itemsize

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self, x: jax.ShapeDtypeStruct, is_training: bool = True
    ) -> CountingResults:
        """Count the memory and flops for the linear layer."""
        flops = (
            2
            * self._linear.weight.shape[0]
            * self._linear.weight.shape[1]
            * np.prod(x.shape[:-1])
        )
        if is_training:
            # Counting the cost of the backward pass.
            flops *= 3
            # Assuming the optimizer uses twice the number of parameters as the model, such as in
            # AdamW.
            optimizer_num_params = self._num_trainable_params * 2
            optimizer_param_bytes = self._state_bytes * 2
        else:
            optimizer_num_params = 0
            optimizer_param_bytes = 0
        # Pass both self._linear and x to the eval_shape function instead of capturing them so that
        # eval_shape can wrap them in a `jax.ShapeDtypeStruct`.
        output = nnx.eval_shape(lambda m, x: m(x), self._linear, x)
        num_activation_params = output.size
        activation_bytes = num_activation_params * output.dtype.itemsize
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=optimizer_num_params,
            optimizer_param_bytes=optimizer_param_bytes,
            num_activation_params=num_activation_params,
            activation_bytes=activation_bytes,
            flops=flops,
        )


class EmbeddingMemAndFlopsCounter:
    """Mem and flops counter for embedding layer."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, *, dtype: jnp.dtype = jnp.float32
    ):
        self._embedding = nnx.eval_shape(
            lambda: layers.Embedding(
                num_embeddings,
                embedding_dim,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._num_trainable_params = np.prod(self._embedding.weight.shape)
        self._num_non_trainable_params = 0
        self._state_bytes = self._num_trainable_params * self._embedding.weight.dtype.itemsize

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self, x: jax.ShapeDtypeStruct, is_training: bool = True
    ) -> CountingResults:
        """Count the memory and flops for the embedding layer."""
        output = nnx.eval_shape(lambda m, x: m(x), self._embedding, x)
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=2 * self._num_trainable_params if is_training else 0,
            optimizer_param_bytes=2 * self._state_bytes if is_training else 0,
            num_activation_params=output.size,
            activation_bytes=output.size * output.dtype.itemsize,
            flops=0,
        )


class RMSNormMemAndFlopsCounter:
    """Mem and flops counter for RMSNorm layer."""

    def __init__(self, d_model: int, *, dtype: jnp.dtype = jnp.float32):
        self._rmsnorm = nnx.eval_shape(
            lambda: layers.RMSNorm(d_model=d_model, eps=1e-5, dtype=dtype)
        )
        self._num_trainable_params = d_model
        self._num_non_trainable_params = 0
        self._state_bytes = self._num_trainable_params * self._rmsnorm.weight.dtype.itemsize

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self, x: jax.ShapeDtypeStruct, is_training: bool = True
    ) -> CountingResults:
        """Count the memory and flops for the RMSNorm layer."""
        output = nnx.eval_shape(lambda m, x: m(x), self._rmsnorm, x)
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=2 * self._num_trainable_params if is_training else 0,
            optimizer_param_bytes=2 * self._state_bytes if is_training else 0,
            num_activation_params=output.size,
            activation_bytes=output.size * output.dtype.itemsize,
            # We only count the leading order contributions of the flops, which is basically those
            # from matmuls. Non-linearity only contriutes second order contributions.
            flops=0,
        )


class SwiGLUMemAndFlopsCounter:
    """Mem and flops counter for SwiGLU layer."""

    def __init__(self, d_model: int, d_ff: int, *, dtype: jnp.dtype = jnp.float32):
        self._w1_projection = nnx.eval_shape(
            lambda: layers.Linear(
                in_features=d_model,
                out_features=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._w3_projection = nnx.eval_shape(
            lambda: layers.Linear(
                in_features=d_model,
                out_features=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._w2_projection = nnx.eval_shape(
            lambda: layers.Linear(
                in_features=d_ff,
                out_features=d_model,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._num_trainable_params = (
            self._w1_projection.weight.size
            + self._w3_projection.weight.size
            + self._w2_projection.weight.size
        )
        self._num_non_trainable_params = 0
        self._state_bytes = (
            self._w1_projection.weight.size * self._w1_projection.weight.dtype.itemsize
            + self._w3_projection.weight.size
            * self._w3_projection.weight.dtype.itemsize
            + self._w2_projection.weight.size
            * self._w2_projection.weight.dtype.itemsize
        )

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self, x: jax.ShapeDtypeStruct, is_training: bool = True
    ) -> CountingResults:
        """Count the memory and flops for the SwiGLU layer."""
        w1_out = nnx.eval_shape(lambda m, x: m(x), self._w1_projection, x)
        w3_out = nnx.eval_shape(lambda m, x: m(x), self._w3_projection, x)
        w2_out = nnx.eval_shape(lambda m, x: m(x), self._w2_projection, w1_out)
        flops = (
            2
            * self._w1_projection.weight.shape[0]
            * self._w1_projection.weight.shape[1]
            * np.prod(x.shape[:-1])
            + 2
            * self._w3_projection.weight.shape[0]
            * self._w3_projection.weight.shape[1]
            * np.prod(x.shape[:-1])
            + 2
            * self._w2_projection.weight.shape[0]
            * self._w2_projection.weight.shape[1]
            * np.prod(w1_out.shape[:-1])
        )
        num_activation_params = w1_out.size + w2_out.size + w3_out.size
        activation_bytes = num_activation_params * w2_out.dtype.itemsize
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=2 * self._num_trainable_params if is_training else 0,
            optimizer_param_bytes=2 * self._state_bytes if is_training else 0,
            num_activation_params=num_activation_params,
            activation_bytes=activation_bytes,
            flops=3 * flops if is_training else flops,
        )


class RoPEMemAndFlopsCounter:
    """Mem and flops counter for RoPE layer."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        self._rope = nnx.eval_shape(
            lambda: layers.RoPE(
                theta=theta, d_k=d_k, max_seq_len=max_seq_len, dtype=dtype
            )
        )
        self._num_trainable_params = 0
        self._num_non_trainable_params = self._rope.rope_matrix.size
        self._state_bytes = self._num_non_trainable_params * self._rope.rope_matrix.dtype.itemsize

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self,
        x: jax.ShapeDtypeStruct,
        token_positions: jax.ShapeDtypeStruct,
        is_training: bool = True,
    ) -> CountingResults:
        """Count the memory and flops for the RoPE layer."""
        del is_training
        output = nnx.eval_shape(
            lambda m, x, tp: m(x, tp), self._rope, x, token_positions
        )
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            # RoPE is not optimized, so we don't count the optimizer parameters.
            optimizer_num_params=0,
            optimizer_param_bytes=0,
            num_activation_params=output.size,
            activation_bytes=output.size * output.dtype.itemsize,
            flops=0,
        )


class MultiHeadSelfAttentionMemAndFlopsCounter:
    """Mem and flops counter for Multi-head self-attention layer."""

    def __init__(self, d_model: int, num_heads: int, *, dtype: jnp.dtype = jnp.float32):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self._combined_in_projection = nnx.eval_shape(
            lambda: layers.Linear(
                in_features=d_model,
                out_features=3 * d_model,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._out_projection = nnx.eval_shape(
            lambda: layers.Linear(
                in_features=d_model,
                out_features=d_model,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._num_trainable_params = (
            self._combined_in_projection.weight.size + self._out_projection.weight.size
        )
        self._num_non_trainable_params = 0
        self._state_bytes = (
            self._combined_in_projection.weight.size
            * self._combined_in_projection.weight.dtype.itemsize
            + self._out_projection.weight.size
            * self._out_projection.weight.dtype.itemsize
        )

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self,
        x: jax.ShapeDtypeStruct,
        token_positions: jax.ShapeDtypeStruct,
        is_training: bool = True,
        use_flash_attention: bool = False,
        rope: RoPEMemAndFlopsCounter | None = None,
    ) -> CountingResults:
        """Count the memory and flops for the Multi-head self-attention layer."""
        B, T, D = x.shape  # pylint: disable=invalid-name
        S = T  # pylint: disable=invalid-name
        # in_projection.
        combined_in_projection = nnx.eval_shape(
            lambda m, x: m(x), self._combined_in_projection, x
        )
        num_activation_params = combined_in_projection.size
        activation_bytes = num_activation_params * combined_in_projection.dtype.itemsize
        flops = (
            2
            * self._combined_in_projection.weight.shape[0]
            * self._combined_in_projection.weight.shape[1]
            * np.prod(x.shape[:-1])
        )

        # rope
        # Note that rope is a referenceto to a RoPEMemAndFlopsCounter object, not the RoPE layer
        # itself, so we only count the memory and flops for the RoPE object, not its internal
        # parameters.
        if rope is not None:
            # Query and key are reshaped to (B, num_heads, T, d_head)
            # but RoPE layer expects (... seq_len d_k).
            # Layers.py MultiHeadSelfAttention calls rope(query, token_positions)
            # where query is (..., num_heads, seq_len, d_head).
            # RoPE.__call__ expects x: (..., seq_len, d_k)
            # and it will treat num_heads as a batch dimension.
            rope_input_shape = jax.ShapeDtypeStruct(
                shape=(B, self.num_heads, T, self.d_head), dtype=x.dtype
            )
            # Broadcast token_positions to (B, num_heads, T)
            rope_token_positions = jax.ShapeDtypeStruct(
                shape=(B, self.num_heads, T), dtype=token_positions.dtype
            )
            rope_results = rope.count(rope_input_shape, rope_token_positions)
            num_activation_params += rope_results.num_activation_params
            activation_bytes += rope_results.activation_bytes
            flops += rope_results.flops

        # dot product attention.
        flops += 2 * B * T * S * D * 2
        if use_flash_attention:
            num_new_activation_params = B * S * D
            num_activation_params += num_new_activation_params
            activation_bytes += (
                num_new_activation_params * combined_in_projection.dtype.itemsize
            )
        else:
            num_new_activation_params = B * S * D + B * T * S * self.num_heads
            num_activation_params += num_new_activation_params
            activation_bytes += (
                num_new_activation_params * combined_in_projection.dtype.itemsize
            )
        # out_projection
        out_projection = nnx.eval_shape(lambda m, x: m(x), self._out_projection, x)
        flops += (
            2
            * self._out_projection.weight.shape[0]
            * self._out_projection.weight.shape[1]
            * np.prod(out_projection.shape[:-1])
        )
        num_activation_params += out_projection.size
        activation_bytes += out_projection.size * out_projection.dtype.itemsize
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=2 * self._num_trainable_params if is_training else 0,
            optimizer_param_bytes=2 * self._state_bytes if is_training else 0,
            num_activation_params=num_activation_params,
            activation_bytes=activation_bytes,
            flops=3 * flops if is_training else flops,
        )


class TransformerBlockMemAndFlopsCounter:
    """Mem and flops counter for Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        self._rms_norm_pre_attn = RMSNormMemAndFlopsCounter(
            d_model=d_model, dtype=dtype
        )
        self._attn = MultiHeadSelfAttentionMemAndFlopsCounter(
            d_model=d_model, num_heads=num_heads, dtype=dtype
        )
        self._rms_norm_pre_ff = RMSNormMemAndFlopsCounter(d_model=d_model, dtype=dtype)
        self._ffn = SwiGLUMemAndFlopsCounter(d_model=d_model, d_ff=d_ff, dtype=dtype)
        self._num_trainable_params = (
            self._rms_norm_pre_attn.num_trainable_params
            + self._attn.num_trainable_params
            + self._rms_norm_pre_ff.num_trainable_params
            + self._ffn.num_trainable_params
        )
        self._num_non_trainable_params = (
            self._rms_norm_pre_attn.num_non_trainable_params
            + self._attn.num_non_trainable_params
            + self._rms_norm_pre_ff.num_non_trainable_params
            + self._ffn.num_non_trainable_params
        )
        self._state_bytes = (
            self._rms_norm_pre_attn.state_bytes
            + self._attn.state_bytes
            + self._rms_norm_pre_ff.state_bytes
            + self._ffn.state_bytes
        )

    @property
    def num_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self._num_trainable_params

    @property
    def num_non_trainable_params(self) -> int:
        """Returns the number of non-trainable parameters."""
        return self._num_non_trainable_params

    @property
    def state_bytes(self) -> int:
        """Returns the number of state bytes."""
        return self._state_bytes

    def count(
        self,
        x: jax.ShapeDtypeStruct,
        token_positions: jax.ShapeDtypeStruct,
        is_training: bool = True,
        use_flash_attention: bool = False,
        rope: RoPEMemAndFlopsCounter | None = None,
    ) -> CountingResults:
        """Count the memory and flops for the Transformer block."""
        rms_norm_pre_attn_results = self._rms_norm_pre_attn.count(
            x, is_training=is_training
        )
        attn_results = self._attn.count(
            x,
            token_positions,
            is_training=is_training,
            use_flash_attention=use_flash_attention,
            rope=rope,
        )
        rms_norm_pre_ff_results = self._rms_norm_pre_ff.count(
            x, is_training=is_training
        )
        ffn_results = self._ffn.count(x, is_training=is_training)
        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=(
                rms_norm_pre_attn_results.optimizer_num_params
                + attn_results.optimizer_num_params
                + rms_norm_pre_ff_results.optimizer_num_params
                + ffn_results.optimizer_num_params
            ),
            optimizer_param_bytes=(
                rms_norm_pre_attn_results.optimizer_param_bytes
                + attn_results.optimizer_param_bytes
                + rms_norm_pre_ff_results.optimizer_param_bytes
                + ffn_results.optimizer_param_bytes
            ),
            num_activation_params=(
                rms_norm_pre_attn_results.num_activation_params
                + attn_results.num_activation_params
                + rms_norm_pre_ff_results.num_activation_params
                + ffn_results.num_activation_params
            ),
            activation_bytes=(
                rms_norm_pre_attn_results.activation_bytes
                + attn_results.activation_bytes
                + rms_norm_pre_ff_results.activation_bytes
                + ffn_results.activation_bytes
            ),
            flops=(
                rms_norm_pre_attn_results.flops
                + attn_results.flops
                + rms_norm_pre_ff_results.flops
                + ffn_results.flops
            ),
        )


class TransformerLmMemAndFlopsCounter:
    """Mem and flops counter for Transformer language model."""

    def __init__(
        self,
        config: transformer.TransformerConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        self._dtype = dtype
        self._token_embeddings = EmbeddingMemAndFlopsCounter(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            dtype=dtype,
        )
        self._rope = RoPEMemAndFlopsCounter(
            theta=config.rope_theta,
            d_k=config.d_model // config.num_heads,
            max_seq_len=config.context_length,
            dtype=dtype,
        )
        d_ff = transformer.TransformerLm._get_d_ff(
            None,
            d_model=config.d_model,
            d_ff_to_d_model=config.d_ff_to_d_model,
            d_ff=config.d_ff,
        )
        self._transformer_block = TransformerBlockMemAndFlopsCounter(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=d_ff,
            dtype=dtype,
        )
        self._ln_final = RMSNormMemAndFlopsCounter(d_model=config.d_model, dtype=dtype)
        self._lm_head = LinearMemAndFlopsCounter(
            in_features=config.d_model,
            out_features=config.vocab_size,
            dtype=dtype,
        )
        self._num_layers = config.num_layers
        self._num_trainable_params = (
            self._token_embeddings.num_trainable_params
            + self._transformer_block.num_trainable_params * config.num_layers
            + self._ln_final.num_trainable_params
            + self._lm_head.num_trainable_params
        )
        self._num_non_trainable_params = (
            self._token_embeddings.num_non_trainable_params
            + self._rope.num_non_trainable_params
            + self._transformer_block.num_non_trainable_params * config.num_layers
            + self._ln_final.num_non_trainable_params
            + self._lm_head.num_non_trainable_params
        )
        self._state_bytes = (
            self._token_embeddings.state_bytes
            + self._rope.state_bytes
            + self._transformer_block.state_bytes * config.num_layers
            + self._ln_final.state_bytes
            + self._lm_head.state_bytes
        )

    def count(
        self,
        input_tokens: jax.ShapeDtypeStruct,
        is_training: bool = True,
        use_flash_attention: bool = False,
    ) -> CountingResults:
        """Count the memory and flops for the Transformer language model."""
        B, T = input_tokens.shape  # pylint: disable=invalid-name
        D = self._ln_final.num_trainable_params  # pylint: disable=invalid-name
        x = jax.ShapeDtypeStruct(shape=(B, T, D), dtype=self._dtype)
        token_positions = jax.ShapeDtypeStruct(shape=(B, T), dtype=jnp.int32)

        token_embeddings_results = self._token_embeddings.count(
            input_tokens, is_training=is_training
        )
        # We don't call self._rope.count here because it is called inside transformer_block.count
        # rope_results = self._rope.count(x, token_positions, is_training=is_training)
        # Actually in TransformerLm.__call__, rope is passed to transformer_block.
        transformer_block_results = self._transformer_block.count(
            x,
            token_positions,
            is_training=is_training,
            use_flash_attention=use_flash_attention,
            rope=self._rope,
        )
        ln_final_results = self._ln_final.count(x, is_training=is_training)
        lm_head_results = self._lm_head.count(x, is_training=is_training)

        return CountingResults(
            num_trainable_params=self._num_trainable_params,
            num_non_trainable_params=self._num_non_trainable_params,
            state_bytes=self._state_bytes,
            optimizer_num_params=(
                token_embeddings_results.optimizer_num_params
                + transformer_block_results.optimizer_num_params * self._num_layers
                + ln_final_results.optimizer_num_params
                + lm_head_results.optimizer_num_params
            ),
            optimizer_param_bytes=(
                token_embeddings_results.optimizer_param_bytes
                + transformer_block_results.optimizer_param_bytes * self._num_layers
                + ln_final_results.optimizer_param_bytes
                + lm_head_results.optimizer_param_bytes
            ),
            num_activation_params=(
                token_embeddings_results.num_activation_params
                + transformer_block_results.num_activation_params * self._num_layers
                + ln_final_results.num_activation_params
                + lm_head_results.num_activation_params
            ),
            activation_bytes=(
                token_embeddings_results.activation_bytes
                + transformer_block_results.activation_bytes * self._num_layers
                + ln_final_results.activation_bytes
                + lm_head_results.activation_bytes
            ),
            flops=(
                token_embeddings_results.flops
                + transformer_block_results.flops * self._num_layers
                + ln_final_results.flops
                + lm_head_results.flops
            ),
        )
