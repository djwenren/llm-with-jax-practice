"""Tests for mems and flops counters."""

import jax
import jax.numpy as jnp
import pytest

from llm_with_jax_practice import mem_and_flops_counters


class TestMemAndFlopsCounters:
    """Tests for mem and flops counters."""

    @pytest.mark.parametrize("is_training", [True, False])
    def test_linear(self, is_training):
        """Test linear mem and flops counter."""
        in_features = 10
        out_features = 20
        dtype = jnp.float16
        mem_and_flops_counter = mem_and_flops_counters.LinearMemAndFlopsCounter(
            in_features, out_features, dtype=dtype
        )
        x = jax.ShapeDtypeStruct(shape=(10, 20, in_features), dtype=jnp.float32)
        results = mem_and_flops_counter.count(x, is_training=is_training)
        assert results.num_trainable_params == in_features * out_features
        assert results.num_non_trainable_params == 0
        assert results.state_bytes == in_features * out_features * 2
        if is_training:
            assert results.optimizer_num_params == in_features * out_features * 2
            assert results.optimizer_param_bytes == in_features * out_features * 2 * 2
        else:
            assert results.optimizer_num_params == 0
            assert results.optimizer_param_bytes == 0
        assert results.num_activation_params == out_features * 10 * 20
        assert results.activation_bytes == out_features * 10 * 20 * 4
        assert results.flops == 2 * in_features * out_features * 10 * 20 * (
            3 if is_training else 1
        )

    @pytest.mark.parametrize("is_training", [True, False])
    def test_swiglu(self, is_training):
        """Test SwiGLU mem and flops counter."""
        d_model = 10
        d_ff = 20
        dtype = jnp.float16
        mem_and_flops_counter = mem_and_flops_counters.SwiGLUMemAndFlopsCounter(
            d_model, d_ff, dtype=dtype
        )
        x = jax.ShapeDtypeStruct(shape=(10, 20, d_model), dtype=jnp.float32)
        results = mem_and_flops_counter.count(x, is_training=is_training)
        assert results.num_trainable_params == 3 * d_model * d_ff
        assert results.num_non_trainable_params == 0
        assert results.state_bytes == 3 * d_model * d_ff * 2
        if is_training:
            assert results.optimizer_num_params == 3 * d_model * d_ff * 2
            assert results.optimizer_param_bytes == 3 * d_model * d_ff * 2 * 2
        else:
            assert results.optimizer_num_params == 0
            assert results.optimizer_param_bytes == 0
        assert results.num_activation_params == 10 * 20 * (d_model + d_ff * 2)
        assert results.activation_bytes == 10 * 20 * (d_model + d_ff * 2) * 4
        assert results.flops == 3 * 2 * d_model * d_ff * 10 * 20 * (
            3 if is_training else 1
        )

    @pytest.mark.parametrize("is_training", [True, False])
    def test_multi_head_self_attention_with_flash_attention(self, is_training):
        """Test Multi-head self-attention mem and flops counter."""
        d_model = 16
        num_heads = 4
        dtype = jnp.float16
        mem_and_flops_counter = (
            mem_and_flops_counters.MultiHeadSelfAttentionMemAndFlopsCounter(
                d_model, num_heads, dtype=dtype
            )
        )
        B, T, D = 2, 8, d_model
        x = jax.ShapeDtypeStruct(shape=(B, T, D), dtype=jnp.float32)
        token_positions = jax.ShapeDtypeStruct(shape=(B, T), dtype=jnp.int32)
        results = mem_and_flops_counter.count(
            x, token_positions, is_training=is_training, use_flash_attention=True
        )

        expected_num_params = d_model * (3 * d_model) + d_model * d_model
        assert results.num_trainable_params == expected_num_params
        assert results.num_non_trainable_params == 0
        assert results.state_bytes == expected_num_params * 2

        # in_proj: B*T*(3*D), attn: B*T*D, out_proj: B*T*D
        expected_num_activation_params = B * T * (3 * D) + B * T * D + B * T * D
        assert results.num_activation_params == expected_num_activation_params
        assert results.activation_bytes == expected_num_activation_params * 4

        # in_proj: 2*D*(3*D)*B*T, attn: 2*B*T*T*D*2, out_proj: 2*D*D*B*T
        flops_per_pass = (
            2 * D * (3 * D) * B * T + 2 * B * T * T * D * 2 + 2 * D * D * B * T
        )
        assert results.flops == flops_per_pass * (3 if is_training else 1)

    @pytest.mark.parametrize("is_training", [True, False])
    def test_multi_head_self_attention_without_flash_attention(self, is_training):
        """Test Multi-head self-attention mem and flops counter."""
        d_model = 16
        num_heads = 4
        dtype = jnp.float16
        mem_and_flops_counter = (
            mem_and_flops_counters.MultiHeadSelfAttentionMemAndFlopsCounter(
                d_model, num_heads, dtype=dtype
            )
        )
        B, T, D = 2, 8, d_model
        x = jax.ShapeDtypeStruct(shape=(B, T, D), dtype=jnp.float32)
        token_positions = jax.ShapeDtypeStruct(shape=(B, T), dtype=jnp.int32)
        results = mem_and_flops_counter.count(
            x, token_positions, is_training=is_training, use_flash_attention=False
        )

        expected_num_params = d_model * (3 * d_model) + d_model * d_model
        assert results.num_trainable_params == expected_num_params
        assert results.num_non_trainable_params == 0
        assert results.state_bytes == expected_num_params * 2

        # in_proj: B*T*(3*D), attn: B*T*D + B*T*T*num_heads, out_proj: B*T*D
        expected_num_activation_params = (
            B * T * (3 * D) + (B * T * D + B * T * T * num_heads) + B * T * D
        )
        assert results.num_activation_params == expected_num_activation_params
        assert results.activation_bytes == expected_num_activation_params * 4

        # in_proj: 2*D*(3*D)*B*T, attn: 2*B*T*T*D*2, out_proj: 2*D*D*B*T
        flops_per_pass = (
            2 * D * (3 * D) * B * T + 2 * B * T * T * D * 2 + 2 * D * D * B * T
        )
        assert results.flops == flops_per_pass * (3 if is_training else 1)

    @pytest.mark.parametrize("is_training", [True, False])
    def test_transformer_block(self, is_training):
        """Test Transformer block mem and flops counter."""
        d_model = 16
        num_heads = 4
        d_ff = 32
        dtype = jnp.float16
        mem_and_flops_counter = (
            mem_and_flops_counters.TransformerBlockMemAndFlopsCounter(
                d_model, num_heads, d_ff, dtype=dtype
            )
        )
        B, T, D = 2, 8, d_model
        x = jax.ShapeDtypeStruct(shape=(B, T, D), dtype=jnp.float32)
        token_positions = jax.ShapeDtypeStruct(shape=(B, T), dtype=jnp.int32)
        results = mem_and_flops_counter.count(
            x, token_positions, is_training=is_training
        )

        # rms_norm_pre_attn: D, attn: D*3D + D*D, rms_norm_pre_ff: D, ffn: 3*D*D_ff
        expected_num_params = (
            d_model + (3 * d_model**2 + d_model**2) + d_model + (3 * d_model * d_ff)
        )
        assert results.num_trainable_params == expected_num_params
        assert results.num_non_trainable_params == 0
        assert results.state_bytes == expected_num_params * 2
        if is_training:
            assert results.optimizer_num_params == expected_num_params * 2
            assert results.optimizer_param_bytes == expected_num_params * 2 * 2
        else:
            assert results.optimizer_num_params == 0
            assert results.optimizer_param_bytes == 0

        # RMSNorm1: B*T*D, MHSA: B*T*(5D + T*H), RMSNorm2: B*T*D, SwiGLU: B*T*(2*D_ff + D)
        expected_num_activation_params = (
            B * T * (8 * d_model + T * num_heads + 2 * d_ff)
        )
        assert results.num_activation_params == expected_num_activation_params
        assert results.activation_bytes == expected_num_activation_params * 4

        # MHSA: 2 * D * 3D * B * T + 4 * B * T * T * D + 2 * D * D * B * T = B*T*(8*D^2 + 4*T*D)
        # SwiGLU: 6 * D * D_ff * B * T
        flops_per_pass = B * T * (8 * d_model**2 + 4 * T * d_model + 6 * d_model * d_ff)
        assert results.flops == flops_per_pass * (3 if is_training else 1)

    @pytest.mark.parametrize("is_training", [True, False])
    def test_transformer_lm(self, is_training):
        """Test Transformer language model mem and flops counter."""
        from llm_with_jax_practice import transformer

        config = transformer.TransformerConfig(
            vocab_size=100,
            context_length=16,
            num_layers=2,
            num_heads=4,
            rope_theta=10000,
            d_model=32,
            d_ff=64,
        )
        dtype = jnp.float16
        mem_and_flops_counter = mem_and_flops_counters.TransformerLmMemAndFlopsCounter(
            config, dtype=dtype
        )
        B, T = 2, 8
        input_tokens = jax.ShapeDtypeStruct(shape=(B, T), dtype=jnp.int32)
        results = mem_and_flops_counter.count(input_tokens, is_training=is_training)

        # TransformerBlock params: 2*D (RMSNorms) + (4*D^2) (MHSA) + (3*D*D_ff) (SwiGLU)
        block_params = (
            2 * config.d_model
            + (4 * config.d_model**2)
            + (3 * config.d_model * config.d_ff)
        )

        expected_num_trainable_params = (
            config.vocab_size * config.d_model  # token_embeddings
            + config.num_layers * block_params  # transformer_blocks
            + config.d_model  # ln_final
            + config.d_model * config.vocab_size  # lm_head
        )
        assert results.num_trainable_params == expected_num_trainable_params

        # RoPE params are context_length * (D/num_heads/2) * 2 * 2 = 16 * (32/4/2) * 2 * 2 = 256.
        # But wait, RoPE is float32 by default in my implementation?
        # I fixed it to use dtype. So 256 params.
        expected_num_non_trainable_params = 256
        assert results.num_non_trainable_params == expected_num_non_trainable_params

        assert (
            results.state_bytes
            == (expected_num_trainable_params + expected_num_non_trainable_params) * 2
        )

        if is_training:
            assert results.optimizer_num_params == expected_num_trainable_params * 2
            assert (
                results.optimizer_param_bytes == expected_num_trainable_params * 2 * 2
            )
        else:
            assert results.optimizer_num_params == 0
            assert results.optimizer_param_bytes == 0

        # Total LM: B*T*D (Embed) + L * B*T*(9*D + T*H + 2*D_ff) + B*T*D (LN) + B*T*V (LM Head)
        expected_num_activation_params = (
            B
            * T
            * (
                2 * config.d_model
                + config.vocab_size
                + config.num_layers
                * (9 * config.d_model + T * config.num_heads + 2 * config.d_ff)
            )
        )
        assert results.num_activation_params == expected_num_activation_params
        # Since dtype=jnp.float16, activations should use 2 bytes per param.
        assert results.activation_bytes == expected_num_activation_params * 2

        # Block flops: B*T*(8*D^2 + 4*T*D + 6*D*D_ff)
        # LM Head: 2*D*V * B*T
        # Total: B*T * (L * (8*D^2 + 4*T*D + 6*D*D_ff) + 2*D*V)
        D, V, L, H = (
            config.d_model,
            config.vocab_size,
            config.num_layers,
            config.num_heads,
        )
        flops_per_pass = (
            B * T * (L * (8 * D**2 + 4 * T * D + 6 * D * config.d_ff) + 2 * D * V)
        )
        assert results.flops == flops_per_pass * (3 if is_training else 1)
