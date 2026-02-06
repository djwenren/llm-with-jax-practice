"""Tests for layers."""

import jax
import jax.numpy as jnp
import pytest
import einops

from flax import nnx

from llm_with_jax_practice import layers


@pytest.mark.parametrize("use_jit", [False, True])
class TestLayers:
    """Tests for layers."""

    def test_linear(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test linear layer reference implementation."""
        w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
        linear = layers.Linear(
            in_features=d_model, out_features=d_ff, rngs=nnx.Rngs(jax.random.key(42))
        )
        linear.weight = jnp.array(w1_weight)

        call = nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)

        y = call(linear, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_linear")

    def test_embedding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model
    ):
        """Test embedding layer."""
        embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
        embedding = layers.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            rngs=nnx.Rngs(jax.random.key(42)),
        )
        embedding.weight = jnp.array(embedding_weight)

        call = nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)

        y = call(embedding, jnp.array(in_indices))
        numpy_snapshot.assert_match(y, test_name="test_embedding")

    def test_rmsnorm(self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings):
        """Test RMSNorm layer."""
        state_dict, _ = ts_state_dict
        reference_weights = state_dict["layers.1.ln1.weight"]
        d_model = reference_weights.shape[0]
        rms_norm = layers.RMSNorm(d_model=d_model, eps=1e-5)
        rms_norm.weight = jnp.array(reference_weights)

        call = nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)

        y = call(rms_norm, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_rmsnorm")

    def test_swiglu(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test SwiGLU layer."""
        w1_weight, w2_weight, w3_weight = [
            ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]
        ]
        swiglu = layers.SwiGLU(
            d_model=d_model, d_ff=d_ff, rngs=nnx.Rngs(jax.random.key(42))
        )
        swiglu.in_project_layer_1.weight = jnp.array(w1_weight)
        swiglu.in_project_layer_3.weight = jnp.array(w3_weight)
        swiglu.out_project_layer_2.weight = jnp.array(w2_weight)

        call = nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)

        y = call(swiglu, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_swiglu")

    def test_rope(
        self, use_jit, numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids
    ):
        """Test RoPE layer."""
        rope = layers.RoPE(theta=theta, d_k=d_model, max_seq_len=n_queries)

        call = (
            nnx.jit(lambda model, x, p: model(x, p))
            if use_jit
            else lambda model, x, p: model(x, p)
        )

        y = call(rope, jnp.array(in_embeddings), jnp.array(pos_ids))
        numpy_snapshot.assert_match(y, test_name="test_rope")

    def test_multihead_self_attention(
        self, use_jit, numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict
    ):
        """Test Multi-head self-attention layer."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        multi_head_self_attention = layers.MultiHeadSelfAttention(
            d_model=d_model, num_heads=n_heads, rngs=nnx.Rngs(jax.random.key(42))
        )
        multi_head_self_attention.combined_in_projection.weight = jnp.concatenate(
            [jnp.array(q_proj_weight), jnp.array(k_proj_weight), jnp.array(v_proj_weight)],
            axis=0,
        )
        multi_head_self_attention.out_projection.weight = jnp.array(o_proj_weight)

        call = nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)

        y = call(multi_head_self_attention, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_multihead_self_attention")

    def test_multihead_self_attention_with_rope(
        self,
        use_jit,
        numpy_snapshot,
        in_embeddings,
        d_model,
        n_heads,
        ts_state_dict,
        n_keys,
        theta,
        pos_ids,
    ):
        """Test Multi-head self-attention layer with RoPE."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        pos_ids = einops.rearrange(jnp.array(pos_ids), "seq -> 1 seq")
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        multi_head_self_attention = layers.MultiHeadSelfAttention(
            d_model=d_model, num_heads=n_heads, rngs=nnx.Rngs(jax.random.key(42)), rope=rope
        )
        multi_head_self_attention.combined_in_projection.weight = jnp.concatenate(
            [jnp.array(q_proj_weight), jnp.array(k_proj_weight), jnp.array(v_proj_weight)],
            axis=0,
        )
        multi_head_self_attention.out_projection.weight = jnp.array(o_proj_weight)

        call = (
            nnx.jit(lambda model, x, token_positions: model(x, token_positions=token_positions))
            if use_jit
            else lambda model, x, token_positions: model(x, token_positions=token_positions)
        )

        y = call(
            multi_head_self_attention,
            jnp.array(in_embeddings),
            token_positions=jnp.array(pos_ids),
        )
        numpy_snapshot.assert_match(y, test_name="test_multihead_self_attention_with_rope")

    def test_transformer_block(
        self,
        use_jit,
        numpy_snapshot,
        ts_state_dict,
        in_embeddings,
        d_model,
        n_heads,
        d_ff,
        n_keys,
        theta,
    ):
        """Test Transformer block."""
        block_weights = {
            k.replace("layers.0.", ""): v
            for k, v in ts_state_dict[0].items()
            if "layers.0." in k
        }
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        transformer_block = layers.TransformerBlock(
            d_model=d_model,
            num_heads=n_heads,
            d_ff=d_ff,
            rngs=nnx.Rngs(jax.random.key(42)),
            rope=rope,
        )
        transformer_block.rms_norm_pre_attn.weight = nnx.Param(
            jnp.array(block_weights["ln1.weight"])
        )
        transformer_block.attn.combined_in_projection.weight = nnx.Param(
            jnp.concatenate(
                [
                    jnp.array(block_weights["attn.q_proj.weight"]),
                    jnp.array(block_weights["attn.k_proj.weight"]),
                    jnp.array(block_weights["attn.v_proj.weight"]),
                ],
                axis=0,
            )
        )
        transformer_block.attn.out_projection.weight = nnx.Param(
            jnp.array(block_weights["attn.output_proj.weight"])
        )

        transformer_block.rms_norm_pre_ff.weight = nnx.Param(
            jnp.array(block_weights["ln2.weight"])
        )
        transformer_block.ffn.in_project_layer_1.weight = nnx.Param(
            jnp.array(block_weights["ffn.w1.weight"])
        )
        transformer_block.ffn.in_project_layer_3.weight = nnx.Param(
            jnp.array(block_weights["ffn.w3.weight"])
        )
        transformer_block.ffn.out_project_layer_2.weight = nnx.Param(
            jnp.array(block_weights["ffn.w2.weight"])
        )

        call = (
            nnx.jit(lambda model, x, token_positions: model(in_features=x, token_positions=token_positions))
            if use_jit
            else lambda model, x, token_positions: model(in_features=x, token_positions=token_positions)
        )

        y = call(
            transformer_block,
            jnp.array(in_embeddings),
            token_positions=jnp.arange(in_embeddings.shape[-2]),
        )
        numpy_snapshot.assert_match(y, test_name="test_transformer_block")
