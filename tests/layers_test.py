"""Tests for layers."""

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from llm_with_jax_practice import layers


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    """Test linear layer reference implementation."""
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    linear = layers.Linear(
        in_features=d_model, out_features=d_ff, rngs=nnx.Rngs(jax.random.key(42))
    )
    linear.weight = w1_weight
    y = linear(in_embeddings)
    numpy_snapshot.assert_match(y)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    """Test embedding layer."""
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    embedding = layers.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        rngs=nnx.Rngs(jax.random.key(42)),
    )
    embedding.weight = embedding_weight
    y = embedding(in_indices)
    numpy_snapshot.assert_match(y)


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    """Test RMSNorm layer."""
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]
    rms_norm = layers.RMSNorm(d_model=d_model, eps=1e-5)
    rms_norm.weight = jnp.array(reference_weights)
    y = rms_norm(jnp.array(in_embeddings))
    numpy_snapshot.assert_match(y)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
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
    y = swiglu(jnp.array(in_embeddings))
    numpy_snapshot.assert_match(y)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    """Test RoPE layer."""
    rope = layers.RoPE(theta=theta, d_k=d_model, max_seq_len=n_queries)
    y = rope(jnp.array(in_embeddings), jnp.array(pos_ids))
    numpy_snapshot.assert_match(y)
