"""Tests for layers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

from llm_with_jax_practice import layers


def test_linear():
    """Test linear layer."""
    rngs = nnx.Rngs(jax.random.key(42))
    linear = layers.Linear(in_features=10, out_features=20, rngs=rngs)
    x = jax.random.normal(rngs.key(), (1, 10))
    y = linear(x)
    assert y.shape == (1, 20)
    assert linear.weight.shape == (20, 10)


def test_embedding():
    """Test embedding layer."""
    rngs = nnx.Rngs(jax.random.key(42))
    embedding = layers.Embedding(num_embeddings=100, embedding_dim=10, rngs=rngs)
    token_ids = jnp.array([[0, 1, 2, 3, 4, 5, 6]] * 4)
    y = embedding(token_ids)
    assert y.shape == (4, 7, 10)
    assert embedding.weight.shape == (100, 10)
    assert embedding.weight.std() == pytest.approx(1.0, abs=0.1)
    for i in range(4):
        for j in range(7):
            np.testing.assert_allclose(y[i, j, :], embedding.weight[j])


def test_rms_norm():
    """Test RMSNorm layer."""
    rms_norm = layers.RMSNorm(d_model=10)
    x = np.random.normal(0, 1, (2, 10)) * 10.0
    y = rms_norm(jnp.array(x))
    expected_y = (
        x
        / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + rms_norm.eps)
        * rms_norm.weight
    )
    assert y.shape == (2, 10)
    assert rms_norm.weight.shape == (10,)
    np.testing.assert_allclose(y, expected_y, rtol=1e-4)


def test_rope():
    """Test RoPE layer."""
    rope = layers.RoPE(theta=10000.0, d_k=4, max_seq_len=1024)
    token_positions = jnp.array([[0, 1, 2, 3, 4, 5, 6]])
    x = np.random.normal(0, 1, (1, 7, 4))
    y = rope(x, token_positions)
    assert y.shape == (1, 7, 4)
    assert rope.rope_matrix.shape == (1024, 2, 2, 2)
