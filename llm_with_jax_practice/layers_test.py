"""Tests for layers."""

import jax

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
