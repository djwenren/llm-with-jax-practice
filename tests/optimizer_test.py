"""Optimizer tests."""

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from jaxtyping import Float

from llm_with_jax_practice import layers
from llm_with_jax_practice import optimizer


def _run_optimizer(tx: optax.GradientTransformation) -> jnp.ndarray:
    """Runs the optimizer."""

    def _loss_fn(
        model: nnx.Module, x: Float[jnp.ndarray, "in_features"]
    ) -> Float[jnp.ndarray, ""]:
        y = model(x)
        target = jnp.array([x[0] + x[1], -x[2]])
        return jnp.mean((y - target) ** 2)

    in_features = 3
    out_features = 2
    model = layers.Linear(
        in_features=in_features,
        out_features=out_features,
        rngs=nnx.Rngs(jax.random.key(42)),
    )
    test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    sample_rng = jax.random.key(42)
    for _ in range(100):
        sample_rng, sample_key = jax.random.split(sample_rng)
        x = jax.random.uniform(sample_key, (in_features,))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)
    return model.weight


def test_adam_optimizer():
    """Test Adam optimizer."""
    my_adam_optimizer = optimizer.get_adam_optimizer(
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    final_weight = _run_optimizer(my_adam_optimizer)

    reference_adam_optimizer = optax.adam(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )
    final_weight_reference = _run_optimizer(reference_adam_optimizer)

    assert jnp.allclose(final_weight, final_weight_reference)


def test_adamw_optimizer():
    """Test AdamW optimizer."""
    my_adamw_optimizer = optimizer.get_adamw_optimizer(
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    final_weight = _run_optimizer(my_adamw_optimizer)

    reference_adamw_optimizer = optax.adamw(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-3,
    )
    final_weight_reference = _run_optimizer(reference_adamw_optimizer)

    assert jnp.allclose(final_weight, final_weight_reference)
