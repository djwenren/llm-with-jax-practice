"""MuP optimizer tests."""

import jax
import jax.numpy as jnp
import optax

from flax import nnx

from llm_with_jax_practice import layers as L
from llm_with_jax_practice import optimizer


_IN_FEATURES = 3
_HIDDEN_FEATURES = 6
_OUT_FEATURES = 2
_NUM_STEPS = 50


class TestModel(nnx.Module):
    """Test model."""

    def __init__(self, feature_dims: list[int], rngs: nnx.Rngs):
        super().__init__()
        self.backbone = L.Linear(feature_dims[0], feature_dims[1], rngs=rngs)
        self.head = L.Linear(feature_dims[1], feature_dims[2], rngs=rngs)

    def __call__(self, x):
        return self.head(self.backbone(x))


def _loss_fn(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    target = jnp.array([x[0] + x[1], -x[2]])
    return jnp.mean((model(x) - target) ** 2)


def _run_optimizer_mu_p(
    tx_backbone: optax.GradientTransformation, tx_head: optax.GradientTransformation
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs the optimizer."""

    model = TestModel(
        feature_dims=[_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES],
        rngs=nnx.Rngs(jax.random.key(42)),
    )
    test_optimizer_backbone = nnx.Optimizer(
        model,
        tx_backbone,
        wrt=nnx.All(nnx.Param, nnx.PathContains("backbone")),
    )
    test_optimizer_head = nnx.Optimizer(
        model, tx_head, wrt=nnx.All(nnx.Param, nnx.PathContains("head"))
    )

    sample_rng = jax.random.key(42)
    for _ in range(_NUM_STEPS):
        sample_rng, sample_key = jax.random.split(sample_rng)
        x = jax.random.uniform(sample_key, (_IN_FEATURES,))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer_backbone.update(model, grads)
        test_optimizer_head.update(model, grads)

    return model.backbone.weight, model.head.weight


def _run_optimizer_s_p(
    tx: optax.GradientTransformation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs the optimizer."""
    model = TestModel(
        feature_dims=[_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES],
        rngs=nnx.Rngs(jax.random.key(42)),
    )
    test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    sample_rng = jax.random.key(42)
    for _ in range(_NUM_STEPS):
        sample_rng, sample_key = jax.random.split(sample_rng)
        x = jax.random.uniform(sample_key, (_IN_FEATURES,))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)

    return model.backbone.weight, model.head.weight


def _run_optimizer_multi_lr(
    tx_backbone: optax.GradientTransformation, tx_head: optax.GradientTransformation
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs a single optimizer with multiple learning rates using optax.multi_transform."""
    model = TestModel(
        feature_dims=[_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES],
        rngs=nnx.Rngs(jax.random.key(42)),
    )

    # Map parameters to their respective transformations
    def map_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: "backbone" if "backbone" in str(path) else "head",
            params,
        )

    tx = optax.multi_transform(
        {"backbone": tx_backbone, "head": tx_head},
        map_fn,
    )
    test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    sample_rng = jax.random.key(42)
    for _ in range(_NUM_STEPS):
        sample_rng, sample_key = jax.random.split(sample_rng)
        x = jax.random.uniform(sample_key, (_IN_FEATURES,))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)

    return model.backbone.weight, model.head.weight


def test_mu_p_optimizer_different_learning_rates():
    """Test that different learning rates for backbone and head work correctly."""
    lr_backbone = 1e-3
    lr_head = 5e-4

    tx_backbone = optax.chain(
        optimizer.scale_by_adam(betas=(0.9, 0.999), eps=1e-8),
        optimizer.scale_by_learning_rate(learning_rate=lr_backbone),
    )
    tx_head = optax.chain(
        optimizer.scale_by_adam(betas=(0.9, 0.999), eps=1e-8),
        optimizer.scale_by_learning_rate(learning_rate=lr_head),
    )

    # Run multi-optimizer setup
    final_weight_backbone, final_weight_head = _run_optimizer_mu_p(tx_backbone, tx_head)

    # Run reference multi_transform setup
    final_weight_backbone_ref, final_weight_head_ref = _run_optimizer_multi_lr(
        tx_backbone, tx_head
    )

    assert jnp.allclose(final_weight_backbone, final_weight_backbone_ref)
    assert jnp.allclose(final_weight_head, final_weight_head_ref)


def test_mu_p_optimizer_same_learning_rate():
    """Test that the mu-p optimizer with the same learning rate for the backbone and head works."""
    tx_backbone = optax.chain(
        optimizer.scale_by_adam(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        optimizer.scale_by_learning_rate(learning_rate=1e-3),
    )
    tx_head = optax.chain(
        optimizer.scale_by_adam(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        optimizer.scale_by_learning_rate(learning_rate=1e-3),
    )
    final_weight_backbone, final_weight_head = _run_optimizer_mu_p(tx_backbone, tx_head)

    tx_s_p = optax.chain(
        optimizer.scale_by_adam(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        optimizer.scale_by_learning_rate(learning_rate=1e-3),
    )
    final_weight_backbone_s_p, final_weight_head_s_p = _run_optimizer_s_p(tx_s_p)

    assert jnp.allclose(final_weight_backbone, final_weight_backbone_s_p)
    assert jnp.allclose(final_weight_head, final_weight_head_s_p)
