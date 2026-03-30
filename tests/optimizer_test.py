"""Optimizer tests."""

import jax

jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp
import optax

from flax import nnx
from jaxtyping import Float
from jax.sharding import PartitionSpec as P

from llm_with_jax_practice import sharding as _sharding
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
    for _ in range(50):
        sample_rng, sample_key = jax.random.split(sample_rng)
        x = jax.random.uniform(sample_key, (in_features,))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)
    return model.weight


def test_adam_optimizer():
    """Test Adam optimizer."""
    my_adam_optimizer = optax.chain(
        optimizer.scale_by_adam(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        optimizer.scale_by_learning_rate(learning_rate=1e-3),
    )
    final_weight = _run_optimizer(my_adam_optimizer)

    reference_adam_optimizer = optax.adam(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )
    final_weight_reference = _run_optimizer(reference_adam_optimizer)

    assert jnp.allclose(final_weight, final_weight_reference, atol=1e-5, rtol=1e-4)


def test_adamw_optimizer():
    """Test AdamW optimizer."""
    my_adamw_optimizer = optax.chain(
        optimizer.scale_by_adamw(
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-3,
        ),
        optimizer.scale_by_learning_rate(learning_rate=1e-3),
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

    assert jnp.allclose(final_weight, final_weight_reference, atol=1e-5, rtol=1e-4)


def test_adamw_with_cosine_onecycle_schedule():
    """Test AdamW optimizer with cosine onecycle schedule."""
    my_adamw_optimizer = optax.chain(
        optimizer.scale_by_adamw(
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-3,
        ),
        optimizer.scale_by_schedule(
            optimizer.cosine_onecycle_schedule(
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_iters=50,
                cosine_cycle_iters=50,
            )
        ),
    )
    final_weight = _run_optimizer(my_adamw_optimizer)

    reference_adamw_optimizer = optax.chain(
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-3,
                warmup_steps=50,
                decay_steps=100,
                end_value=1e-4,
                exponent=1.0,
            ),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-3,
        ),
    )
    final_weight_reference = _run_optimizer(reference_adamw_optimizer)

    assert jnp.allclose(final_weight, final_weight_reference, atol=1e-5, rtol=1e-4)


def test_adam_state_sharding_and_dtype():
    """Test that Adam state has the same sharding and dtype as model parameters."""
    mesh = jax.make_mesh((4, 2), ("data", "model"))

    class TestModel(nnx.Module):
        """Test model."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            rngs: nnx.Rngs,
            *,
            dtype: jnp.dtype = jnp.float32,
            linear1_sharding: _sharding.LinearSharding = _sharding.LinearSharding(),
            linear2_sharding: _sharding.LinearSharding = _sharding.LinearSharding(),
        ):
            self.linear1 = layers.Linear(
                in_features,
                hidden_features,
                rngs,
                dtype=dtype,
                sharding=linear1_sharding,
            )
            self.linear2 = layers.Linear(
                hidden_features,
                out_features,
                rngs,
                dtype=dtype,
                sharding=linear2_sharding,
            )

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.linear2(self.linear1(x))

    with jax.set_mesh(mesh):
        in_features = 8
        hidden_features = 32
        out_features = 16
        dtype = jnp.float16

        model = TestModel(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            rngs=nnx.Rngs(jax.random.key(0)),
            dtype=dtype,
            linear1_sharding=_sharding.LinearSharding(weight=P("data", "model")),
            linear2_sharding=_sharding.LinearSharding(weight=P("model", "data")),
        )

        tx = optimizer.scale_by_adam()
        test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # Access the optax state. In NNX, it's stored in the optimizer object as opt_state.
        # Since it's a scale_by_adam transformation, the state is an AdamState.
        adam_state = test_optimizer.opt_state
        assert isinstance(adam_state, optimizer.AdamState)

        # Get the parameters to compare against.
        _, params, _ = nnx.split(model, nnx.Param, ...)

        # Extract leaves (the raw JAX arrays) from the states.
        # This avoids type mismatch between Param and OptVariable nodes.
        params_leaves = jax.tree.leaves(params)
        ms_leaves = jax.tree.leaves(adam_state.ms)
        vs_leaves = jax.tree.leaves(adam_state.vs)

        assert len(params_leaves) == len(ms_leaves)
        assert len(params_leaves) == len(vs_leaves)

        for p, m, v in zip(params_leaves, ms_leaves, vs_leaves):
            assert p.dtype == m.dtype
            assert p.sharding.spec == m.sharding.spec
            assert p.dtype == v.dtype
            assert p.sharding.spec == v.sharding.spec
