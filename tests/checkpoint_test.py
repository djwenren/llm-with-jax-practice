"""Checkpoint tests."""

import pathlib
import tempfile
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from llm_with_jax_practice import checkpoint
from llm_with_jax_practice import optimizer


def _assert_all_close(x, y):
    assert jnp.allclose(x, y), f"Mismatch: {x} vs {y}"


def _assert_not_all_close(x, y):
    assert not jnp.allclose(x, y), f"Should not be close: {x} vs {y}"


def _loss_fn(model, x):
    return jnp.mean(model(x) ** 2)


def test_checkpoint_manager_with_new_model():
    """Test checkpoint manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = pathlib.Path(temp_dir)
        # Set save_interval_steps=1 to save every time for testing
        manager = checkpoint.CheckpointManager(checkpoint_dir, save_interval_steps=1)

        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        tx = optax.chain(
            optimizer.scale_by_adam(
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            optimizer.scale_by_learning_rate(learning_rate=1e-3),
        )
        test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        metadata = {"epoch": 1, "loss": 0.5}
        # Update model and optimizer state
        x = jax.random.normal(jax.random.key(1), (1, 2))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)
        # Save checkpoint
        manager.save(step=1, model=model, optimizer=test_optimizer, metadata=metadata)

        # Create a new model and optimizer to restore into
        new_model = nnx.Linear(2, 3, rngs=nnx.Rngs(42))
        # Restore checkpoint
        restored_model, restored_optimizer, restored_metadata = manager.restore(
            step=1, abstract_model=new_model, tx=tx
        )

        # Verify restoration
        # Compare all model parameters and buffers
        jax.tree.map(_assert_all_close, restored_model, model)
        jax.tree.map(
            _assert_all_close, restored_optimizer.opt_state, test_optimizer.opt_state
        )
        assert restored_metadata == metadata

        # After taking another step on both old and new models, they should still match.
        x = jax.random.normal(jax.random.key(2), (1, 2))
        old_grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, old_grads)
        jax.tree.map(_assert_not_all_close, model, restored_model)
        jax.tree.map(
            _assert_not_all_close,
            test_optimizer.opt_state,
            restored_optimizer.opt_state,
        )
        new_grads = nnx.grad(_loss_fn)(restored_model, x)
        restored_optimizer.update(restored_model, new_grads)
        jax.tree.map(_assert_all_close, model, restored_model)
        jax.tree.map(
            _assert_all_close, test_optimizer.opt_state, restored_optimizer.opt_state
        )


def test_checkpoint_manager_with_abstract_model():
    """Test checkpoint manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = pathlib.Path(temp_dir)
        # Set save_interval_steps=1 to save every time for testing
        manager = checkpoint.CheckpointManager(checkpoint_dir, save_interval_steps=1)

        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        tx = optax.chain(
            optimizer.scale_by_adam(
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            optimizer.scale_by_learning_rate(learning_rate=1e-3),
        )
        test_optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        metadata = {"epoch": 1, "loss": 0.5}
        # Update model and optimizer state
        x = jax.random.normal(jax.random.key(1), (1, 2))
        grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, grads)
        # Save checkpoint
        manager.save(step=1, model=model, optimizer=test_optimizer, metadata=metadata)

        # Create an abstract model to restore into
        abstract_model = nnx.eval_shape(lambda: nnx.Linear(2, 3, rngs=nnx.Rngs(42)))
        # Restore checkpoint
        restored_model, restored_optimizer, restored_metadata = manager.restore(
            step=1, abstract_model=abstract_model, tx=tx
        )

        # Verify restoration
        # Compare all model parameters and buffers
        jax.tree.map(_assert_all_close, restored_model, model)
        jax.tree.map(
            _assert_all_close, restored_optimizer.opt_state, test_optimizer.opt_state
        )
        assert restored_metadata == metadata

        # After taking another step on both old and new models, they should still match.
        x = jax.random.normal(jax.random.key(2), (1, 2))
        old_grads = nnx.grad(_loss_fn)(model, x)
        test_optimizer.update(model, old_grads)
        jax.tree.map(_assert_not_all_close, model, restored_model)
        jax.tree.map(
            _assert_not_all_close,
            test_optimizer.opt_state,
            restored_optimizer.opt_state,
        )
        new_grads = nnx.grad(_loss_fn)(restored_model, x)
        restored_optimizer.update(restored_model, new_grads)
        jax.tree.map(_assert_all_close, model, restored_model)
        jax.tree.map(
            _assert_all_close, test_optimizer.opt_state, restored_optimizer.opt_state
        )
