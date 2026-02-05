"""Optimizer for Transformer language model."""

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from jaxtyping import PyTree


@nnx.dataclass
class AdamState(nnx.Pytree):
    """Adam state."""

    ms: PyTree[jax.Array] = nnx.data()
    vs: PyTree[jax.Array] = nnx.data()
    # This must be an array instead of a Python int. This is because the nnx.Optimizer that the
    # optimizer will be pased to will jit the update function. If the step is a Python int, then
    # it will be treated as a static field instead of a data field. All the update steps will see
    # `step` as 0. If this is a data field, it will be incremented correctly.
    step: jax.Array = nnx.data()


def get_adam_optimizer(
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Gets a AdamW optimizer."""

    def init_fn(params: PyTree[jax.Array]) -> AdamState:
        """Initializes the AdamW optimizer."""
        return AdamState(
            ms=jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params),
            vs=jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: PyTree[jax.Array],
        state: AdamState,
        params: PyTree[jax.Array] | None = None,
    ) -> tuple[PyTree[jax.Array], AdamState]:
        """Updates the AdamW optimizer."""
        del params
        step = state.step + 1
        ms = state.ms
        vs = state.vs
        beta_1, beta_2 = betas
        ms = jax.tree.map(
            lambda m_i, update_i: beta_1 * m_i + (1 - beta_1) * update_i,
            ms,
            updates,
        )
        vs = jax.tree.map(
            lambda v_i, update_i: beta_2 * v_i + (1 - beta_2) * (update_i**2),
            vs,
            updates,
        )
        alpha_t = learning_rate * (1 - beta_2**step) ** 0.5 / (1 - beta_1**step)
        output_updates = jax.tree.map(
            lambda m_i, v_i: -alpha_t * m_i / jnp.sqrt(v_i + eps),
            ms,
            vs,
        )
        return output_updates, AdamState(ms=ms, vs=vs, step=step)

    return optax.GradientTransformation(init_fn, update_fn)


def get_weight_decay_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
) -> optax.GradientTransformation:
    """Gets a weight decay optimizer."""

    def init_fn(params: PyTree[jax.Array]) -> optax.EmptyState:
        """Initializes the weight decay optimizer."""
        del params
        return optax.EmptyState()

    def update_fn(
        updates: PyTree[jax.Array],
        state: optax.EmptyState,
        params: PyTree[jax.Array] | None = None,
    ) -> tuple[PyTree[jax.Array], optax.EmptyState]:
        """Updates the weight decay optimizer."""
        output_updates = jax.tree.map(
            lambda update, param: update - learning_rate * weight_decay * param,
            updates,
            params,
        )
        return output_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def get_adamw_optimizer(
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-3,
) -> optax.GradientTransformation:
    """Gets a AdamW optimizer."""
    return optax.chain(
        get_adam_optimizer(learning_rate, betas, eps),
        get_weight_decay_optimizer(learning_rate, weight_decay),
    )
