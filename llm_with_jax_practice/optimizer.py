"""Optimizer for Transformer language model."""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from jaxtyping import Int
from jaxtyping import Float
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


def scale_by_adam(
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
        alpha_t = (1 - beta_2**step) ** 0.5 / (1 - beta_1**step)
        output_updates = jax.tree.map(
            lambda m_i, v_i: alpha_t * m_i / jnp.sqrt(v_i + eps),
            ms,
            vs,
        )
        return output_updates, AdamState(ms=ms, vs=vs, step=step)

    return optax.GradientTransformation(init_fn, update_fn)


def get_weight_decay_optimizer(
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
            lambda update, param: update + weight_decay * param,
            updates,
            params,
        )
        return output_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_adamw(
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-3,
) -> optax.GradientTransformation:
    """Gets a AdamW optimizer."""
    return optax.chain(
        scale_by_adam(betas, eps),
        get_weight_decay_optimizer(weight_decay),
    )


def scale_by_learning_rate(
    learning_rate: float, minimize: bool = True
) -> optax.GradientTransformation:
    """Scales the updates by the learning rate."""

    def init_fn(params: PyTree[jax.Array]) -> optax.EmptyState:
        """Initializes the learning rate optimizer."""
        del params
        return optax.EmptyState()

    def update_fn(
        updates: PyTree[jax.Array],
        state: optax.EmptyState,
        params: PyTree[jax.Array] | None = None,
    ) -> tuple[PyTree[jax.Array], optax.EmptyState]:
        """Updates the learning rate optimizer."""
        del params
        output_updates = jax.tree.map(
            lambda update: (
                -learning_rate * update if minimize else learning_rate * update
            ),
            updates,
        )
        return output_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


@nnx.dataclass
class ScaleByScheduleState(nnx.Pytree):
    """Scale by schedule state."""

    step: jax.Array = nnx.data()


def scale_by_schedule(
    schedule: Callable[[Int[jax.Array, ""]], Float[jax.Array, ""]],
    minimize: bool = True,
) -> optax.GradientTransformation:
    """Scales the updates by the schedule."""

    def init_fn(params: PyTree[jax.Array]) -> ScaleByScheduleState:
        """Initializes the scale by schedule optimizer."""
        del params
        return ScaleByScheduleState(step=jnp.array(0, dtype=jnp.int32))

    def update_fn(
        updates: PyTree[jax.Array],
        state: ScaleByScheduleState,
        params: PyTree[jax.Array] | None = None,
    ) -> tuple[PyTree[jax.Array], ScaleByScheduleState]:
        """Updates the scale by schedule optimizer."""
        del params
        step = state.step
        output_updates = jax.tree.map(
            lambda update: (
                -schedule(step) * update if minimize else schedule(step) * update
            ),
            updates,
        )
        return output_updates, ScaleByScheduleState(step=step + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def cosine_onecycle_schedule(
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> Callable[[Int[jax.Array, ""]], Float[jax.Array, ""]]:
    """Cosine onecycle schedule."""

    def schedule(step: Int[jax.Array, ""]) -> Float[jax.Array, ""]:
        """Cosine onecycle schedule."""
        return jnp.where(
            step < warmup_iters,
            max_learning_rate * step / warmup_iters,
            jnp.where(
                step <= cosine_cycle_iters + warmup_iters,
                min_learning_rate
                + (max_learning_rate - min_learning_rate)
                * (jnp.cos(jnp.pi * (step - warmup_iters) / cosine_cycle_iters) + 1)
                / 2,
                min_learning_rate,
            ),
        )

    return schedule
