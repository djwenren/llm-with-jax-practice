"""Checkpoint for Transformer language model."""

import pathlib

from typing import Any

import optax
import orbax.checkpoint as ocp

from flax import nnx
from jaxtyping import PyTree


class CheckpointManager:
    """Checkpoint manager for Transformer language model."""

    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        max_to_keep: int = 3,
        save_interval_steps: int = 2,
    ):
        self._ocp_checkpoint_manager_options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        self._ocp_checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=self._ocp_checkpoint_manager_options,
            item_names=("model_state", "optimizer_state", "metadata"),
        )

    def save(
        self,
        step: int,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metadata: PyTree[Any],
    ) -> None:
        """Saves the checkpoint."""
        _, model_state = nnx.split(model)
        _, optimizer_state = nnx.split(optimizer)
        self._ocp_checkpoint_manager.save(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardSave(model_state),
                optimizer_state=ocp.args.StandardSave(optimizer_state),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )

    def restore(
        self, step: int, abstract_model: nnx.Module, tx: optax.GradientTransformation
    ) -> tuple[nnx.Module, nnx.Optimizer, PyTree[Any]]:
        """Restores the checkpoint."""
        graph_def, abstract_model_state = nnx.split(abstract_model)
        abstract_optimizer = nnx.Optimizer(abstract_model, tx, wrt=nnx.Param)
        _, abstract_optimizer_state = nnx.split(abstract_optimizer)
        restored_args = self._ocp_checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_model_state),
                optimizer_state=ocp.args.StandardRestore(abstract_optimizer_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        restored_model = nnx.merge(graph_def, restored_args.model_state)
        restored_optimizer = nnx.Optimizer(restored_model, tx, wrt=nnx.Param)
        nnx.update(restored_optimizer, restored_args.optimizer_state)
        return restored_model, restored_optimizer, restored_args.metadata
