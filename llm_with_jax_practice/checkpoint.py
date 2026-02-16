"""Checkpoint for Transformer language model."""

import dataclasses
import os

from typing import Any
from typing import Sequence

import jax
import optax
import orbax.checkpoint as ocp

from flax import nnx
from jaxtyping import PyTree

from llm_with_jax_practice import _train_config
from llm_with_jax_practice import transformer


def _canonicalize_sharding(tree: PyTree[Any]) -> PyTree[Any]:
    """Canonicalizes the sharding of the tree.

    When restoring a checkpoing, we typically pass in an abstract model to restore into. Such
    abstract models are typically constructed using `nnx.eval_shape` or `nnx.get_abstract_model`.
    These functions will abstract the physical meshes (CPUs, GPUs, or TPUs) into an `AbstractMesh`.
    Currently, Orbax cannot restore a checkpoint into an `AbstractMesh`, and will raise errors like:

    ```
    ValueError: _device_assignment is not implemented for `jax.sharding.AbstractMesh`
    ```

    This function canonicalizes the sharding of the tree to a `NamedSharding` with a restored
    physical mesh.

    Args:
        tree: The tree to canonicalize.

    Returns:
        The canonicalized tree.
    """
    try:
        mesh = jax.sharding.get_mesh()
    except RuntimeError:
        mesh = None

    if mesh is None or isinstance(mesh, jax.sharding.AbstractMesh):
        return tree

    def fix_sharding(x):
        if hasattr(x, "sharding") and isinstance(
            x.sharding, jax.sharding.NamedSharding
        ):
            if isinstance(x.sharding.mesh, jax.sharding.AbstractMesh):
                return jax.ShapeDtypeStruct(
                    x.shape,
                    x.dtype,
                    sharding=jax.sharding.NamedSharding(mesh, x.sharding.spec),
                )
        return x

    return jax.tree_util.tree_map(fix_sharding, tree)


class CheckpointManager:
    """Checkpoint manager for Transformer language model."""

    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        max_to_keep: int = 3,
        save_interval_steps: int = 2,
        *,
        train_config: _train_config.TrainConfig | None = None,
        model_config: transformer.TransformerConfig | None = None,
    ):
        self._ocp_checkpoint_manager_options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        self._ocp_checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=self._ocp_checkpoint_manager_options,
            item_names=("model_state", "optimizer_state", "metadata"),
            metadata={
                "train_config": (
                    None if train_config is None else dataclasses.asdict(train_config)
                ),
                "model_config": (
                    None if model_config is None else dataclasses.asdict(model_config)
                ),
            },
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
        # 1. Canonicalize abstract model shardings
        _, model_state = nnx.split(abstract_model)
        abstract_model_state = _canonicalize_sharding(model_state)
        nnx.update(abstract_model, abstract_model_state)

        # 2. Create and canonicalize abstract optimizer shardings
        abstract_optimizer = nnx.Optimizer(abstract_model, tx, wrt=nnx.Param)
        _, optimizer_state = nnx.split(abstract_optimizer)
        abstract_optimizer_state = _canonicalize_sharding(optimizer_state)

        # 3. Restore using fixed shardings
        restored_args = self._ocp_checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_model_state),
                optimizer_state=ocp.args.StandardRestore(abstract_optimizer_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        graph_def, _ = nnx.split(abstract_model)
        restored_model = nnx.merge(graph_def, restored_args.model_state)
        # We can't use the same `nnx.merge` call to restore the optimizer becasue the abstract
        # optimizer constructed above is linked to the abstract model's state (parameters). The
        # returned optimizer should be linked to the restored model's state (parameters), so we
        # need to construct a new optimizer.
        # TODO(djwenren): what is the implication on sharding and memory footprint?
        restored_optimizer = nnx.Optimizer(restored_model, tx, wrt=nnx.Param)
        nnx.update(restored_optimizer, restored_args.optimizer_state)
        return restored_model, restored_optimizer, restored_args.metadata

    def all_steps(self) -> Sequence[int]:
        """Returns all steps in the checkpoint."""
        return self._ocp_checkpoint_manager.all_steps()

    def latest_step(self) -> int | None:
        """Returns the latest step in the checkpoint."""
        return self._ocp_checkpoint_manager.latest_step()

    def wait_until_finished(self) -> None:
        """Blocks until the checkpoint is finished."""
        self._ocp_checkpoint_manager.wait_until_finished()

    def close(self) -> None:
        """Closes the checkpoint manager."""
        self._ocp_checkpoint_manager.close()

    def metadata(self, step: int | None = None) -> Any:
        """Returns the metadata for the checkpoint."""
        return self._ocp_checkpoint_manager.metadata(step=step)

    def train_config(self) -> train_config.TrainConfig | None:
        """Returns the configuration for the checkpoint."""
        train_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "train_config"
        ]
        if train_config_dict is None:
            return None
        return train_config.TrainConfig(**train_config_dict)

    def model_config(self) -> transformer.TransformerConfig | None:
        """Returns the configuration for the checkpoint."""
        model_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "model_config"
        ]
        if model_config_dict is None:
            return None
        return transformer.TransformerConfig(**model_config_dict)
