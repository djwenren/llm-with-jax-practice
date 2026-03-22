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

from llm_with_jax_practice import train_config as _train_config
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


class BaseCheckpointManager:
    """Base checkpoint manager."""

    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        max_to_keep: int = 3,
        save_interval_steps: int = 2,
        *,
        train_config: _train_config.TrainConfig | None = None,
        model_config: transformer.TransformerConfig | None = None,
    ):
        self._checkpoint_dir = checkpoint_dir
        self._ocp_checkpoint_manager_options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        self._ocp_checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=self._ocp_checkpoint_manager_options,
            item_names=self.get_item_names(),
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
        metadata: PyTree[Any],
        **kwargs,
    ) -> None:
        """Saves the checkpoint."""
        raise NotImplementedError("Subclasses must implement this method.")

    def restore(
        self,
        step: int,
        abstract_model: nnx.Module,
        **kwargs,
    ) -> tuple[nnx.Module, PyTree[Any], ...]:
        """Restores the checkpoint."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_item_names(self) -> tuple[str, ...]:
        """Returns the item names for the checkpoint."""
        raise NotImplementedError("Subclasses must implement this method.")

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

    def train_config(self) -> _train_config.TrainConfig | None:
        """Returns the configuration for the checkpoint."""
        train_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "train_config"
        ]
        if train_config_dict is None:
            return None
        return _train_config.TrainConfig(**train_config_dict)

    def model_config(self) -> transformer.TransformerConfig | None:
        """Returns the configuration for the checkpoint."""
        model_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "model_config"
        ]
        if model_config_dict is None:
            return None
        return transformer.TransformerConfig(**model_config_dict)

    @property
    def checkpoint_dir(self) -> os.PathLike:
        """Returns the checkpoint directory."""
        return self._checkpoint_dir


class CheckpointManager(BaseCheckpointManager):
    """Checkpoint manager for Transformer language model."""

    def get_item_names(self) -> tuple[str, ...]:
        """Returns the item names for the checkpoint."""
        return ("model_state", "optimizer_state", "metadata")

    def save(
        self,
        step: int,
        model: nnx.Module,
        metadata: PyTree[Any],
        **kwargs,
    ) -> None:
        """Saves the checkpoint."""
        assert "optimizer" in kwargs, "optimizer must be provided"
        optimizer = kwargs["optimizer"]
        assert isinstance(
            optimizer, nnx.Optimizer
        ), "optimizer must be an instance of nnx.Optimizer"

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
        self,
        step: int,
        abstract_model: nnx.Module,
        **kwargs,
    ) -> tuple[nnx.Module, PyTree[Any], ...]:
        """Restores the checkpoint."""
        assert "tx" in kwargs, "tx must be provided"
        tx = kwargs["tx"]
        assert isinstance(
            tx, optax.GradientTransformation
        ), "tx must be an instance of optax.GradientTransformation"

        # 1. Create abstract optimizer on top of abstract model.
        # Since abstract_model contains ShapeDtypeStructs, no real arrays are allocated here.
        abstract_optimizer = nnx.Optimizer(abstract_model, tx, wrt=nnx.Param)

        # 2. Split both together to get a unified GraphDef and combined abstract state.
        # This allows us to restore both in one merge call, ensuring correct linking.
        # Path 0: optimizer state, Path 1: model state.
        opt_model_graph_def, abstract_combined_state = nnx.split(
            (abstract_optimizer, abstract_model)
        )
        abstract_combined_state = _canonicalize_sharding(abstract_combined_state)

        # 3. Restore using fixed shardings from their respective checkpoint slots.
        restored_args = self._ocp_checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_combined_state[1]),
                optimizer_state=ocp.args.StandardRestore(abstract_combined_state[0]),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        # 4. Merge everything back into real objects in one go.
        # This bypasses optax.init() and prevents materializing zero-filled states.
        full_restored_state = nnx.State(
            {0: restored_args.optimizer_state, 1: restored_args.model_state}
        )
        restored_optimizer, restored_model = nnx.merge(
            opt_model_graph_def, full_restored_state
        )

        return restored_model, restored_args.metadata, restored_optimizer

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

    def train_config(self) -> _train_config.TrainConfig | None:
        """Returns the configuration for the checkpoint."""
        train_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "train_config"
        ]
        if train_config_dict is None:
            return None
        return _train_config.TrainConfig(**train_config_dict)

    def model_config(self) -> transformer.TransformerConfig | None:
        """Returns the configuration for the checkpoint."""
        model_config_dict = self._ocp_checkpoint_manager.metadata().custom_metadata[
            "model_config"
        ]
        if model_config_dict is None:
            return None
        return transformer.TransformerConfig(**model_config_dict)


class MuPCheckpointManager(BaseCheckpointManager):
    """Checkpoint manager for mu-p."""

    def get_item_names(self) -> tuple[str, ...]:
        """Returns the item names for the checkpoint."""
        return (
            "model_state",
            "embedding_optimizer_state",
            "block_and_output_optimizer_state",
            "metadata",
        )

    def save(
        self,
        step: int,
        model: nnx.Module,
        metadata: PyTree[Any],
        **kwargs,
    ) -> None:
        """Saves the checkpoint.

        Args:
            step: The step to save at.
            model: The model to save.
            metadata: The metadata to save.
            embedding_optimizer: The embedding optimizer to save.
            block_and_output_optimizer: The block and output optimizer to save.
        """
        assert "embedding_optimizer" in kwargs, "embedding_optimizer must be provided"
        embedding_optimizer = kwargs["embedding_optimizer"]
        assert isinstance(
            embedding_optimizer, nnx.Optimizer
        ), "embedding_optimizer must be an instance of nnx.Optimizer"
        assert (
            "block_and_output_optimizer" in kwargs
        ), "block_and_output_optimizer must be provided"
        block_and_output_optimizer = kwargs["block_and_output_optimizer"]
        assert isinstance(
            block_and_output_optimizer, nnx.Optimizer
        ), "block_and_output_optimizer must be an instance of nnx.Optimizer"

        _, model_state = nnx.split(model)
        _, embedding_optimizer_state = nnx.split(embedding_optimizer)
        _, block_and_output_optimizer_state = nnx.split(block_and_output_optimizer)
        self._ocp_checkpoint_manager.save(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardSave(model_state),
                embedding_optimizer_state=ocp.args.StandardSave(
                    embedding_optimizer_state
                ),
                block_and_output_optimizer_state=ocp.args.StandardSave(
                    block_and_output_optimizer_state
                ),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )

    def restore(
        self,
        step: int,
        abstract_model: nnx.Module,
        **kwargs,
    ) -> tuple[nnx.Module, PyTree[Any], ...]:
        """Restores the checkpoint.

        Args:
            step: The step to restore from.
            abstract_model: The abstract model to restore into.
            embedding_tx: The embedding gradient transformation.
            block_and_output_tx: The block and output gradient transformation.
            embedding_params_filter: The embedding parameters filter.
            block_and_output_params_filter: The block and output parameters filter.
        Returns:
            A tuple of (model, metadata, embedding_optimizer, block_and_output_optimizer).
        """
        assert "embedding_tx" in kwargs, "embedding_tx must be provided"
        embedding_tx = kwargs["embedding_tx"]
        assert isinstance(
            embedding_tx, optax.GradientTransformation
        ), "embedding_tx must be an instance of optax.GradientTransformation"
        assert "block_and_output_tx" in kwargs, "block_and_output_tx must be provided"
        block_and_output_tx = kwargs["block_and_output_tx"]
        assert isinstance(
            block_and_output_tx, optax.GradientTransformation
        ), "block_and_output_tx must be an instance of optax.GradientTransformation"

        assert (
            "embedding_params_filter" in kwargs
        ), "embedding_params_filter must be provided"
        embedding_params_filter = kwargs["embedding_params_filter"]
        assert (
            "block_and_output_params_filter" in kwargs
        ), "block_and_output_params_filter must be provided"
        block_and_output_params_filter = kwargs["block_and_output_params_filter"]

        # 1. Create abstract optimizers on top of abstract model.
        # Since abstract_model contains ShapeDtypeStructs, no real arrays are allocated here.
        abstract_embedding_optimizer = nnx.Optimizer(
            abstract_model, embedding_tx, wrt=embedding_params_filter
        )
        abstract_block_and_output_optimizer = nnx.Optimizer(
            abstract_model, block_and_output_tx, wrt=block_and_output_params_filter
        )

        # 2. Split both together to get a unified GraphDef and combined abstract state.
        # This allows us to restore both in one merge call, ensuring correct linking.
        # Path 0: embedding optimizer state, Path 1: block and output optimizer state, Path 2: model state.
        opt_model_graph_def, abstract_combined_state = nnx.split(
            (
                abstract_embedding_optimizer,
                abstract_block_and_output_optimizer,
                abstract_model,
            )
        )
        abstract_combined_state = _canonicalize_sharding(abstract_combined_state)

        # 3. Restore using fixed shardings from their respective checkpoint slots.
        restored_args = self._ocp_checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_combined_state[2]),
                embedding_optimizer_state=ocp.args.StandardRestore(
                    abstract_combined_state[0]
                ),
                block_and_output_optimizer_state=ocp.args.StandardRestore(
                    abstract_combined_state[1]
                ),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        # 4. Merge everything back into real objects in one go.
        # This bypasses optax.init() and prevents materializing zero-filled states.
        full_restored_state = nnx.State(
            {
                0: restored_args.embedding_optimizer_state,
                1: restored_args.block_and_output_optimizer_state,
                2: restored_args.model_state,
            }
        )
        (
            restored_embedding_optimizer,
            restored_block_and_output_optimizer,
            restored_model,
        ) = nnx.merge(opt_model_graph_def, full_restored_state)

        return (
            restored_model,
            restored_args.metadata,
            restored_embedding_optimizer,
            restored_block_and_output_optimizer,
        )
