"""Train main."""

import dataclasses

from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import jax

jax.config.update("jax_num_cpu_devices", 8)

import optax
import wandb

from flax import nnx

from llm_with_jax_practice import checkpoint
from llm_with_jax_practice import optimizer as _optimizer
from llm_with_jax_practice import sharding as _sharding
from llm_with_jax_practice import train_config as _train_config
from llm_with_jax_practice import train_utils
from llm_with_jax_practice import transformer


_checkpoint_dir = flags.DEFINE_string(
    "checkpoint_dir", "checkpoints", "Checkpoint directory."
)
_max_ckpts_to_keep = flags.DEFINE_integer(
    "max_ckpts_to_keep", 4, "Maximum number of checkpoints to keep."
)
_ckpt_save_interval_steps = flags.DEFINE_integer(
    "ckpt_save_interval_steps", 10, "Checkpoint save interval steps."
)
_training_data_source_path = flags.DEFINE_string(
    "training_data_source_path",
    "",
    "Training data source path. The data is in the format of numpy array of tokens.",
)
_validation_data_source_path = flags.DEFINE_string(
    "validation_data_source_path",
    "",
    "Validation data source path. The data is in the format of numpy array of tokens.",
)
_use_model_and_train_config_from_checkpoint = flags.DEFINE_boolean(
    "use_model_and_train_config_from_checkpoint",
    False,
    "Use model and train config from checkpoint.",
)
_wandb_entity = flags.DEFINE_string("wandb_entity", "transformer-lm", "Wandb entity.")
_wandb_project = flags.DEFINE_string(
    "wandb_project", "transformer-lm", "Wandb project."
)
_wandb_run_name = flags.DEFINE_string(
    "wandb_run_name", "transformer-lm", "Wandb run name."
)
_log_train_metrics_every_n_steps = flags.DEFINE_integer(
    "log_train_metrics_every_n_steps", 10, "Log train metrics every n steps."
)
_validation_every_n_steps = flags.DEFINE_integer(
    "validation_every_n_steps", 10, "Validation every n steps."
)
_sharding_strategy = flags.DEFINE_enum(
    "sharding_strategy", "none", ["none", "fsdp_tp"], "Sharding strategy."
)


def _get_mesh_and_sharding(
    sharding_strategy: str,
) -> tuple[jax.sharding.Mesh | None, _sharding.TransformerLmSharding]:
    """Gets the mesh and sharding."""
    if sharding_strategy == "fsdp_tp":
        logging.info("Setting mesh for FSDP + TP sharding.")
        mesh = jax.make_mesh((4, 2), ("data", "model"))
        return mesh, _sharding.FSDP_TP_SHARDING
    return None, _sharding.TransformerLmSharding()


def _reconcile_train_config_and_model_config(
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    ckpt_manager: checkpoint.CheckpointManager,
    use_model_and_train_config_from_checkpoint: bool,
) -> tuple[_train_config.TrainConfig, transformer.TransformerConfig]:
    """Reconciles the train config and model config."""
    if use_model_and_train_config_from_checkpoint:
        assert (
            ckpt_manager.train_config() is not None
        ), f"No train config found in checkpoint {ckpt_manager.checkpoint_dir}"
        assert (
            ckpt_manager.model_config() is not None
        ), f"No model config found in checkpoint {ckpt_manager.checkpoint_dir}"
        return ckpt_manager.train_config(), ckpt_manager.model_config()
    return train_config, model_config


def _get_model_and_optimizer(
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding: _sharding.TransformerLmSharding,
    ckpt_manager: checkpoint.CheckpointManager,
) -> tuple[nnx.Module, nnx.Optimizer]:
    """Gets the model and optimizer."""
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.max_total_gradient_l2_norm),
        _optimizer.scale_by_adamw(
            betas=(train_config.adamw_beta_1, train_config.adamw_beta_2),
            eps=train_config.adamw_eps,
            weight_decay=train_config.adamw_weight_decay,
        ),
        _optimizer.scale_by_schedule(
            _optimizer.cosine_onecycle_schedule(
                max_learning_rate=train_config.cosine_onecycle_max_learning_rate,
                min_learning_rate=train_config.cosine_onecycle_min_learning_rate,
                warmup_iters=train_config.cosine_onecycle_warmup_iters,
                cosine_cycle_iters=train_config.cosine_onecycle_cosine_cycle_iters,
            )
        ),
    )
    latest_step = ckpt_manager.latest_step()

    @nnx.jit
    def _get_fresh_model_and_optimizer():
        model = transformer.TransformerLm(
            config=model_config, rngs=nnx.Rngs(jax.random.key(42)), sharding=sharding
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return model, optimizer

    if latest_step is None:
        return _get_fresh_model_and_optimizer()

    # Restoration involves complex file IO, so we do it outside of JIT.
    abstract_model = nnx.eval_shape(
        lambda: transformer.TransformerLm(
            config=model_config, rngs=nnx.Rngs(jax.random.key(42)), sharding=sharding
        )
    )
    model, optimizer, _ = ckpt_manager.restore(
        step=latest_step,
        abstract_model=abstract_model,
        tx=tx,
    )
    return model, optimizer


def _get_wandb_run(
    *,
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding_strategy: str,
    wandb_entity: str,
    wandb_project: str,
    wandb_run_name: str,
) -> wandb.Run:
    """Gets the wandb run."""
    return wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_run_name,
        config=dataclasses.asdict(train_config)
        | dataclasses.asdict(model_config)
        | {"sharding_strategy": sharding_strategy},
    )


def main(argv: Sequence[str]) -> None:
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    del argv  # Unused.

    mesh, sharding = _get_mesh_and_sharding(sharding_strategy=_sharding_strategy.value)
    if mesh is not None:
        jax.set_mesh(mesh)

    train_config = _train_config.get_train_config()
    model_config = transformer.get_transformer_config()
    ckpt_manager = checkpoint.CheckpointManager(
        checkpoint_dir=_checkpoint_dir.value,
        max_to_keep=_max_ckpts_to_keep.value,
        save_interval_steps=_ckpt_save_interval_steps.value,
        train_config=train_config,
        model_config=model_config,
    )
    train_config, model_config = _reconcile_train_config_and_model_config(
        train_config=train_config,
        model_config=model_config,
        ckpt_manager=ckpt_manager,
        use_model_and_train_config_from_checkpoint=_use_model_and_train_config_from_checkpoint.value,  # pylint: disable=line-too-long
    )
    wandb_run = _get_wandb_run(
        train_config=train_config,
        model_config=model_config,
        sharding_strategy=_sharding_strategy.value,
        wandb_entity=_wandb_entity.value,
        wandb_project=_wandb_project.value,
        wandb_run_name=_wandb_run_name.value,
    )
    logging.info("Loading model with model config: %s", model_config)
    model, optimizer = _get_model_and_optimizer(
        train_config=train_config,
        model_config=model_config,
        sharding=sharding,
        ckpt_manager=ckpt_manager,
    )
    training_dataset, validation_dataset = train_utils.get_datasets(
        training_data_source_path=_training_data_source_path.value,
        validation_data_source_path=_validation_data_source_path.value,
        train_config=train_config,
        model_config=model_config,
        seed=42,
    )
    logging.info(
        "Model and optimizer loaded. Starting training loop with train config: %s",
        train_config,
    )
    train_utils.train_loop(
        model=model,
        nnx_optimizer=optimizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        train_config=train_config,
        ckpt_manager=ckpt_manager,
        start_step=ckpt_manager.latest_step() or 0,
        wandb_run=wandb_run,
        log_train_metrics_every_n_steps=_log_train_metrics_every_n_steps.value,
        validation_every_n_steps=_validation_every_n_steps.value,
    )
    logging.info("Training loop completed.")


if __name__ == "__main__":
    app.run(main)
