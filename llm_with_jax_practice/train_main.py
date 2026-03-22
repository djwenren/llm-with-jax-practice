"""Train main."""

from typing import Sequence

import dataclasses

from absl import app
from absl import flags

import jax

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
_run_xprof_profiler = flags.DEFINE_boolean(
    "run_xprof_profiler", False, "Run XProf profiler."
)
_xprof_output_filepath = flags.DEFINE_string(
    "xprof_output_filepath", "xprof", "XProf output filepath."
)


def main(argv: Sequence[str]) -> None:
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    del argv  # Unused.

    mesh, sharding = _sharding.get_mesh_and_sharding(
        sharding_strategy=_sharding_strategy.value
    )
    if mesh is not None:
        jax.set_mesh(mesh)

    train_config = _train_config.get_train_config()
    model_config = transformer.get_transformer_config(use_mu_p=train_config.use_mu_p)
    # If there is already a checkpoint directory, we use it, the train_config and model_config
    # returned from the ckpt_manager will be whatever that is already in the checkpoint directory,
    # independent on the train_config and model_config passed in. The difference in train_config
    # and model_config in the ckpt_manager and the ones loaded from command line are then reconciled
    # in _reconcile_train_config_and_model_config based on the
    # use_model_and_train_config_from_checkpoint flag.
    ckpt_manager = train_utils.get_ckpt_manager(
        checkpoint_dir=_checkpoint_dir.value,
        max_to_keep=_max_ckpts_to_keep.value,
        save_interval_steps=_ckpt_save_interval_steps.value,
        train_config=train_config,
        model_config=model_config,
    )
    train_config, model_config = train_utils.reconcile_train_config_and_model_config(
        train_config=train_config,
        model_config=model_config,
        ckpt_manager=ckpt_manager,
        use_model_and_train_config_from_checkpoint=_use_model_and_train_config_from_checkpoint.value,  # pylint: disable=line-too-long
    )
    wandb_run = train_utils.get_wandb_run(
        train_config=train_config,
        model_config=model_config,
        sharding_strategy=_sharding_strategy.value,
        wandb_entity=_wandb_entity.value,
        wandb_project=_wandb_project.value,
        wandb_run_name=_wandb_run_name.value,
    )
    training_dataset, validation_dataset = train_utils.get_datasets(
        training_data_source_path=_training_data_source_path.value,
        validation_data_source_path=_validation_data_source_path.value,
        train_config=train_config,
        model_config=model_config,
        seed=42,
    )
    if _run_xprof_profiler.value and _xprof_output_filepath.value:
        jax.profiler.start_trace(_xprof_output_filepath.value)
    if train_config.use_mu_p:
        train_utils.run_mu_p_training(
            train_config=train_config,
            model_config=model_config,
            sharding=sharding,
            ckpt_manager=ckpt_manager,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            wandb_run=wandb_run,
            log_train_metrics_every_n_steps=_log_train_metrics_every_n_steps.value,
            validation_every_n_steps=_validation_every_n_steps.value,
        )
    else:
        train_utils.run_sp_training(
            train_config=train_config,
            model_config=model_config,
            sharding=sharding,
            ckpt_manager=ckpt_manager,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            wandb_run=wandb_run,
            log_train_metrics_every_n_steps=_log_train_metrics_every_n_steps.value,
            validation_every_n_steps=_validation_every_n_steps.value,
        )
    if _run_xprof_profiler.value and _xprof_output_filepath.value:
        jax.profiler.stop_trace()
    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == "__main__":
    app.run(main)
