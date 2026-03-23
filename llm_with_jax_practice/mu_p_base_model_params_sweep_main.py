"""MuP base model params sweep main."""

from typing import Any
from typing import Sequence

import dataclasses

from absl import app
from absl import flags
from absl import logging

import wandb

from llm_with_jax_practice import sharding as _sharding
from llm_with_jax_practice import train_config as _train_config
from llm_with_jax_practice import train_utils
from llm_with_jax_practice import transformer

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
_wandb_entity = flags.DEFINE_string("wandb_entity", "transformer-lm", "Wandb entity.")
_wandb_project = flags.DEFINE_string(
    "wandb_project", "transformer-lm", "Wandb project."
)
_wandb_sweep_method = flags.DEFINE_enum(
    "wandb_sweep_method", "random", ["grid", "random", "bayes"], "Wandb sweep method."
)
_wandb_sweep_name = flags.DEFINE_string(
    "wandb_sweep_name", "mu-p-base-model-params-sweep", "Wandb sweep name."
)
_log_train_metrics_every_n_steps = flags.DEFINE_integer(
    "log_train_metrics_every_n_steps", 10, "Log train metrics every n steps."
)


def _override_train_config(
    train_config: _train_config.TrainConfig, wandb_config: dict[str, Any]
) -> _train_config.TrainConfig:
    assert (
        "sweep_max_learning_rate" in wandb_config
    ), "max_learning_rate must be in wandb config."
    old_train_config_dict = dataclasses.asdict(train_config)
    old_train_config_dict["cosine_onecycle_max_learning_rate"] = wandb_config[
        "sweep_max_learning_rate"
    ]
    old_train_config_dict["cosine_onecycle_min_learning_rate"] = (
        old_train_config_dict["cosine_onecycle_max_learning_rate"] * 0.1
    )
    return _train_config.TrainConfig(**old_train_config_dict)


def _override_model_config(
    model_config: transformer.TransformerConfig, wandb_config: dict[str, Any]
) -> transformer.TransformerConfig:
    assert "sweep_alpha_input" in wandb_config, "alpha_input must be in wandb config."
    assert "sweep_alpha_output" in wandb_config, "alpha_output must be in wandb config."
    assert "sweep_std_base" in wandb_config, "std_base must be in wandb config."
    old_model_config_dict = dataclasses.asdict(model_config)
    old_model_config_dict["alpha_input"] = wandb_config["sweep_alpha_input"]
    old_model_config_dict["alpha_output"] = wandb_config["sweep_alpha_output"]
    old_model_config_dict["std_base"] = wandb_config["sweep_std_base"]
    return transformer.TransformerConfig(**old_model_config_dict)


def train_main() -> None:
    """Train main function."""
    train_config = _train_config.get_train_config()
    model_config = transformer.get_transformer_config(use_mu_p=True)
    wandb_run = train_utils.get_wandb_run(
        train_config=train_config,
        model_config=model_config,
        sharding_strategy="none",
        wandb_entity=_wandb_entity.value,
        wandb_project=_wandb_project.value,
        wandb_run_name=_wandb_sweep_name.value,
    )
    train_config = _override_train_config(
        train_config=train_config, wandb_config=wandb_run.config.as_dict()
    )
    model_config = _override_model_config(
        model_config=model_config,
        wandb_config=wandb_run.config.as_dict(),
    )
    logging.info("Sweep run with config: %s", wandb.config)
    training_dataset, validation_dataset = train_utils.get_datasets(
        training_data_source_path=_training_data_source_path.value,
        validation_data_source_path=_validation_data_source_path.value,
        train_config=train_config,
        model_config=model_config,
        seed=42,
    )
    train_utils.run_mu_p_training(
        train_config=train_config,
        model_config=model_config,
        sharding=_sharding.TransformerLmSharding(),
        ckpt_manager=None,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        wandb_run=wandb_run,
        log_train_metrics_every_n_steps=_log_train_metrics_every_n_steps.value,
        validation_every_n_steps=0,
    )


def main(argv: Sequence[str]) -> None:
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    del argv  # Unused.

    sweep_config = {
        "method": _wandb_sweep_method.value,
        "name": _wandb_sweep_name.value,
        "metric": {"goal": "minimize", "name": "train/loss"},
        "parameters": {
            "sweep_max_learning_rate": {
                # https://wandb.ai/fm966hz/llm-with-jax-practice/sweeps/h22rq71j
                # "values": [2e-3, 1e-3, 5e-4],
                # https://wandb.ai/fm966hz/llm-with-jax-practice/sweeps/vy0aof8n
                # "values": [2e-3, 4e-3, 6e-3, 8e-3, 1e-2],
                # https://wandb.ai/fm966hz/llm-with-jax-practice/sweeps/1ap57na1
                "values": [8e-3],
            },
            # "sweep_alpha_input": {"values": [0.8, 1.0, 1.2]},
            # "sweep_alpha_input": {"values": [1.2, 1.4, 1.6]},
            "sweep_alpha_input": {"values": [1.6, 2.4, 2.8, 3.2]},
            # "sweep_alpha_output": {"values": [0.8, 1.0, 1.2]},
            # "sweep_alpha_output": {"values": [1.2, 1.4, 1.6]},
            "sweep_alpha_output": {"values": [1.6, 2.4, 2.8, 3.2]},
            # "sweep_std_base": {"values": [0.05, 0.01, 0.005]},
            # "sweep_std_base": {"values": [0.05, 0.075, 0.1]},
            "sweep_std_base": {"values": [0.15, 0.2, 0.25]},
        },
    }
    sweep_id = wandb.sweep(
        sweep_config, project=_wandb_project.value, entity=_wandb_entity.value
    )

    wandb.agent(sweep_id, function=train_main, count=10)


if __name__ == "__main__":
    app.run(main)
