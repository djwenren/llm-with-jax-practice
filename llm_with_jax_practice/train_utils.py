"""Train utilities."""

import dataclasses

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax

from absl import logging
from flax import nnx
from jaxtyping import Int
from jaxtyping import Float
from tqdm import tqdm

import wandb

from llm_with_jax_practice import checkpoint
from llm_with_jax_practice import data_loader
from llm_with_jax_practice import functions
from llm_with_jax_practice import optimizer as _optimizer
from llm_with_jax_practice import sharding as _sharding
from llm_with_jax_practice import train_config as _train_config
from llm_with_jax_practice import transformer


def get_datasets(
    training_data_source_path: str,
    validation_data_source_path: str,
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    seed: int,
) -> tuple[grain.IterDataset, grain.IterDataset]:
    """Gets the training and validation datasets."""
    training_token_data = np.load(training_data_source_path, mmap_mode="r")
    validation_token_data = np.load(validation_data_source_path, mmap_mode="r")
    training_dataset = data_loader.get_dataset(
        np_data=training_token_data,
        context_length=model_config.context_length,
        batch_size=train_config.training_batch_size,
        shuffle=True,
        seed=seed,
        use_repeat=True,
        num_repeats=None,
        num_workers=jax.local_device_count(),
        prefetch_size=2,
    )
    validation_dataset = data_loader.get_dataset(
        np_data=validation_token_data,
        context_length=model_config.context_length,
        batch_size=train_config.validation_batch_size,
        shuffle=True,
        seed=seed,
        use_repeat=True,
        num_repeats=None,
        num_workers=jax.local_device_count(),
        prefetch_size=2,
    )
    return training_dataset, validation_dataset


@nnx.jit()
def loss_fn(
    model: nnx.Module,
    input_seq: Int[jnp.ndarray, "batch_size context_length"],
    target_seq: Int[jnp.ndarray, "batch_size context_length"],
) -> Float[jnp.ndarray, ""]:
    """Computes the loss for the model."""
    logits = model(input_seq)
    return functions.cross_entropy_loss(logits=logits, target_seq=target_seq)


@nnx.jit()
def run_validation(
    model: nnx.Module,
    input_seq: Int[jnp.ndarray, "batch_size context_length"],
    target_seq: Int[jnp.ndarray, "batch_size context_length"],
) -> tuple[Float[jnp.ndarray, ""], Float[jnp.ndarray, ""]]:
    """Runs validation."""
    loss = loss_fn(model, input_seq, target_seq)
    perplexity = jnp.exp(loss)
    return loss, perplexity


def sp_train_loop(
    model: nnx.Module,
    nnx_optimizer: nnx.Optimizer,
    train_dataset: grain.IterDataset,
    validation_dataset: grain.IterDataset,
    train_config: _train_config.TrainConfig,
    ckpt_manager: checkpoint.BaseCheckpointManager | None,
    start_step: int = 0,
    *,
    wandb_run: wandb.Run | None = None,
    log_train_metrics_every_n_steps: int = 10,
    validation_every_n_steps: int = 10,
) -> None:
    """Trains the model with standard parameterization."""
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.CheckpointManager
        ), "ckpt_manager must be an instance of CheckpointManager"

    @nnx.jit(donate_argnames=("local_model", "local_optimizer"))
    def _train_step(
        local_model: nnx.Module,
        local_optimizer: nnx.Optimizer,
        input_seq: Int[jnp.ndarray, "batch_size context_length"],
        target_seq: Int[jnp.ndarray, "batch_size context_length"],
    ) -> tuple[Float[jnp.ndarray, ""], Float[jnp.ndarray, ""]]:
        """Trains the model for one step."""
        graphdef, state = nnx.split(local_model)

        def loss_fn_pure(local_state, local_input_seq, local_target_seq):
            model = nnx.merge(graphdef, local_state)
            return loss_fn(model, local_input_seq, local_target_seq)

        value_and_grad_fn = jax.value_and_grad(loss_fn_pure)
        if train_config.num_microbatches > 1:
            value_and_grad_fn = optax.microbatching.microbatch(
                value_and_grad_fn,
                argnums=(1, 2),
                microbatch_size=train_config.training_batch_size
                // train_config.num_microbatches,
                accumulator=(
                    optax.microbatching.AccumulationType.MEAN,
                    optax.microbatching.AccumulationType.MEAN,
                ),
            )
        loss, grads = value_and_grad_fn(state, input_seq, target_seq)

        local_optimizer.update(local_model, grads)
        return (
            loss,
            # Compute the total L2 norm of the gradients.
            jnp.sqrt(
                jax.tree.reduce(
                    lambda acc, x: acc + jnp.sum(jnp.square(x)),
                    grads,
                    0,
                )
            ),
        )

    train_iter = iter(train_dataset)
    validation_iter = iter(validation_dataset)

    # Prefetch the first batch to device.
    input_seq, target_seq = next(train_iter)
    input_seq = jax.device_put(input_seq)
    target_seq = jax.device_put(target_seq)

    for step in tqdm(
        range(start_step, train_config.num_steps),
        initial=start_step,
        total=train_config.num_steps,
        desc="Training",
    ):
        loss, total_gradient_l2_norm = _train_step(
            model, nnx_optimizer, input_seq, target_seq
        )

        # Prefetch next batch asynchronously.
        try:
            input_seq, target_seq = next(train_iter)
            input_seq = jax.device_put(input_seq)
            target_seq = jax.device_put(target_seq)
        except StopIteration:
            break

        if wandb_run and (step - start_step) % log_train_metrics_every_n_steps == 0:
            wandb_run.log(
                {
                    "train/loss": loss,
                    "train/total_gradient_l2_norm": total_gradient_l2_norm,
                },
                step=step,
            )
        if (
            validation_every_n_steps
            and (step - start_step) % validation_every_n_steps == 0
        ):
            validation_input_seq, validation_target_seq = next(validation_iter)
            validation_input_seq = jax.device_put(validation_input_seq)
            validation_target_seq = jax.device_put(validation_target_seq)
            val_loss, val_perplexity = run_validation(
                model=model,
                input_seq=validation_input_seq,
                target_seq=validation_target_seq,
            )
            wandb_run.log(
                {
                    "validation/loss": val_loss,
                    "validation/perplexity": val_perplexity,
                },
                step=step,
            )
            logging.info(
                f"Step {step}: Validation loss: {val_loss}, perplexity: {val_perplexity}"
            )
        if ckpt_manager is not None:
            ckpt_manager.save(
                step=step,
                model=model,
                metadata={},
                optimizer=nnx_optimizer,
            )


def mu_p_train_loop(
    model: nnx.Module,
    embedding_optimizer: nnx.Optimizer,
    block_and_output_optimizer: nnx.Optimizer,
    train_dataset: grain.IterDataset,
    validation_dataset: grain.IterDataset,
    train_config: _train_config.TrainConfig,
    ckpt_manager: checkpoint.BaseCheckpointManager | None,
    start_step: int = 0,
    *,
    wandb_run: wandb.Run | None = None,
    log_train_metrics_every_n_steps: int = 10,
    validation_every_n_steps: int = 10,
) -> None:
    """Trains the model with mu-p parameterization."""
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.MuPCheckpointManager
        ), "ckpt_manager must be an instance of MuPCheckpointManager"

    @nnx.jit(
        donate_argnames=(
            "local_model",
            "local_embedding_optimizer",
            "local_block_and_output_optimizer",
        )
    )
    def _train_step(
        local_model: nnx.Module,
        local_embedding_optimizer: nnx.Optimizer,
        local_block_and_output_optimizer: nnx.Optimizer,
        input_seq: Int[jnp.ndarray, "batch_size context_length"],
        target_seq: Int[jnp.ndarray, "batch_size context_length"],
    ) -> tuple[Float[jnp.ndarray, ""], Float[jnp.ndarray, ""]]:
        """Trains the model for one step."""
        graphdef, state = nnx.split(local_model)

        def loss_fn_pure(local_state, local_input_seq, local_target_seq):
            model = nnx.merge(graphdef, local_state)
            return loss_fn(model, local_input_seq, local_target_seq)

        value_and_grad_fn = jax.value_and_grad(loss_fn_pure)
        if train_config.num_microbatches > 1:
            value_and_grad_fn = optax.microbatching.microbatch(
                value_and_grad_fn,
                argnums=(1, 2),
                microbatch_size=train_config.training_batch_size
                // train_config.num_microbatches,
                accumulator=(
                    optax.microbatching.AccumulationType.MEAN,
                    optax.microbatching.AccumulationType.MEAN,
                ),
            )
        loss, grads = value_and_grad_fn(state, input_seq, target_seq)
        local_embedding_optimizer.update(local_model, grads)
        local_block_and_output_optimizer.update(local_model, grads)
        return (
            loss,
            # Compute the total L2 norm of the gradients.
            jnp.sqrt(
                jax.tree.reduce(
                    lambda acc, x: acc + jnp.sum(jnp.square(x)),
                    grads,
                    0,
                )
            ),
        )

    train_iter = iter(train_dataset)
    validation_iter = iter(validation_dataset)

    # Prefetch first batch.
    input_seq, target_seq = next(train_iter)
    input_seq = jax.device_put(input_seq)
    target_seq = jax.device_put(target_seq)

    for step in tqdm(
        range(start_step, train_config.num_steps),
        initial=start_step,
        total=train_config.num_steps,
        desc="Training",
    ):
        loss, total_gradient_l2_norm = _train_step(
            model,
            embedding_optimizer,
            block_and_output_optimizer,
            input_seq,
            target_seq,
        )

        # Prefetch next batch.
        try:
            input_seq, target_seq = next(train_iter)
            input_seq = jax.device_put(input_seq)
            target_seq = jax.device_put(target_seq)
        except StopIteration:
            break

        if wandb_run and (step - start_step) % log_train_metrics_every_n_steps == 0:
            wandb_run.log(
                {
                    "train/loss": loss,
                    "train/total_gradient_l2_norm": total_gradient_l2_norm,
                },
                step=step,
            )
        if (
            validation_every_n_steps
            and (step - start_step) % validation_every_n_steps == 0
        ):
            validation_input_seq, validation_target_seq = next(validation_iter)
            validation_input_seq = jax.device_put(validation_input_seq)
            validation_target_seq = jax.device_put(validation_target_seq)
            val_loss, val_perplexity = run_validation(
                model=model,
                input_seq=validation_input_seq,
                target_seq=validation_target_seq,
            )
            wandb_run.log(
                {
                    "validation/loss": val_loss,
                    "validation/perplexity": val_perplexity,
                },
                step=step,
            )
            logging.info(
                f"Step {step}: Validation loss: {val_loss}, perplexity: {val_perplexity}"
            )
        if ckpt_manager is not None:
            ckpt_manager.save(
                step=step,
                model=model,
                metadata={},
                embedding_optimizer=embedding_optimizer,
                block_and_output_optimizer=block_and_output_optimizer,
            )


def get_ckpt_manager(
    *,
    checkpoint_dir: str,
    max_to_keep: int,
    save_interval_steps: int,
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
) -> checkpoint.BaseCheckpointManager:
    """Gets the checkpoint manager."""
    if train_config.use_mu_p:
        return checkpoint.MuPCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            train_config=train_config,
            model_config=model_config,
        )
    return checkpoint.CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        train_config=train_config,
        model_config=model_config,
    )


def reconcile_train_config_and_model_config(
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    ckpt_manager: checkpoint.BaseCheckpointManager,
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


def get_sp_model_and_optimizer(
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding: _sharding.TransformerLmSharding,
    ckpt_manager: checkpoint.BaseCheckpointManager | None = None,
) -> tuple[nnx.Module, nnx.Optimizer]:
    """Gets the model and optimizers."""
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.CheckpointManager
        ), "ckpt_manager must be an instance of CheckpointManager"
    assert (
        not train_config.use_mu_p
    ), "use_mu_p must be False to get SP model and optimizer."

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

    @nnx.jit
    def _get_fresh_model_and_optimizer():
        model = transformer.TransformerLm(
            config=model_config,
            rngs=nnx.Rngs(jax.random.key(42)),
            sharding=sharding,
            use_mu_p=False,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return model, optimizer

    if ckpt_manager is None or ckpt_manager.latest_step() is None:
        return _get_fresh_model_and_optimizer()

    # Restoration involves complex file IO, so we do it outside of JIT.
    abstract_model = nnx.eval_shape(
        lambda: transformer.TransformerLm(
            config=model_config,
            rngs=nnx.Rngs(jax.random.key(42)),
            sharding=sharding,
            use_mu_p=False,
        )
    )
    latest_step = ckpt_manager.latest_step()
    model, _, optimizer = ckpt_manager.restore(
        step=latest_step,
        abstract_model=abstract_model,
        tx=tx,
    )
    return model, optimizer


def get_mu_p_model_and_optimizer(
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding: _sharding.TransformerLmSharding,
    ckpt_manager: checkpoint.BaseCheckpointManager | None = None,
) -> tuple[nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    """Gets the mu-p model and optimizers.

    Args:
        train_config: The train configuration.
        model_config: The model configuration.
        sharding: The sharding.
        ckpt_manager: The checkpoint manager.
    Returns:
        A tuple of (model, embedding_optimizer, block_and_output_optimizer).
    """
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.MuPCheckpointManager
        ), "ckpt_manager must be an instance of MuPCheckpointManager"
    assert (
        train_config.use_mu_p
    ), "use_mu_p must be True to get mu-p model and optimizers."

    tx_common = optax.chain(
        optax.clip_by_global_norm(train_config.max_total_gradient_l2_norm),
        _optimizer.scale_by_adamw(
            betas=(train_config.adamw_beta_1, train_config.adamw_beta_2),
            eps=train_config.adamw_eps,
            weight_decay=train_config.adamw_weight_decay,
        ),
    )

    def _get_schedule(lr_multiplier=1.0):
        return _optimizer.scale_by_schedule(
            _optimizer.cosine_onecycle_schedule(
                max_learning_rate=train_config.cosine_onecycle_max_learning_rate
                * lr_multiplier,
                min_learning_rate=train_config.cosine_onecycle_min_learning_rate
                * lr_multiplier,
                warmup_iters=train_config.cosine_onecycle_warmup_iters,
                cosine_cycle_iters=train_config.cosine_onecycle_cosine_cycle_iters,
            )
        )

    tx_embedding = optax.chain(tx_common, _get_schedule(1.0))
    tx_block_and_output = optax.chain(
        tx_common, _get_schedule(1.0 / model_config.m_p)
    )

    embedding_params_filter = nnx.All(nnx.Param, nnx.PathContains("token_embeddings"))
    block_and_output_params_filter = nnx.All(
        nnx.Param,
        nnx.Not(nnx.PathContains("token_embeddings")),
    )

    @nnx.jit
    def _get_fresh_model_and_optimizers():
        model = transformer.TransformerLm(
            config=model_config,
            rngs=nnx.Rngs(jax.random.key(42)),
            sharding=sharding,
            use_mu_p=True,
        )
        embedding_optimizer = nnx.Optimizer(
            model, tx_embedding, wrt=embedding_params_filter
        )
        block_and_output_optimizer = nnx.Optimizer(
            model, tx_block_and_output, wrt=block_and_output_params_filter
        )
        return model, embedding_optimizer, block_and_output_optimizer

    if ckpt_manager is None or ckpt_manager.latest_step() is None:
        return _get_fresh_model_and_optimizers()

    latest_step = ckpt_manager.latest_step()

    # Restoration involves complex file IO, so we do it outside of JIT.
    abstract_model = nnx.eval_shape(
        lambda: transformer.TransformerLm(
            config=model_config,
            rngs=nnx.Rngs(jax.random.key(42)),
            sharding=sharding,
            use_mu_p=True,
        )
    )

    model, _, embedding_optimizer, block_and_output_optimizer = (
        ckpt_manager.restore(
            step=latest_step,
            abstract_model=abstract_model,
            embedding_tx=tx_embedding,
            block_and_output_tx=tx_block_and_output,
            embedding_params_filter=embedding_params_filter,
            block_and_output_params_filter=block_and_output_params_filter,
        )
    )
    return model, embedding_optimizer, block_and_output_optimizer


def run_sp_training(
    *,
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding: _sharding.TransformerLmSharding,
    ckpt_manager: checkpoint.BaseCheckpointManager | None,
    training_dataset: grain.IterDataset,
    validation_dataset: grain.IterDataset,
    wandb_run: wandb.Run,
    log_train_metrics_every_n_steps: int,
    validation_every_n_steps: int,
) -> None:
    """Runs SP training."""
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.CheckpointManager
        ), "ckpt_manager must be an instance of CheckpointManager"

    logging.info("Running training with standard parameterization.")
    logging.info("Loading model with model config: %s", model_config)
    model, optimizer = get_sp_model_and_optimizer(
        train_config=train_config,
        model_config=model_config,
        sharding=sharding,
        ckpt_manager=ckpt_manager,
    )
    logging.info(
        "Model and optimizer loaded. Starting training loop with train config: %s",
        train_config,
    )
    sp_train_loop(
        model=model,
        nnx_optimizer=optimizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        train_config=train_config,
        ckpt_manager=ckpt_manager,
        start_step=(
            ckpt_manager.latest_step()
            if ckpt_manager is not None and ckpt_manager.latest_step() is not None
            else 0
        ),
        wandb_run=wandb_run,
        log_train_metrics_every_n_steps=log_train_metrics_every_n_steps,
        validation_every_n_steps=validation_every_n_steps,
    )
    logging.info("Training loop completed.")


def run_mu_p_training(
    *,
    train_config: _train_config.TrainConfig,
    model_config: transformer.TransformerConfig,
    sharding: _sharding.TransformerLmSharding,
    ckpt_manager: checkpoint.BaseCheckpointManager | None,
    training_dataset: grain.IterDataset,
    validation_dataset: grain.IterDataset,
    wandb_run: wandb.Run,
    log_train_metrics_every_n_steps: int,
    validation_every_n_steps: int,
) -> None:
    """Runs MuP training."""
    if ckpt_manager is not None:
        assert isinstance(
            ckpt_manager, checkpoint.MuPCheckpointManager
        ), "ckpt_manager must be an instance of MuPCheckpointManager"

    logging.info("Running training with mu-p parameterization.")
    logging.info("Loading model with model config: %s", model_config)
    model, embedding_optimizer, block_and_output_optimizer = (
        get_mu_p_model_and_optimizer(
            train_config=train_config,
            model_config=model_config,
            sharding=sharding,
            ckpt_manager=ckpt_manager,
        )
    )
    logging.info(
        "Model, embedding optimizer, and block+output optimizer loaded. "
        "Starting training loop with train config: %s",
        train_config,
    )
    mu_p_train_loop(
        model=model,
        embedding_optimizer=embedding_optimizer,
        block_and_output_optimizer=block_and_output_optimizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        train_config=train_config,
        ckpt_manager=ckpt_manager,
        start_step=(
            ckpt_manager.latest_step()
            if ckpt_manager is not None and ckpt_manager.latest_step() is not None
            else 0
        ),
        wandb_run=wandb_run,
        log_train_metrics_every_n_steps=log_train_metrics_every_n_steps,
        validation_every_n_steps=validation_every_n_steps,
    )
    logging.info("Training loop completed.")


def get_wandb_run(
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
