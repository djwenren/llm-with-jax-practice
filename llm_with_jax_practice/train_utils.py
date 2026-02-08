"""Train utilities."""

import grain
import jax
import jax.numpy as jnp
import wandb

from absl import logging
from flax import nnx
from jaxtyping import Int
from jaxtyping import Float
from tqdm import tqdm

from llm_with_jax_practice import checkpoint
from llm_with_jax_practice import functions
from llm_with_jax_practice import train_config as _train_config


@nnx.jit()
def loss_fn(
    model: nnx.Module,
    input_seq: Int[jnp.ndarray, "batch_size context_length"],
    target_seq: Int[jnp.ndarray, "batch_size context_length"],
) -> Float[jnp.ndarray, ""]:
    """Computes the loss for the model."""
    logits = model(input_seq)
    return functions.cross_entropy_loss(logits=logits, target_seq=target_seq)


def train_loop(
    model: nnx.Module,
    nnx_optimizer: nnx.Optimizer,
    train_dataset: grain.IterDataset,
    validation_dataset: grain.IterDataset,
    train_config: _train_config.TrainConfig,
    ckpt_manager: checkpoint.CheckpointManager,
    start_step: int = 0,
    *,
    wandb_run: wandb.Run | None = None,
    log_train_metrics_every_n_steps: int = 10,
    validation_every_n_steps: int = 10,
) -> None:
    """Trains the model."""

    @nnx.jit(donate_argnames=("local_model", "local_optimizer"))
    def _train_step(
        local_model: nnx.Module,
        local_optimizer: nnx.Optimizer,
        input_seq: Int[jnp.ndarray, "batch_size context_length"],
        target_seq: Int[jnp.ndarray, "batch_size context_length"],
    ) -> tuple[Float[jnp.ndarray, ""], Float[jnp.ndarray, ""]]:
        """Trains the model for one step."""
        loss, grads = nnx.value_and_grad(loss_fn)(local_model, input_seq, target_seq)
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
    for step in tqdm(
        range(start_step, train_config.num_steps),
        initial=start_step,
        total=train_config.num_steps,
        desc="Training",
    ):
        input_seq, target_seq = next(train_iter)
        loss, total_gradient_l2_norm = _train_step(
            model, nnx_optimizer, input_seq, target_seq
        )
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
            run_validation(
                model=model,
                input_seq=validation_input_seq,
                target_seq=validation_target_seq,
                wandb_run=wandb_run,
                step=step,
            )
        ckpt_manager.save(
            step=step,
            model=model,
            optimizer=nnx_optimizer,
            metadata={},
        )
    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


def run_validation(
    model: nnx.Module,
    input_seq: Int[jnp.ndarray, "batch_size context_length"],
    target_seq: Int[jnp.ndarray, "batch_size context_length"],
    wandb_run: wandb.Run,
    step: int,
) -> None:
    """Runs validation."""
    loss = loss_fn(model, input_seq, target_seq)
    perplexity = jnp.exp(loss)
    wandb_run.log(
        {
            "validation/loss": loss,
            "validation/perplexity": perplexity,
        },
        step=step,
    )
    logging.info(f"Step {step}: Validation loss: {loss}, perplexity: {perplexity}")
