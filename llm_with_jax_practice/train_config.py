"""Train configuration."""

from absl import flags
from flax import nnx


_num_steps = flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
_training_batch_size = flags.DEFINE_integer(
    "training_batch_size", 128, "Training batch size."
)
_context_length = flags.DEFINE_integer("context_length", 16, "Context length.")
_validation_batch_size = flags.DEFINE_integer(
    "validation_batch_size", 128, "Validation batch size."
)
_max_total_gradient_l2_norm = flags.DEFINE_float(
    "max_total_gradient_l2_norm", None, "Maximum total gradient L2 norm."
)


@nnx.dataclass
class TrainConfig(nnx.Pytree):
    """Train configuration."""

    num_steps: int
    training_batch_size: int
    context_length: int

    validation_batch_size: int

    max_total_gradient_l2_norm: float | None = None


def get_train_config() -> TrainConfig:
    """Get train configuration."""
    return TrainConfig(
        num_steps=_num_steps.value,
        training_batch_size=_training_batch_size.value,
        context_length=_context_length.value,
        validation_batch_size=_validation_batch_size.value,
        max_total_gradient_l2_norm=_max_total_gradient_l2_norm.value,
    )
