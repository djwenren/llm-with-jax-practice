"""Train configuration."""

from absl import flags
from flax import nnx


_num_steps = flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
_training_batch_size = flags.DEFINE_integer(
    "training_batch_size", 128, "Training batch size."
)
_validation_batch_size = flags.DEFINE_integer(
    "validation_batch_size", 128, "Validation batch size."
)
_max_total_gradient_l2_norm = flags.DEFINE_float(
    "max_total_gradient_l2_norm", None, "Maximum total gradient L2 norm."
)
_adamw_beta_1 = flags.DEFINE_float("adamw_beta_1", 0.9, "AdamW beta 1.")
_adamw_beta_2 = flags.DEFINE_float("adamw_beta_2", 0.999, "AdamW beta 2.")
_adamw_eps = flags.DEFINE_float("adamw_eps", 1e-8, "AdamW eps.")
_adamw_weight_decay = flags.DEFINE_float(
    "adamw_weight_decay", 1e-3, "AdamW weight decay."
)
_cosine_onecycle_max_learning_rate = flags.DEFINE_float(
    "cosine_onecycle_max_learning_rate", 1e-3, "Cosine onecycle max learning rate."
)
_cosine_onecycle_min_learning_rate = flags.DEFINE_float(
    "cosine_onecycle_min_learning_rate", 1e-4, "Cosine onecycle min learning rate."
)
_cosine_onecycle_warmup_iters = flags.DEFINE_integer(
    "cosine_onecycle_warmup_iters", 50, "Cosine onecycle warmup iters."
)
_cosine_onecycle_cosine_cycle_iters = flags.DEFINE_integer(
    "cosine_onecycle_cosine_cycle_iters", 50, "Cosine onecycle cosine cycle iters."
)
_use_mu_p = flags.DEFINE_boolean(
    "use_mu_p", False, "Use mu-p parameter initialization."
)
_num_microbatches = flags.DEFINE_integer(
    "num_microbatches", 1, "Number of microbatches."
)


@nnx.dataclass
class TrainConfig(nnx.Pytree):
    """Train configuration."""

    num_steps: int
    training_batch_size: int

    validation_batch_size: int

    adamw_beta_1: float = 0.9
    adamw_beta_2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 1e-3

    # In the case muP, these learning rates are the base learning rates for the embedding and
    # block/output layers.
    # In the case SP, these learning rates are the only learning rates used.
    cosine_onecycle_max_learning_rate: float = 1e-3
    cosine_onecycle_min_learning_rate: float = 1e-4
    cosine_onecycle_warmup_iters: int = 50
    cosine_onecycle_cosine_cycle_iters: int = 50

    max_total_gradient_l2_norm: float | None = None

    use_mu_p: bool = False

    num_microbatches: int = 1


def get_train_config() -> TrainConfig:
    """Get train configuration."""
    if _cosine_onecycle_min_learning_rate.value > 0:
        min_learning_rate = _cosine_onecycle_min_learning_rate.value
    else:
        min_learning_rate = _cosine_onecycle_max_learning_rate.value * 0.1
    return TrainConfig(
        num_steps=_num_steps.value,
        training_batch_size=_training_batch_size.value,
        validation_batch_size=_validation_batch_size.value,
        adamw_beta_1=_adamw_beta_1.value,
        adamw_beta_2=_adamw_beta_2.value,
        adamw_eps=_adamw_eps.value,
        adamw_weight_decay=_adamw_weight_decay.value,
        cosine_onecycle_max_learning_rate=_cosine_onecycle_max_learning_rate.value,
        cosine_onecycle_min_learning_rate=min_learning_rate,
        cosine_onecycle_warmup_iters=_cosine_onecycle_warmup_iters.value,
        cosine_onecycle_cosine_cycle_iters=_cosine_onecycle_cosine_cycle_iters.value,
        max_total_gradient_l2_norm=_max_total_gradient_l2_norm.value,
        use_mu_p=_use_mu_p.value,
        num_microbatches=_num_microbatches.value,
    )
