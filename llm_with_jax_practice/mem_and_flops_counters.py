"""Mem and flops counters."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from llm_with_jax_practice import layers


@dataclasses.dataclass(kw_only=True, frozen=True)
class CountingResults:
    """Results of counting memory and flops."""

    num_params: int
    param_bytes: int
    num_activation_params: int
    activation_bytes: int
    flops: int


class LinearMemAndFlopsCounter:
    """Mem and flops counter for linear layer."""

    def __init__(
        self, in_features: int, out_features: int, *, dtype: jnp.dtype = jnp.float32
    ):
        self._linear = nnx.eval_shape(
            lambda: layers.Linear(
                in_features,
                out_features,
                rngs=nnx.Rngs(jax.random.key(42)),
                dtype=dtype,
            )
        )
        self._num_params = np.prod(self._linear.weight.shape)
        self._param_bytes = self._num_params * self._linear.weight.dtype.itemsize

    def count(
        self, x: jax.ShapeDtypeStruct, is_training: bool = True
    ) -> CountingResults:
        """Count the memory and flops for the linear layer."""
        flops = (
            2
            * self._linear.weight.shape[0]
            * self._linear.weight.shape[1]
            * np.prod(x.shape[:-1])
        )
        if is_training:
            # Counting the cost of the backward pass.
            flops *= 3
            num_params = self._num_params * 3
            param_bytes = self._param_bytes * 3
        else:
            num_params = self._num_params
            param_bytes = self._param_bytes
        # Pass both self._linear and x to the eval_shape function instead of capturing them so that
        # eval_shape can wrap them in a `jax.ShapeDtypeStruct`.
        output = nnx.eval_shape(lambda m, x: m(x), self._linear, x)
        num_activation_params = output.size
        activation_bytes = num_activation_params * output.dtype.itemsize
        return CountingResults(
            num_params=num_params,
            param_bytes=param_bytes,
            num_activation_params=num_activation_params,
            activation_bytes=activation_bytes,
            flops=flops,
        )
