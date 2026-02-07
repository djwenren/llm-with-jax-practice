"""Data loader for LLM with JAX Practice."""

import grain
import jax.numpy as jnp
import numpy.typing as npt

from jaxtyping import Int


class TransformerLmDataSource(grain.sources.RandomAccessDataSource):
    """Data source for Transformer language model."""

    def __init__(self, np_data: Int[npt.NDArray, "dataset_size"], context_length: int):
        self._np_data = np_data
        self._context_length = context_length
        self._total_num_sequences = self._np_data.shape[0] - self._context_length

    def __getitem__(
        self, index: int
    ) -> tuple[Int[jnp.ndarray, "context_length"], Int[jnp.ndarray, "context_length"]]:
        input_tokens = self._np_data[index : index + self._context_length]
        target_tokens = self._np_data[index + 1 : index + self._context_length + 1]
        return input_tokens, target_tokens

    def __len__(self) -> int:
        return self._total_num_sequences


def get_dataset(
    np_data: Int[npt.NDArray, "dataset_size"],
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
    use_repeat: bool = False,
    num_repeats: int | None = None,
) -> grain.IterDataset[
    tuple[
        Int[jnp.ndarray, "batch_size context_length"],
        Int[jnp.ndarray, "batch_size context_length"],
    ]
]:
    """Gets dataset for Transformer language model."""
    data_source = TransformerLmDataSource(np_data, context_length)
    dataset = grain.MapDataset.source(data_source)
    if shuffle:
        dataset = dataset.shuffle(seed)
    if use_repeat:
        dataset = dataset.repeat(num_repeats)
    # No need to convert to Jax (not recommended). Also no need to pin_memory or send to GPU like
    # in PyTorch. https://gemini.google.com/app/b3ca6a3cd9b704c7.
    return dataset.to_iter_dataset().batch(batch_size=batch_size)
