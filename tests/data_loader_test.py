"""Data loader test."""

import math
from collections import Counter

import numpy as np
import jax

from llm_with_jax_practice import data_loader


def test_get_dataset():
    """Test get dataset."""
    dataset_size = 103
    test_numpy_data = np.arange(dataset_size)
    batch_size = 32
    context_length = 7
    seed = 42

    dataset = data_loader.get_dataset(
        np_data=test_numpy_data,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_epochs = 10
    for _ in range(num_epochs):
        for x, y in dataset:
            # Make sure the shape is correct
            assert x.shape == (batch_size, context_length)
            assert y.shape == (batch_size, context_length)

            # Make sure the y's are always offset by 1
            np.testing.assert_allclose(x + 1, y)

            starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = dataset_size - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_epochs * dataset_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_epochs * dataset_size)
        * (1 / num_possible_starting_indices)
        * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )


def test_same_shuffle_every_epoch_without_using_repeat():
    """Test that the shuffle is the same every epoch."""
    dataset_size = 10
    test_numpy_data = np.arange(dataset_size)
    batch_size = 2
    context_length = 4
    seed = 42

    dataset = data_loader.get_dataset(
        np_data=test_numpy_data,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    first_epoch_data = []
    second_epoch_data = []
    for x, y in dataset:
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        np.testing.assert_allclose(x + 1, y)
        first_epoch_data.append((x, y))
    for x, y in dataset:
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        np.testing.assert_allclose(x + 1, y)
        second_epoch_data.append((x, y))
    jax.tree.map(lambda x, y: x == y, first_epoch_data, second_epoch_data)


def test_different_shuffle_every_epoch_with_using_repeat():
    """Test that the shuffle is different every epoch when using repeat."""
    dataset_size = 10
    test_numpy_data = np.arange(dataset_size)
    batch_size = 2
    context_length = 4
    seed = 42

    dataset = data_loader.get_dataset(
        np_data=test_numpy_data,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        use_repeat=True,
        num_repeats=2,
    )

    both_epoch_data = []
    for x, y in dataset:
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        np.testing.assert_allclose(x + 1, y)
        both_epoch_data.append((x, y))
    num_batechs_per_epoch = len(both_epoch_data) // 2
    jax.tree.map(
        lambda x, y: x != y,
        both_epoch_data[:num_batechs_per_epoch],
        both_epoch_data[num_batechs_per_epoch:],
    )
