"""Tests for mems and flops counters."""

import jax
import jax.numpy as jnp

from llm_with_jax_practice import mem_and_flops_counters


class TestMemAndFlopsCounters:
    """Tests for mem and flops counters."""

    def test_linear(self):
        """Test linear mem and flops counter."""
        in_features = 10
        out_features = 20
        dtype = jnp.float16
        mem_and_flops_counter = mem_and_flops_counters.LinearMemAndFlopsCounter(
            in_features, out_features, dtype=dtype
        )
        x = jax.ShapeDtypeStruct(shape=(10, 20, in_features), dtype=jnp.float32)
        results = mem_and_flops_counter.count(x)
        print(results)
