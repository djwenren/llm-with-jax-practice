# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A transformer-based language model implementation using JAX/Flax NNX, with support for muP (Maximum Update Parameterization) and distributed training (FSDP, Tensor Parallelism). Uses Grain for data loading, Orbax for checkpointing, and wandb for experiment tracking.

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/layers_test.py

# Run a single test
uv run pytest tests/layers_test.py::test_function_name -s

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Run training (example: SP on CPU)
uv run llm_with_jax_practice/train_main.py --checkpoint_dir=/tmp/test ...

# Training scripts are in llm_with_jax_practice/scripts/
bash llm_with_jax_practice/scripts/run_llm_train_sp_single_cpu_integration_test.sh
```

## Architecture

### Core Modules

- **`transformer.py`** — `TransformerLm` model and `TransformerConfig`. The model uses `jax.lax.scan` over a stack of `TransformerBlock`s. Config is a frozen dataclass with absl flags for CLI.
- **`layers.py`** — Building blocks: `Linear`, `Embedding`, `RMSNorm`, `RoPE`, `SwiGLU`, `MultiHeadSelfAttention`, `TransformerBlock`. All are Flax NNX modules (`nnx.Module`).
- **`functions.py`** — Activation functions (`silu`, `softmax`), `scaled_dot_product_attention`, `cross_entropy_loss`.
- **`optimizer.py`** — Custom `AdamW` implementation using Optax primitives with learning rate and weight decay scaling.
- **`train_main.py`** — Training entry point. Handles checkpoint restore, sharding setup, config reconciliation between CLI flags and checkpoint state.
- **`train_utils.py`** — Training loop implementations (`sp_train_loop`, `mup_train_loop`, FSDP variants), loss/validation functions, dataset creation, wandb integration, gradient accumulation.
- **`sharding.py`** — Sharding specs (`LinearSharding`, `EmbeddingSharding`, etc.) using JAX `PartitionSpec`. Presets: `"none"`, `"fsdp"`, `"fsdp_tp"`.
- **`checkpoint.py`** — Orbax-based checkpoint management. Saves model, optimizer state, and configs.
- **`data_loader.py`** — Grain-based `TransformerLmDataSource` with shuffle/repeat/batch pipeline.

### Key Patterns

- **Configuration**: All config via `absl.flags` defined per-module, collected into frozen dataclasses (`TransformerConfig`, `TrainConfig`).
- **muP (Maximum Update Parameterization)**: Embedding uses `alpha_input` scaling; Linear layers have `alpha` output scaling; initialization std scales with `1/sqrt(m_p)`. Controlled by `use_mu_p` flag.
- **Sharding**: `Linear._match_sharding()` resolves sharding specs based on weight shape. FSDP shards along "data" mesh axis; FSDP+TP adds "model" axis.
- **Training loop**: Supports gradient accumulation (`num_microbatches`), gradient clipping, cosine LR schedule with warmup, periodic validation and checkpointing.

### Test Infrastructure

Tests live in `tests/` and use pytest. Snapshot testing (`NumpySnapshot` for `.npz`, `Snapshot` for `.pkl`) validates numerical outputs against reference data in `tests/_snapshots/`. Reference PyTorch model in `tests/fixtures/ts_tests/` is used for cross-framework validation.

## Dependencies

Python 3.12+. Core: JAX (0.9.0+), Flax (0.12.3+), Optax, Grain, Orbax. Ruff for linting (80-char line length).
