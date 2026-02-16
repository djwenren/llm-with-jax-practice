#!/bin/bash

set -euo pipefail

BASE_CHECKPOINT_DIR="$1"
BASE_DATA_DIR="$2"
EXP_NAME="$3"

CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${EXP_NAME}"
MAX_CKPTS_TO_KEEP=4
CKPT_SAVE_INTERVAL_STEPS=10
TRAINING_DATA_SOURCE_PATH="${BASE_DATA_DIR}/tiny_stories_train_tokens.npy"
VALIDATION_DATA_SOURCE_PATH="${BASE_DATA_DIR}/tiny_stories_valid_tokens.npy"
USE_MODEL_AND_TRAIN_CONFIG_FROM_CHECKPOINT=True
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="llm-with-jax-practice"
WANDB_RUN_NAME="${EXP_NAME}"
LOG_TRAIN_METRICS_EVERY_N_STEPS=10
VALIDATION_EVERY_N_STEPS=20

# Training config
NUM_STEPS=5000
TRAINING_BATCH_SIZE=32
VALIDATION_BATCH_SIZE=32
MAX_TOTAL_GRADIENT_L2_NORM=4.0
ADAMW_BETA_1=0.9
ADAMW_BETA_2=0.999
ADAMW_EPS=1e-8
ADAMW_WEIGHT_DECAY=1e-3
COSINE_ONECYCLE_MAX_LEARNING_RATE=1e-3
COSINE_ONECYCLE_MIN_LEARNING_RATE=1e-4
COSINE_ONECYCLE_WARMUP_ITERS=100
COSINE_ONECYCLE_COSINE_CYCLE_ITERS=4900

# Model config
# VOCAB_SIZE=10000
# CONTEXT_LENGTH=256
# NUM_LAYERS=4
# NUM_HEADS=16
# ROPE_THETA=10000
# D_MODEL=512
# D_FF_TO_D_MODEL=2.6666667
# D_FF=1344
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
NUM_LAYERS=4
NUM_HEADS=8
ROPE_THETA=10000
D_MODEL=256
D_FF_TO_D_MODEL=2.6666667
D_FF=672
SHARDING_STRATEGY="fsdp_tp"

TRAIN_CMD="uv run llm_with_jax_practice/train_main.py"

${TRAIN_CMD} \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --max_ckpts_to_keep="${MAX_CKPTS_TO_KEEP}" \
  --ckpt_save_interval_steps="${CKPT_SAVE_INTERVAL_STEPS}" \
  --training_data_source_path="${TRAINING_DATA_SOURCE_PATH}" \
  --validation_data_source_path="${VALIDATION_DATA_SOURCE_PATH}" \
  --use_model_and_train_config_from_checkpoint="${USE_MODEL_AND_TRAIN_CONFIG_FROM_CHECKPOINT}" \
  --wandb_entity="${WANDB_ENTITY}" \
  --wandb_project="${WANDB_PROJECT}" \
  --wandb_run_name="${WANDB_RUN_NAME}" \
  --log_train_metrics_every_n_steps="${LOG_TRAIN_METRICS_EVERY_N_STEPS}" \
  --validation_every_n_steps="${VALIDATION_EVERY_N_STEPS}" \
  --num_steps="${NUM_STEPS}" \
  --training_batch_size="${TRAINING_BATCH_SIZE}" \
  --validation_batch_size="${VALIDATION_BATCH_SIZE}" \
  --max_total_gradient_l2_norm="${MAX_TOTAL_GRADIENT_L2_NORM}" \
  --adamw_beta_1="${ADAMW_BETA_1}" \
  --adamw_beta_2="${ADAMW_BETA_2}" \
  --adamw_eps="${ADAMW_EPS}" \
  --adamw_weight_decay="${ADAMW_WEIGHT_DECAY}" \
  --cosine_onecycle_max_learning_rate="${COSINE_ONECYCLE_MAX_LEARNING_RATE}" \
  --cosine_onecycle_min_learning_rate="${COSINE_ONECYCLE_MIN_LEARNING_RATE}" \
  --cosine_onecycle_warmup_iters="${COSINE_ONECYCLE_WARMUP_ITERS}" \
  --cosine_onecycle_cosine_cycle_iters="${COSINE_ONECYCLE_COSINE_CYCLE_ITERS}" \
  --vocab_size="${VOCAB_SIZE}" \
  --context_length="${CONTEXT_LENGTH}" \
  --num_layers="${NUM_LAYERS}" \
  --num_heads="${NUM_HEADS}" \
  --rope_theta="${ROPE_THETA}" \
  --d_model="${D_MODEL}" \
  --d_ff_to_d_model="${D_FF_TO_D_MODEL}" \
  --d_ff="${D_FF}" \
  --sharding_strategy="${SHARDING_STRATEGY}"