#!/usr/bin/env bash
# Route project scratch/cache files away from the nodes' shared root /tmp.
# Source this file before starting training, evaluation, plotting, or TeX jobs.

IRES_SCRATCH_ROOT="${IRES_SCRATCH_ROOT:-${HOME}/.cache/ires-design}"

export IRES_SCRATCH_ROOT
export TMPDIR="${IRES_TMPDIR:-$IRES_SCRATCH_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${IRES_XDG_CACHE_HOME:-$IRES_SCRATCH_ROOT/cache}"
export MPLCONFIGDIR="${IRES_MPLCONFIGDIR:-$IRES_SCRATCH_ROOT/matplotlib}"
export TORCH_HOME="${IRES_TORCH_HOME:-$IRES_SCRATCH_ROOT/torch}"
export HF_HOME="${IRES_HF_HOME:-$IRES_SCRATCH_ROOT/huggingface}"
export WANDB_DIR="${IRES_WANDB_DIR:-$IRES_SCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${IRES_WANDB_CACHE_DIR:-$IRES_SCRATCH_ROOT/wandb-cache}"
export CUDA_CACHE_PATH="${IRES_CUDA_CACHE_PATH:-$IRES_SCRATCH_ROOT/cuda-cache}"
export TRITON_CACHE_DIR="${IRES_TRITON_CACHE_DIR:-$IRES_SCRATCH_ROOT/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${IRES_TORCHINDUCTOR_CACHE_DIR:-$IRES_SCRATCH_ROOT/torchinductor-cache}"
export PYTHONPYCACHEPREFIX="${IRES_PYTHONPYCACHEPREFIX:-$IRES_SCRATCH_ROOT/pycache}"

mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX"
