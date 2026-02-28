#!/bin/bash
# Training script for CMG-based teacher models
# Usage: bash train_teacher_cmg.sh <speed_mode> <exptid> <device> [resumeid] [checkpoint] [num_envs] [max_iterations] [unlock_upper_body]
#   speed_mode: slow | medium | fast
#   exptid: experiment ID (e.g., cmg_slow_v1)
#   device: cuda device (e.g., cuda:0)
#   resumeid: optional run folder to resume from under legged_gym/logs/<proj_name>/
#   checkpoint: optional checkpoint number, e.g. 10000 (default: latest if resumeid is set)
#   num_envs: optional env count override, e.g. 1024
#   max_iterations: optional training iterations override, e.g. 6001
#   unlock_upper_body: optional flag (unlock|true|1) to enable upper-body DOFs (default: unlock)

set -e

# Isaac Gym binary extension needs conda's libpython at runtime.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Reduce CUDA allocator fragmentation risk in long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd legged_gym/legged_gym/scripts

speed_mode=${1:-slow}
exptid=${2:-cmg_${speed_mode}_v1}
device=${3:-cuda:0}
resumeid=${4:-}
checkpoint=${5:-}
num_envs=${6:-}
max_iterations=${7:-}
unlock_upper_body=${8:-unlock}

# Map speed mode to task name
case $speed_mode in
    slow)
        task_name="g1_cmg_slow"
        proj_name="g1_cmg_slow"
        ;;
    medium)
        task_name="g1_cmg_medium"
        proj_name="g1_cmg_medium"
        ;;
    fast)
        task_name="g1_cmg_fast"
        proj_name="g1_cmg_fast"
        ;;
    *)
        echo "Error: Invalid speed_mode '$speed_mode'. Must be one of: slow, medium, fast"
        exit 1
        ;;
esac

echo "============================================"
echo "CMG Teacher Training"
echo "============================================"
echo "Speed Mode: $speed_mode"
echo "Task Name: $task_name"
echo "Project Name: $proj_name"
echo "Experiment ID: $exptid"
echo "Device: $device"
if [ -n "$resumeid" ]; then
    echo "Resume ID: $resumeid"
fi
if [ -n "$checkpoint" ]; then
    echo "Checkpoint: $checkpoint"
fi
if [ -n "$num_envs" ]; then
    echo "Num Envs: $num_envs"
fi
if [ -n "$max_iterations" ]; then
    echo "Max Iterations: $max_iterations"
fi
if [ -n "$unlock_upper_body" ]; then
    echo "Unlock Upper Body: $unlock_upper_body"
fi
echo "============================================"

cmd=(
    python train.py
    --task "${task_name}"
    --proj_name "${proj_name}"
    --exptid "${exptid}"
    --device "${device}"
)

if [ -n "$resumeid" ]; then
    cmd+=(--resumeid "${resumeid}")
fi

if [ -n "$checkpoint" ]; then
    cmd+=(--checkpoint "${checkpoint}")
fi

if [ -n "$num_envs" ]; then
    cmd+=(--num_envs "${num_envs}")
fi

if [ -n "$max_iterations" ]; then
    cmd+=(--max_iterations "${max_iterations}")
fi

if [ "${unlock_upper_body}" = "unlock" ] || [ "${unlock_upper_body}" = "true" ] || [ "${unlock_upper_body}" = "1" ]; then
    cmd+=(--unlock_upper_body)
fi

"${cmd[@]}"
