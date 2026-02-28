#!/bin/bash
# Student training script (DAgger: RL+BC)
# Usage:
# bash train_student.sh <student_exptid> <teacher_exptid> <device> [resumeid] [checkpoint]
# Example:
# bash train_student.sh 0927_twist_rlbcstu 0927_twist_teacher cuda:0
# Resume example:
# bash train_student.sh 0927_twist_rlbcstu 0927_twist_teacher cuda:0 0927_twist_rlbcstu 12000

set -e

# Isaac Gym binary extension needs conda's libpython at runtime.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

exptid=${1}
teacher_exptid=${2}
device=${3}
resumeid=${4:-}
checkpoint=${5:-}

task_name="g1_stu_rl"

proj_name="g1_stu_rl"

cd legged_gym/legged_gym/scripts


cmd=(
    python train.py
    --task "${task_name}"
    --proj_name "${proj_name}"
    --exptid "${exptid}"
    --teacher_exptid "${teacher_exptid}"
    --device "${device}"
)

if [ -n "$resumeid" ]; then
    cmd+=(--resumeid "${resumeid}")
fi

if [ -n "$checkpoint" ]; then
    cmd+=(--checkpoint "${checkpoint}")
fi

"${cmd[@]}"
