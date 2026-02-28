#!/bin/bash
# Student training script using a CMG teacher checkpoint
# Usage:
# bash train_student_cmg.sh <speed_mode> <student_exptid> <teacher_exptid> <device> [teacher_checkpoint] [resumeid] [checkpoint] [num_envs] [max_iterations] [unlock_upper_body=unlock] [curriculum_stage=full] [resume_proj_name] [startup_check=1]
#
# Notes:
# - Stage-B defaults: num_envs=2048 and startup_check=1
# - Stage-C curriculum_stage for fast task: full | narrow
# - Cross-project resume (medium->fast migration) can be enabled with resume_proj_name

set -euo pipefail

# Isaac Gym binary extension needs conda's libpython at runtime.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Reduce CUDA allocator fragmentation risk in long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
if command -v rg >/dev/null 2>&1; then
    SEARCH_BIN="rg"
else
    SEARCH_BIN="grep -E"
fi

speed_mode=${1:-medium}
exptid=${2:-cmg_student_${speed_mode}_v1}
teacher_exptid=${3:-cmg_teacher_${speed_mode}_v1}
device=${4:-cuda:0}
teacher_checkpoint=${5:--1}
resumeid=${6:-}
checkpoint=${7:-}
num_envs=${8:-2048}
max_iterations=${9:-}
unlock_upper_body=${10:-unlock}
curriculum_stage=${11:-full}
resume_proj_name=${12:-}
startup_check=${13:-1}

# Map CMG speed mode to teacher project folder and CMG-aligned student task.
case "${speed_mode}" in
    slow)
        teacher_proj_name="g1_cmg_slow"
        task_name="g1_stu_rl_cmg_slow"
        proj_name="g1_stu_rl_cmg_slow"
        ;;
    medium)
        teacher_proj_name="g1_cmg_medium"
        task_name="g1_stu_rl_cmg_medium"
        proj_name="g1_stu_rl_cmg_medium"
        ;;
    fast)
        teacher_proj_name="g1_cmg_fast"
        task_name="g1_stu_rl_cmg_fast"
        proj_name="g1_stu_rl_cmg_fast"
        if [ "${curriculum_stage}" = "narrow" ]; then
            task_name="g1_stu_rl_cmg_fast_narrow"
        fi
        ;;
    *)
        echo "Error: Invalid speed_mode '${speed_mode}'. Must be one of: slow, medium, fast"
        exit 1
        ;;
esac

if [ "${speed_mode}" != "fast" ] && [ "${curriculum_stage}" != "full" ]; then
    echo "Warning: curriculum_stage='${curriculum_stage}' is ignored for speed_mode='${speed_mode}'"
fi

if [ -n "${resumeid}" ] && [ -z "${checkpoint}" ]; then
    checkpoint="-1"
fi

resume_proj_name_effective="${resume_proj_name:-${proj_name}}"

resolve_model_path() {
    local run_dir="$1"
    local ckpt="$2"

    if [ ! -d "${run_dir}" ]; then
        return 1
    fi

    if [ -z "${ckpt}" ] || [ "${ckpt}" = "-1" ]; then
        local latest
        latest=$(ls -1 "${run_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n1 || true)
        if [ -z "${latest}" ]; then
            return 1
        fi
        echo "${latest}"
    else
        local pth="${run_dir}/model_${ckpt}.pt"
        if [ ! -f "${pth}" ]; then
            return 1
        fi
        echo "${pth}"
    fi
}

echo "============================================"
echo "CMG Student Training (DAgger: RL+BC)"
echo "============================================"
echo "Speed Mode: ${speed_mode}"
echo "Task Name: ${task_name}"
echo "Project Name: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo "Teacher Experiment ID: ${teacher_exptid}"
echo "Teacher Project Name: ${teacher_proj_name}"
echo "Teacher Checkpoint: ${teacher_checkpoint}"
echo "Device: ${device}"
echo "Num Envs: ${num_envs}"
echo "Curriculum Stage: ${curriculum_stage}"
if [ -n "${resumeid}" ]; then
    echo "Resume ID: ${resumeid}"
    echo "Resume Project Name: ${resume_proj_name_effective}"
    echo "Student Checkpoint: ${checkpoint}"
fi
if [ -n "${max_iterations}" ]; then
    echo "Max Iterations: ${max_iterations}"
fi
if [ -n "${unlock_upper_body}" ]; then
    echo "Unlock Upper Body: ${unlock_upper_body}"
fi
echo "============================================"

teacher_run_dir="legged_gym/logs/${teacher_proj_name}/${teacher_exptid}"
teacher_model_path=$(resolve_model_path "${teacher_run_dir}" "${teacher_checkpoint}" || true)
if [ -z "${teacher_model_path}" ]; then
    echo "Error: teacher checkpoint not found in '${teacher_run_dir}' with checkpoint='${teacher_checkpoint}'"
    exit 1
fi
echo "[Precheck] Teacher model: ${teacher_model_path}"

if [ -n "${resumeid}" ]; then
    student_resume_dir="legged_gym/logs/${resume_proj_name_effective}/${resumeid}"
    student_model_path=$(resolve_model_path "${student_resume_dir}" "${checkpoint}" || true)
    if [ -z "${student_model_path}" ]; then
        echo "Error: student resume checkpoint not found in '${student_resume_dir}' with checkpoint='${checkpoint}'"
        exit 1
    fi
    echo "[Precheck] Student resume model: ${student_model_path}"
fi

cd legged_gym/legged_gym/scripts

cmd=(
    "${PYTHON_BIN}" train.py
    --task "${task_name}"
    --proj_name "${proj_name}"
    --exptid "${exptid}"
    --teacher_exptid "${teacher_exptid}"
    --teacher_proj_name "${teacher_proj_name}"
    --teacher_checkpoint "${teacher_checkpoint}"
    --device "${device}"
    --num_envs "${num_envs}"
)

if [ -n "${resumeid}" ]; then
    cmd+=(--resumeid "${resumeid}")
    cmd+=(--resume_proj_name "${resume_proj_name_effective}")
fi

if [ -n "${checkpoint}" ]; then
    cmd+=(--checkpoint "${checkpoint}")
fi

if [ -n "${max_iterations}" ]; then
    cmd+=(--max_iterations "${max_iterations}")
fi

if [ "${unlock_upper_body}" = "unlock" ] || [ "${unlock_upper_body}" = "true" ] || [ "${unlock_upper_body}" = "1" ]; then
    cmd+=(--unlock_upper_body)
fi

run_log_dir="../../logs/${proj_name}/${exptid}"
mkdir -p "${run_log_dir}"
run_log="${run_log_dir}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[TrainLog] ${run_log}"
echo "[TrainCmd] ${cmd[*]}"

echo "[Template] Resume command (single line):"
echo "bash train_student_cmg.sh ${speed_mode} ${exptid} ${teacher_exptid} ${device} ${teacher_checkpoint} ${exptid} -1 ${num_envs} ${max_iterations:-30002} ${unlock_upper_body} ${curriculum_stage} ${proj_name} 1"

# Keep a plain training log for startup checks and gate checks.
("${cmd[@]}" 2>&1 | tee -a "${run_log}") &
train_pid=$!

if [ "${startup_check}" = "1" ]; then
    sleep 30
    echo "[StartupCheck] 30s check on ${run_log}"

    startup_ok=1

    if ! ${SEARCH_BIN} -q "\[TrainInfo\] num_envs=${num_envs}([^0-9]|$)" "${run_log}"; then
        echo "[StartupCheck][FAIL] Num Envs mismatch or not printed. Expected num_envs=${num_envs}"
        startup_ok=0
    else
        echo "[StartupCheck][OK] Num Envs=${num_envs}"
    fi

    if ! ${SEARCH_BIN} -q "Loading teacher policy from" "${run_log}"; then
        echo "[StartupCheck][FAIL] Teacher checkpoint load log not found"
        startup_ok=0
    else
        echo "[StartupCheck][OK] Teacher checkpoint loaded"
    fi

    if [ -n "${resumeid}" ]; then
        if ! ${SEARCH_BIN} -q "Loading model from" "${run_log}"; then
            echo "[StartupCheck][FAIL] Student resume checkpoint load log not found"
            startup_ok=0
        else
            echo "[StartupCheck][OK] Student resume checkpoint loaded"
        fi
    fi

    if ! ${SEARCH_BIN} -q "Learning iteration|Training:" "${run_log}"; then
        echo "[StartupCheck][FAIL] Iteration logs not observed in first 30s"
        startup_ok=0
    else
        echo "[StartupCheck][OK] Iteration logs observed"
    fi

    if [ "${startup_ok}" = "0" ]; then
        echo "[StartupCheck] FAILED. Check log and stop/restart if needed."
    else
        echo "[StartupCheck] PASSED. Training looks healthy in first 30s."
    fi
fi

wait "${train_pid}"
train_status=$?

if [ "${train_status}" -eq 0 ]; then
    echo "[Post] Training finished. You can evaluate gates with:"
    echo "python3 ../../tools/check_student_gates.py --log ${run_log}"
fi

exit "${train_status}"
