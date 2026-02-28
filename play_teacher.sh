# Usage:
# bash play_teacher.sh <teacher_exptid> [teacher_proj_name] [extra_play_args...]
# Examples:
# bash play_teacher.sh cmg_teacher_medium_v1
# bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium
# bash play_teacher.sh 0927_twist_teacher g1_priv_mimic
# bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium --cmd_vx 2.0 --cmd_vy 0.0 --cmd_yaw 0.0

# Isaac Gym binary extension needs conda's libpython at runtime.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Reduce CUDA allocator fragmentation risk in long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exptid=$1
proj_name=${2:-}
extra_args=("${@:3}")

# Infer project/task for CMG teachers when not provided explicitly.
if [ -z "${proj_name}" ]; then
    case "${exptid}" in
        cmg_teacher_slow*)
            proj_name="g1_cmg_slow"
            ;;
        cmg_teacher_medium*)
            proj_name="g1_cmg_medium"
            ;;
        cmg_teacher_fast*)
            proj_name="g1_cmg_fast"
            ;;
        *)
            proj_name="g1_priv_mimic"
            ;;
    esac
fi

task_name="${proj_name}"

echo "Teacher exptid: ${exptid}"
echo "Teacher proj_name: ${proj_name}"
echo "Task name: ${task_name}"

cd legged_gym/legged_gym/scripts

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --record_video \
                "${extra_args[@]}" \
