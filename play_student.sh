
# Usage:
# bash play_student.sh <student_exptid> <teacher_exptid> [teacher_proj_name] [extra_play_args...]
# Examples:
# bash play_student.sh cmg_student_medium_v1 cmg_teacher_medium_v1
# bash play_student.sh cmg_student_medium_v1 cmg_teacher_medium_v1 g1_cmg_medium
# bash play_student.sh cmg_student_fast_v1 cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 3.0 --cmd_vy 0.0 --cmd_yaw 0.0
# bash play_student.sh cmg_student_fast_v2_low_cmdswitch cmg_teacher_fast_v1 g1_cmg_fast --cmdswitch

# Isaac Gym binary extension needs conda's libpython at runtime.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Reduce CUDA allocator fragmentation risk in long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd legged_gym/legged_gym/scripts


exptid=$1
teacher_exptid=$2
teacher_proj_name=""
extra_args=()
cmdswitch_mode=0

if [ -n "${3:-}" ]; then
    if [[ "${3}" == --* ]]; then
        extra_args=("${@:3}")
    else
        teacher_proj_name="${3}"
        extra_args=("${@:4}")
    fi
fi
task_name="g1_stu_rl"
proj_name="g1_stu_rl"

# Infer CMG teacher project when not provided explicitly.
if [ -z "${teacher_proj_name}" ]; then
    case "${teacher_exptid}" in
        cmg_teacher_slow*)
            teacher_proj_name="g1_cmg_slow"
            task_name="g1_stu_rl_cmg_slow"
            proj_name="g1_stu_rl_cmg_slow"
            ;;
        cmg_teacher_medium*)
            teacher_proj_name="g1_cmg_medium"
            task_name="g1_stu_rl_cmg_medium"
            proj_name="g1_stu_rl_cmg_medium"
            ;;
        cmg_teacher_fast*)
            teacher_proj_name="g1_cmg_fast"
            task_name="g1_stu_rl_cmg_fast"
            proj_name="g1_stu_rl_cmg_fast"
            ;;
        *)
            teacher_proj_name="g1_priv_mimic"
            ;;
    esac
fi

# If teacher project is provided explicitly, align student CMG task with it when applicable.
case "${teacher_proj_name}" in
    g1_cmg_slow)
        task_name="g1_stu_rl_cmg_slow"
        proj_name="g1_stu_rl_cmg_slow"
        ;;
    g1_cmg_medium)
        task_name="g1_stu_rl_cmg_medium"
        proj_name="g1_stu_rl_cmg_medium"
        ;;
    g1_cmg_fast)
        task_name="g1_stu_rl_cmg_fast"
        proj_name="g1_stu_rl_cmg_fast"
        ;;
esac

# Parse local wrapper flags from extra args, and keep the rest for play.py.
filtered_extra_args=()
for arg in "${extra_args[@]}"; do
    case "${arg}" in
        --cmdswitch|--cmd-switch|--cmd_switch)
            cmdswitch_mode=1
            ;;
        *)
            filtered_extra_args+=("${arg}")
            ;;
    esac
done
extra_args=("${filtered_extra_args[@]}")

if [ "${cmdswitch_mode}" = "1" ]; then
    if [ "${teacher_proj_name}" != "g1_cmg_fast" ]; then
        echo "Error: --cmdswitch mode currently supports fast CMG only (teacher_proj_name must be g1_cmg_fast)."
        exit 1
    fi
    task_name="g1_stu_rl_cmg_fast_low_cmdswitch"
fi

echo "Student exptid: ${exptid}"
echo "Teacher exptid: ${teacher_exptid}"
echo "Teacher proj_name: ${teacher_proj_name}"
echo "Student task_name: ${task_name}"
echo "Student proj_name: ${proj_name}"
echo "Cmdswitch mode: ${cmdswitch_mode}"
if [ "${cmdswitch_mode}" = "1" ]; then
    echo "Cmdswitch details: cmd_A and cmd_B sampled in task command range, switch time sampled in [0.1, 1.0]s."
fi

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --teacher_exptid "${teacher_exptid}" \
                --teacher_proj_name "${teacher_proj_name}" \
                --record_video \
                "${extra_args[@]}" \
