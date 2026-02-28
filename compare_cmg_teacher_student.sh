#!/bin/bash
# Compare CMG teacher vs student under identical command settings.
# Usage:
# bash compare_cmg_teacher_student.sh <speed_mode> <student_exptid> <teacher_exptid> [student_checkpoint] [teacher_checkpoint] [cmd_vx] [cmd_vy] [cmd_yaw]
#
# Example:
# bash compare_cmg_teacher_student.sh fast cmg_student_fast_v2 cmg_teacher_fast_v1 -1 -1 3.0 0.0 0.0

set -e

speed_mode=${1:-fast}
student_exptid=${2:-cmg_student_${speed_mode}_v1}
teacher_exptid=${3:-cmg_teacher_${speed_mode}_v1}
student_checkpoint=${4:--1}
teacher_checkpoint=${5:--1}

case "${speed_mode}" in
    slow)
        teacher_proj_name="g1_cmg_slow"
        default_vx="1.0"
        ;;
    medium)
        teacher_proj_name="g1_cmg_medium"
        default_vx="2.0"
        ;;
    fast)
        teacher_proj_name="g1_cmg_fast"
        default_vx="3.0"
        ;;
    *)
        echo "Error: Invalid speed_mode '${speed_mode}'. Must be one of: slow, medium, fast"
        exit 1
        ;;
esac

cmd_vx=${6:-${default_vx}}
cmd_vy=${7:-0.0}
cmd_yaw=${8:-0.0}

echo "============================================"
echo "CMG Teacher/Student Alignment Compare"
echo "============================================"
echo "Speed mode: ${speed_mode}"
echo "Teacher: ${teacher_exptid} (${teacher_proj_name}), checkpoint=${teacher_checkpoint}"
echo "Student: ${student_exptid}, checkpoint=${student_checkpoint}"
echo "Commands: vx=${cmd_vx}, vy=${cmd_vy}, yaw=${cmd_yaw}"
echo "Play mode: deterministic default (pass --eval_randomized to override manually)"
echo "============================================"

echo "[1/2] Running teacher playback..."
bash play_teacher.sh "${teacher_exptid}" "${teacher_proj_name}" \
    --checkpoint "${teacher_checkpoint}" \
    --cmd_vx "${cmd_vx}" \
    --cmd_vy "${cmd_vy}" \
    --cmd_yaw "${cmd_yaw}" \
    --record_log

echo "[2/2] Running student playback..."
bash play_student.sh "${student_exptid}" "${teacher_exptid}" "${teacher_proj_name}" \
    --checkpoint "${student_checkpoint}" \
    --cmd_vx "${cmd_vx}" \
    --cmd_vy "${cmd_vy}" \
    --cmd_yaw "${cmd_yaw}" \
    --record_log

echo "Done. Check outputs under:"
echo "  - legged_gym/logs/videos_retarget/"
echo "  - legged_gym/logs/env_logs/"
