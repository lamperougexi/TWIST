#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <run_dir> [cmd_template]"
  echo "Example (default mode, parse train logs):"
  echo "  $0 /root/TWIST-jiarui/legged_gym/logs/g1_stu_rl_cmg_fast/cmg_student_fast_v2_low"
  echo "Example (run same custom cmd for every checkpoint):"
  echo "  $0 /path/to/run 'python eval.py --checkpoint {checkpoint}'"
  echo "Supported placeholders in cmd_template: {checkpoint} {model_path} {run_dir}"
  exit 1
fi

RUN_DIR="$1"
CMD_TEMPLATE="${2:-}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Error: run_dir does not exist: ${RUN_DIR}" >&2
  exit 1
fi

PY_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "Error: neither 'python' nor 'python3' found in PATH." >&2
    exit 1
  fi
fi

"${PY_BIN}" - "${RUN_DIR}" "${CMD_TEMPLATE}" <<'PY'
import glob
import os
import re
import subprocess
import sys
from typing import Dict, Optional

run_dir = sys.argv[1]
cmd_template = sys.argv[2] if len(sys.argv) > 2 else ""

model_paths = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")))
if not model_paths:
    print(f"Error: no model_*.pt found in {run_dir}", file=sys.stderr)
    sys.exit(1)

ansi_re = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
iter_re = re.compile(r"Learning iteration\s+(\d+)")
num_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
nan_or_num_re = r"(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN)"

ckpt_re = re.compile(r"model_(\d+)\.pt$")
reward_re = re.compile(r"Mean reward \(total\):\s*" + num_re, re.IGNORECASE)
ep_len_re = re.compile(r"Mean episode length:\s*" + num_re, re.IGNORECASE)

reward_generic_re = re.compile(
    rf"(?:[\"']?(?:mean_reward|reward_mean|reward)[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
ep_len_generic_re = re.compile(
    rf"(?:[\"']?(?:ep_len_mean|ep_len|episode[_ ]length)[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
success_rate_re = re.compile(
    rf"(?:[\"']?(?:success(?:[_ ]?rate)?|succ_rate)[\"']?)\s*[:=]\s*({nan_or_num_re})\s*%?",
    re.IGNORECASE,
)
fall_rate_re = re.compile(
    rf"(?:[\"']?(?:fall(?:[_ ]?rate)?|fail(?:ure)?(?:[_ ]?rate)?)[\"']?)\s*[:=]\s*({nan_or_num_re})\s*%?",
    re.IGNORECASE,
)
track_vel_rmse_re = re.compile(
    rf"(?:[\"']?(?:track(?:ing)?[_ ]?(?:cmd[_ ]?)?vel(?:ocity)?[_ ]?(?:rmse|err(?:or)?)|vel(?:ocity)?[_ ]?(?:rmse|err(?:or)?))[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
track_yaw_rmse_re = re.compile(
    rf"(?:[\"']?(?:track(?:ing)?[_ ]?(?:cmd[_ ]?)?yaw(?:[_ ]?rate)?[_ ]?(?:rmse|err(?:or)?)|yaw(?:[_ ]?rate)?[_ ]?(?:rmse|err(?:or)?))[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
track_vel_re = re.compile(
    rf"(?:Mean episode\s+)?(?:[\"']?(?:rew_)?tracking_cmd_vel[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
track_yaw_re = re.compile(
    rf"(?:Mean episode\s+)?(?:[\"']?(?:rew_)?tracking_cmd_yaw[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
track_vel_generic_re = re.compile(
    rf"(?:[\"']?(?:track(?:ing)?[_ ]?vel(?:ocity)?|track_vel)[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
track_yaw_generic_re = re.compile(
    rf"(?:[\"']?(?:track(?:ing)?[_ ]?yaw(?:[_ ]?rate)?|track_yaw)[\"']?)\s*[:=]\s*({nan_or_num_re})",
    re.IGNORECASE,
)
mean_episode_key_re = re.compile(
    rf"Mean episode\s+([A-Za-z0-9_./ -]+):\s*({nan_or_num_re})",
    re.IGNORECASE,
)


def parse_num(text: str) -> Optional[float]:
    t = text.strip()
    if t.lower() == "nan":
        return None
    try:
        return float(t)
    except Exception:
        return None


def normalize_ratio(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if v > 1.0:
        if v <= 100.0:
            v = v / 100.0
        else:
            return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def scan_last_value(text: str, pattern: re.Pattern) -> Optional[float]:
    out = None
    for m in pattern.finditer(text):
        out = parse_num(m.group(1))
    return out


def parse_metrics_from_text(text: str, ckpt: int) -> Dict[str, Optional[float]]:
    text = ansi_re.sub("", text)
    row = {
        "reward": None,
        "ep_len": None,
        "success_rate": None,
        "track_vel": None,
        "track_yaw": None,
        "track_vel_rmse": None,
        "track_yaw_rmse": None,
    }

    # Prefer CSV rows like:
    # checkpoint,reward,ep_len,success_rate,track_vel,track_yaw,track_vel_rmse,track_yaw_rmse
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            row_ckpt = int(parts[0])
        except Exception:
            continue
        if row_ckpt != ckpt:
            continue
        row["reward"] = parse_num(parts[1])
        row["ep_len"] = parse_num(parts[2])
        if len(parts) > 3:
            row["success_rate"] = normalize_ratio(parse_num(parts[3]))
        if len(parts) > 4:
            row["track_vel"] = parse_num(parts[4])
        if len(parts) > 5:
            row["track_yaw"] = parse_num(parts[5])
        if len(parts) > 6:
            row["track_vel_rmse"] = parse_num(parts[6])
        if len(parts) > 7:
            row["track_yaw_rmse"] = parse_num(parts[7])

    if row["reward"] is None:
        row["reward"] = scan_last_value(text, reward_re)
    if row["reward"] is None:
        row["reward"] = scan_last_value(text, reward_generic_re)

    if row["ep_len"] is None:
        row["ep_len"] = scan_last_value(text, ep_len_re)
    if row["ep_len"] is None:
        row["ep_len"] = scan_last_value(text, ep_len_generic_re)

    if row["success_rate"] is None:
        row["success_rate"] = normalize_ratio(scan_last_value(text, success_rate_re))
    if row["success_rate"] is None:
        fall_rate = normalize_ratio(scan_last_value(text, fall_rate_re))
        if fall_rate is not None:
            row["success_rate"] = 1.0 - fall_rate

    if row["track_vel_rmse"] is None:
        row["track_vel_rmse"] = scan_last_value(text, track_vel_rmse_re)
    if row["track_yaw_rmse"] is None:
        row["track_yaw_rmse"] = scan_last_value(text, track_yaw_rmse_re)

    if row["track_vel"] is None:
        row["track_vel"] = scan_last_value(text, track_vel_re)
    if row["track_vel"] is None:
        row["track_vel"] = scan_last_value(text, track_vel_generic_re)

    if row["track_yaw"] is None:
        row["track_yaw"] = scan_last_value(text, track_yaw_re)
    if row["track_yaw"] is None:
        row["track_yaw"] = scan_last_value(text, track_yaw_generic_re)

    for m in mean_episode_key_re.finditer(text):
        key = m.group(1).strip().lower()
        value = parse_num(m.group(2))
        if value is None:
            continue
        if row["success_rate"] is None and "success" in key:
            row["success_rate"] = normalize_ratio(value)
        if "tracking_cmd_vel" in key or key == "track_vel":
            if "rmse" in key or "err" in key:
                row["track_vel_rmse"] = value
            elif row["track_vel"] is None:
                row["track_vel"] = value
        if "tracking_cmd_yaw" in key or key == "track_yaw":
            if "rmse" in key or "err" in key:
                row["track_yaw_rmse"] = value
            elif row["track_yaw"] is None:
                row["track_yaw"] = value

    return row


def parse_line_metrics(line: str, row: Dict[str, Optional[float]]) -> None:
    m = reward_re.search(line)
    if m:
        row["reward"] = parse_num(m.group(1))

    m = ep_len_re.search(line)
    if m:
        row["ep_len"] = parse_num(m.group(1))

    m = success_rate_re.search(line)
    if m:
        row["success_rate"] = normalize_ratio(parse_num(m.group(1)))

    m = fall_rate_re.search(line)
    if m and row.get("success_rate") is None:
        fall_rate = normalize_ratio(parse_num(m.group(1)))
        if fall_rate is not None:
            row["success_rate"] = 1.0 - fall_rate

    m = track_vel_rmse_re.search(line)
    if m:
        row["track_vel_rmse"] = parse_num(m.group(1))

    m = track_yaw_rmse_re.search(line)
    if m:
        row["track_yaw_rmse"] = parse_num(m.group(1))

    m = track_vel_re.search(line)
    if m and row.get("track_vel") is None:
        row["track_vel"] = parse_num(m.group(1))

    m = track_yaw_re.search(line)
    if m and row.get("track_yaw") is None:
        row["track_yaw"] = parse_num(m.group(1))

    m = track_vel_generic_re.search(line)
    if m and row.get("track_vel") is None:
        row["track_vel"] = parse_num(m.group(1))

    m = track_yaw_generic_re.search(line)
    if m and row.get("track_yaw") is None:
        row["track_yaw"] = parse_num(m.group(1))

    m = mean_episode_key_re.search(line)
    if m:
        key = m.group(1).strip().lower()
        value = parse_num(m.group(2))
        if value is None:
            return
        if row.get("success_rate") is None and "success" in key:
            row["success_rate"] = normalize_ratio(value)
        if "tracking_cmd_vel" in key or key == "track_vel":
            if "rmse" in key or "err" in key:
                row["track_vel_rmse"] = value
            elif row.get("track_vel") is None:
                row["track_vel"] = value
        if "tracking_cmd_yaw" in key or key == "track_yaw":
            if "rmse" in key or "err" in key:
                row["track_yaw_rmse"] = value
            elif row.get("track_yaw") is None:
                row["track_yaw"] = value


def collect_from_logs() -> Dict[int, Dict[str, Optional[float]]]:
    train_logs = sorted(glob.glob(os.path.join(run_dir, "train_*.log")))
    if not train_logs:
        print(f"Error: no train_*.log found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    metrics_local: Dict[int, Dict[str, Optional[float]]] = {}
    for log_path in train_logs:
        current_it = None
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = ansi_re.sub("", raw)
                m_it = iter_re.search(line)
                if m_it:
                    current_it = int(m_it.group(1))
                    metrics_local.setdefault(
                        current_it,
                        {
                            "reward": None,
                            "ep_len": None,
                            "success_rate": None,
                            "track_vel": None,
                            "track_yaw": None,
                            "track_vel_rmse": None,
                            "track_yaw_rmse": None,
                        },
                    )
                if current_it is None:
                    continue
                parse_line_metrics(line, metrics_local[current_it])
    return metrics_local


def compute_scores(
    metrics: Dict[int, Dict[str, Optional[float]]],
    checkpoints,
) -> None:
    weights = {
        "reward": 1.0,
        "ep_len": 1.0,
        "success_rate": 2.0,
        "track_vel": 1.0,
        "track_yaw": 1.0,
        "track_vel_rmse": 1.0,
        "track_yaw_rmse": 1.0,
    }
    maximize_keys = {"reward", "ep_len", "success_rate", "track_vel", "track_yaw"}
    minimize_keys = {"track_vel_rmse", "track_yaw_rmse"}

    normalized: Dict[str, Dict[int, float]] = {}

    for key in list(maximize_keys) + list(minimize_keys):
        values: Dict[int, float] = {}
        for ckpt in checkpoints:
            row = metrics.get(ckpt, {})
            v = row.get(key)
            if v is not None:
                values[ckpt] = float(v)
        if not values:
            continue
        lo = min(values.values())
        hi = max(values.values())
        span = hi - lo
        normalized[key] = {}
        for ckpt, v in values.items():
            if span <= 1e-12:
                n = 1.0
            elif key in maximize_keys:
                n = (v - lo) / span
            else:
                n = (hi - v) / span
            normalized[key][ckpt] = n

    for ckpt in checkpoints:
        row = metrics.setdefault(ckpt, {})
        w_sum = 0.0
        s_sum = 0.0
        for key, w in weights.items():
            nk = normalized.get(key, {})
            if ckpt not in nk:
                continue
            s_sum += w * nk[ckpt]
            w_sum += w
        score = (100.0 * s_sum / w_sum) if w_sum > 0 else None
        row["score"] = score
        row["sorce"] = score


def fmt(v: Optional[float]) -> str:
    return f"{v:.6f}" if v is not None else "NA"


checkpoints = []
ckpt_to_model = {}
for p in model_paths:
    m = ckpt_re.search(os.path.basename(p))
    if m:
        ckpt = int(m.group(1))
        checkpoints.append(ckpt)
        ckpt_to_model[ckpt] = p
checkpoints = sorted(set(checkpoints))

metrics: Dict[int, Dict[str, Optional[float]]] = {}
if cmd_template:
    for ckpt in checkpoints:
        model_path = ckpt_to_model[ckpt]
        cmd = (
            cmd_template
            .replace("{checkpoint}", str(ckpt))
            .replace("{model_path}", model_path)
            .replace("{run_dir}", run_dir)
        )
        env = os.environ.copy()
        env["CMG_CHECKPOINT"] = str(ckpt)
        env["CMG_MODEL_PATH"] = model_path
        env["CMG_RUN_DIR"] = run_dir
        proc = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, env=env)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        metrics[ckpt] = parse_metrics_from_text(text, ckpt)
        if proc.returncode != 0:
            print(
                f"[WARN] checkpoint={ckpt} cmd exit={proc.returncode}; metrics may be NA",
                file=sys.stderr,
            )
else:
    metrics = collect_from_logs()

for ckpt in checkpoints:
    metrics.setdefault(
        ckpt,
        {
            "reward": None,
            "ep_len": None,
            "success_rate": None,
            "track_vel": None,
            "track_yaw": None,
            "track_vel_rmse": None,
            "track_yaw_rmse": None,
        },
    )

compute_scores(metrics, checkpoints)

print(
    "checkpoint,reward,ep_len,success_rate,track_vel,track_yaw,track_vel_rmse,track_yaw_rmse,score,sorce"
)
for ckpt in checkpoints:
    row = metrics.get(ckpt, {})
    print(
        f"{ckpt},{fmt(row.get('reward'))},{fmt(row.get('ep_len'))},{fmt(row.get('success_rate'))},"
        f"{fmt(row.get('track_vel'))},{fmt(row.get('track_yaw'))},{fmt(row.get('track_vel_rmse'))},"
        f"{fmt(row.get('track_yaw_rmse'))},{fmt(row.get('score'))},{fmt(row.get('sorce'))}"
    )
PY
