#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)")
NUM_RE = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
REWARD_RE = re.compile(r"Mean reward \(total\):\s*" + NUM_RE)
EP_LEN_RE = re.compile(r"Mean episode length:\s*" + NUM_RE)
TRACK_VEL_RE = re.compile(r"Mean episode (?:rew_)?tracking_cmd_vel:\s*" + NUM_RE)
TRACK_YAW_RE = re.compile(r"Mean episode (?:rew_)?tracking_cmd_yaw:\s*" + NUM_RE)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_metrics(log_path: str):
    metrics = defaultdict(dict)
    current_it = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw)
            m_it = ITER_RE.search(line)
            if m_it:
                current_it = int(m_it.group(1))

            if current_it is None:
                continue

            m = REWARD_RE.search(line)
            if m:
                metrics[current_it]["reward"] = float(m.group(1))

            m = EP_LEN_RE.search(line)
            if m:
                metrics[current_it]["ep_len"] = float(m.group(1))

            m = TRACK_VEL_RE.search(line)
            if m:
                metrics[current_it]["track_vel"] = float(m.group(1))

            m = TRACK_YAW_RE.search(line)
            if m:
                metrics[current_it]["track_yaw"] = float(m.group(1))

    return metrics


def get_series(metrics, key, it_max):
    out = []
    for it in sorted(metrics.keys()):
        if it > it_max:
            break
        if key in metrics[it]:
            out.append((it, metrics[it][key]))
    return out


def avg(values):
    return sum(values) / len(values) if values else None


def first_last_avgs(series, k=20):
    vals = [v for _, v in series]
    if len(vals) < 10:
        return None, None
    left_k = min(k, len(vals))
    right_k = min(k, len(vals))
    return avg(vals[:left_k]), avg(vals[-right_k:])


def gate_report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{name}] {status}: {detail}")


def main():
    parser = argparse.ArgumentParser(description="Check CMG student training gates from a training log.")
    parser.add_argument("--log", required=True, help="Path to train log file")
    parser.add_argument("--gate1_iter", type=int, default=200)
    parser.add_argument("--gate2_iter", type=int, default=600)
    parser.add_argument("--gate3_iter", type=int, default=1000)
    parser.add_argument(
        "--absolute_iter",
        action="store_true",
        default=False,
        help="Evaluate gates against absolute iteration indices instead of relative-to-log-start windows.",
    )
    parser.add_argument("--gate2_ep_len", type=float, default=15.0)
    args = parser.parse_args()

    metrics = parse_metrics(args.log)
    if not metrics:
        print("No iteration metrics found. Check log path/content.")
        return

    max_it = max(metrics.keys())
    min_it = min(metrics.keys())
    print(f"Parsed iterations range: {min_it} -> {max_it}")

    if args.absolute_iter:
        gate1_end = args.gate1_iter
        gate2_end = args.gate2_iter
        gate3_end = args.gate3_iter
    else:
        gate1_end = min_it + args.gate1_iter
        gate2_end = min_it + args.gate2_iter
        gate3_end = min_it + args.gate3_iter

    # Gate-1: early reward should stay positive and not trend down.
    reward_series = get_series(metrics, "reward", gate1_end)
    g1_first, g1_last = first_last_avgs(reward_series)
    if g1_first is None:
        gate_report("Gate-1", False, f"insufficient reward points before iter {gate1_end}")
    else:
        g1_ok = (g1_last > 0.0) and (g1_last >= g1_first - 1e-3)
        gate_report(
            "Gate-1",
            g1_ok,
            f"iter<={gate1_end} reward first_avg={g1_first:.4f}, last_avg={g1_last:.4f}"
        )

    # Gate-2: episode length should clearly rise above 7~11 baseline.
    ep_series = get_series(metrics, "ep_len", gate2_end)
    g2_first, g2_last = first_last_avgs(ep_series)
    if g2_first is None:
        gate_report("Gate-2", False, f"insufficient ep_len points before iter {gate2_end}")
    else:
        g2_ok = g2_last >= args.gate2_ep_len
        gate_report(
            "Gate-2",
            g2_ok,
            f"iter<={gate2_end} ep_len first_avg={g2_first:.2f}, last_avg={g2_last:.2f}, threshold={args.gate2_ep_len:.2f}"
        )

    # Gate-3: command tracking should not stagnate.
    vel_series = get_series(metrics, "track_vel", gate3_end)
    yaw_series = get_series(metrics, "track_yaw", gate3_end)
    v_first, v_last = first_last_avgs(vel_series)
    y_first, y_last = first_last_avgs(yaw_series)
    if v_first is None or y_first is None:
        gate_report("Gate-3", False, f"insufficient tracking points before iter {gate3_end}")
    else:
        g3_ok = (v_last > v_first + 1e-3) and (y_last > y_first + 5e-4)
        gate_report(
            "Gate-3",
            g3_ok,
            f"iter<={gate3_end} track_vel {v_first:.4f}->{v_last:.4f}, track_yaw {y_first:.4f}->{y_last:.4f}"
        )


if __name__ == "__main__":
    main()
