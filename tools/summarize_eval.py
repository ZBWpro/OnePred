"""
Evaluation post-processing: generate summary.json + report.txt from val_traces + structured_traces.

Usage:
  python -m tools.summarize_eval <output_dir> [--step_name global_step_300]

Reads:
  - <output_dir>/val_traces/*.jsonl   (score and other evaluation metrics)
  - <output_dir>/structured_traces/traces.jsonl  (turn information)
Generates:
  - <output_dir>/summary.json
  - <output_dir>/report.txt
"""

import argparse
import json
import os
import re
import sys
from collections import Counter


def _load_jsonl_dir(dirpath: str) -> list[dict]:
    """Read all jsonl files in a directory."""
    records = []
    if not os.path.isdir(dirpath):
        return records
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_eval_data(output_dir: str) -> list[dict]:
    """Load evaluation data, merging val_traces scores with structured_traces turn information.

    structured_traces: len(turns) is the accurate turn count.
    val_traces: contains score and other metrics.
    The two correspond one-to-one in order.
    """
    val_traces = _load_jsonl_dir(os.path.join(output_dir, "val_traces"))
    structured_traces = _load_jsonl_dir(os.path.join(output_dir, "structured_traces"))

    if not val_traces:
        print(f"[Error] val_traces directory is empty or does not exist: {output_dir}/val_traces/")
        sys.exit(1)

    # If structured_traces and val_traces have the same count, use structured_traces turn count
    if len(structured_traces) == len(val_traces):
        for vt, st in zip(val_traces, structured_traces):
            vt["total_turns"] = len(st["turns"])
    else:
        # fallback: infer turns from max N in [Turn N] patterns in val_traces input
        print(f"[Warning] structured_traces ({len(structured_traces)}) and "
              f"val_traces ({len(val_traces)}) count mismatch, inferring turns from input")
        for vt in val_traces:
            matches = re.findall(r"\[Turn (\d+)\]", vt.get("input", ""))
            if matches:
                # [Turn N]: N-1 = observed turns count, plus the first turn = N
                # But in val_traces, [Turn 2] actually corresponds to a 1-turn interaction sample
                # So max(N) - 1 = actual turn count
                vt["total_turns"] = max(int(m) for m in matches) - 1
            else:
                vt["total_turns"] = 1

    return val_traces


def generate_summary_and_report(records: list[dict], output_dir: str,
                                 step_name: str, eval_data: str):
    """Generate summary.json and report.txt."""
    n = len(records)
    if n == 0:
        print("[Warning] No evaluation records!")
        return

    scores = [r["score"] for r in records if r.get("score") is not None]
    avg_reward = sum(scores) / len(scores) if scores else 0.0
    score_dist = Counter(scores)

    # Group by turns (using total_turns)
    turn_scores: dict[int, list[float]] = {}
    for r in records:
        t = r.get("total_turns", 1)
        if r.get("score") is not None:
            turn_scores.setdefault(t, []).append(r["score"])

    turn_avg = {t: sum(s) / len(s) for t, s in sorted(turn_scores.items())}
    turn_counts = {t: len(s) for t, s in sorted(turn_scores.items())}

    # ---- summary.json ----
    summary = {
        "checkpoint": step_name,
        "data": eval_data,
        "num_samples": n,
        "avg_reward": round(avg_reward, 4),
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
        "score_distribution_pct": {
            str(k): round(100 * v / len(scores), 1)
            for k, v in sorted(score_dist.items())
        },
        "turn_avg_scores": {str(t): round(v, 4) for t, v in sorted(turn_avg.items())},
        "turn_sample_counts": {str(t): v for t, v in sorted(turn_counts.items())},
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Summary] summary.json -> {summary_path}")

    # ---- report.txt ----
    lines = []
    lines.append("=" * 70)
    lines.append("OnePred Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Checkpoint: {step_name}")
    lines.append(f"Data: {eval_data}")
    lines.append(f"Samples: {n}")
    lines.append(f"Average reward: {avg_reward:.4f}")
    lines.append("")
    lines.append("--- Score distribution ---")
    for s in sorted(score_dist.keys()):
        count = score_dist[s]
        pct = 100 * count / len(scores)
        bar = "█" * int(pct / 2)
        lines.append(f"  {s:.2f}: {count:3d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("--- Grouped by turns ---")
    lines.append(f"  {'Turns':>6s}  {'AvgScr':>6s}  {'Count':>5s}")
    for t in sorted(turn_avg.keys()):
        lines.append(f"  {t:6d}  {turn_avg[t]:.4f}  {turn_counts[t]:5d}")

    lines.append("=" * 70)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Summary] report.txt  -> {report_path}")

    # Also print to stdout
    print()
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluation post-processing: generate summary.json + report.txt")
    parser.add_argument("output_dir", type=str, help="Evaluation output directory (containing val_traces/)")
    parser.add_argument("--step_name", type=str, default=None,
                        help="Checkpoint step name (default: extracted from directory name)")
    parser.add_argument("--eval_data", type=str, default="data/eval_test_1000.parquet",
                        help="Evaluation data path")
    args = parser.parse_args()

    # Auto-extract step_name
    if args.step_name is None:
        dirname = os.path.basename(args.output_dir.rstrip("/"))
        # eval-global_step_300-20260324_155758 -> global_step_300
        parts = dirname.split("-")
        if len(parts) >= 3:
            args.step_name = "-".join(parts[1:-1])  # global_step_300
        else:
            args.step_name = dirname

    print(f"[Summary] Output directory: {args.output_dir}")
    print(f"[Summary] Step: {args.step_name}")

    records = load_eval_data(args.output_dir)
    print(f"[Summary] Loaded {len(records)} records")

    generate_summary_and_report(records, args.output_dir, args.step_name, args.eval_data)


if __name__ == "__main__":
    main()
