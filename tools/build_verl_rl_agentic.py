"""
Build Agentic RL data: convert JSONL data into per-turn interaction format parquet.

Supports two modes:
  1. Single-file mode: --input specifies one JSONL, auto-split into train/val
  2. Dual-file mode: --train_input + --test_input specify train/test JSONL separately

Core design:
  - prompt contains only system (with user profile) + first turn observation
  - Remaining turns are stored in extra_info.interaction_kwargs.turns, fed by Interaction step by step
  - interaction_kwargs.name = "onepred" routes to OnePredInteraction
  - Each turn observation contains query, response, user_feedback (if any)
"""

import argparse
import json
import os
import random

import pandas as pd

from onepred.locale import (
    SYSTEM_PROMPT_TEMPLATE,
    observation_header,
    observation_user_label,
    observation_response_label,
    feedback_label,
    feedback_map,
    feedback_prefix,
    feedback_joiner,
    profile_long_term_label,
    profile_short_term_label,
    system_instruction,
    fallback_none,
    response_truncation_marker,
)

# Max characters for response in the first turn observation, consistent with interaction
MAX_RESPONSE_CHARS = 500

# Filter out samples where any turn's query exceeds this length (English data may have very long queries)
MAX_QUERY_CHARS = 5000

# Filter out samples with too many turns (to avoid OOM)
MAX_TURNS = 15

# Max characters for target
MAX_TARGET_CHARS = 100


def format_user_profile(raw_profile) -> str:
    """Convert user_profile to readable text. Supports both str and dict input formats."""
    if isinstance(raw_profile, str):
        return raw_profile.strip()
    if not isinstance(raw_profile, dict):
        return ""

    parts = []
    ltm = raw_profile.get("long_term_memory", "")
    if ltm and isinstance(ltm, str) and ltm.strip():
        parts.append(f"{profile_long_term_label}\n{ltm.strip()}")
    stm = raw_profile.get("short_term_memory", "")
    if stm and isinstance(stm, str) and stm.strip():
        parts.append(f"{profile_short_term_label}\n{stm.strip()}")
    return "\n\n".join(parts) if parts else ""


def format_user_feedback(raw_feedback) -> str:
    """Convert user_feedback to readable text. Supports both str and dict input formats."""
    if isinstance(raw_feedback, str):
        return raw_feedback.strip()
    if not isinstance(raw_feedback, dict):
        return ""

    labels = []
    for key, label in feedback_map.items():
        if raw_feedback.get(key):
            labels.append(label)
    return feedback_prefix + feedback_joiner.join(labels) if labels else ""


def build_first_observation(turn: dict, is_only_turn: bool) -> str:
    """Build the user message for the first turn observation."""
    turn_idx = turn.get("turn_idx", 1)
    query = turn.get("query", "")
    response = turn.get("response", "")

    text = f"{observation_header(turn_idx)}\n{observation_user_label}{query}\n{observation_response_label}{response}"

    feedback = format_user_feedback(turn.get("user_feedback", ""))
    if feedback:
        text += f"\n{feedback_label}{feedback}"

    if is_only_turn:
        text += system_instruction

    return text


def convert_sample(sample: dict) -> dict | None:
    """Convert a single sample to the format required for agentic RL.

    Note: history/target filtering is already done in _load_and_convert.
    """
    history = sample.get("history", [])

    if not history:
        return None

    target = sample.get("target", "").strip()
    if not target:
        return None

    user_profile = format_user_profile(sample.get("user_profile", ""))
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile if user_profile else fallback_none)

    first_turn = history[0]
    is_only_turn = len(history) == 1

    # Remaining turns (turn 1 ~ n-1) stored in interaction_kwargs
    remaining_turns = []
    for t in history[1:]:
        resp = t.get("response", "")
        remaining_turns.append({
            "turn_idx": t["turn_idx"],
            "query": t.get("query", ""),
            "response": resp,
            "user_feedback": format_user_feedback(t.get("user_feedback", "")),
        })

    return {
        "data_source": "onepred",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_first_observation(first_turn, is_only_turn)},
        ],
        "ability": "next_query_prediction",
        "reward_model": {
            "style": "rule",
            "ground_truth": target,
        },
        "extra_info": {
            "session_id": sample["session_id"],
            "pivot_turn_idx": sample.get("pivot_turn_idx", 0),
            "num_history_turns": sample["num_history_turns"],
            "target_source": sample.get("target_source", ""),
            "interaction_kwargs": {
                "name": "onepred",
                "turns": remaining_turns,
                "ground_truth": target,
                "all_queries": [t.get("query", "") for t in history],
            },
        },
    }


def _load_and_convert(input_path: str, max_turns: int, max_target_chars: int) -> list[dict]:
    """Load JSONL and convert samples, returning valid records."""
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  Raw samples: {len(samples)}")

    records = []
    skipped_turns = 0
    skipped_target = 0
    skipped_empty = 0
    skipped_query = 0
    for s in samples:
        if not s.get("history"):
            skipped_empty += 1
            continue
        if len(s.get("history", [])) > max_turns:
            skipped_turns += 1
            continue
        target = s.get("target", "").strip()
        if not target or len(target) > max_target_chars:
            skipped_target += 1
            continue
        # Filter out samples with overly long queries in any turn
        if any(len(t.get("query", "")) > MAX_QUERY_CHARS for t in s["history"]):
            skipped_query += 1
            continue
        r = convert_sample(s)
        if r is not None:
            records.append(r)
    print(f"  Valid: {len(records)}")
    print(f"  Skipped - turns>{max_turns}: {skipped_turns}, target empty or >{max_target_chars} chars: {skipped_target}, empty history: {skipped_empty}, query>{MAX_QUERY_CHARS}: {skipped_query}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Build Agentic RL data (parquet)")
    # Single-file mode
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single JSONL file, auto-split to train/val",
    )
    # Dual-file mode
    parser.add_argument("--train_input", type=str, default=None, help="Train JSONL (dual-file mode)")
    parser.add_argument("--test_input", type=str, default=None, help="Test/val JSONL (dual-file mode)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_size", type=int, default=50, help="Fixed val size (single-file mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max_target_chars", type=int, default=MAX_TARGET_CHARS)
    args = parser.parse_args()

    if args.train_input and args.test_input:
        # Dual-file mode
        print(f"[Dual-file mode]")
        print(f"Train: {args.train_input}")
        train_records = _load_and_convert(args.train_input, args.max_turns, args.max_target_chars)
        print(f"Val: {args.test_input}")
        val_records = _load_and_convert(args.test_input, args.max_turns, args.max_target_chars)
    else:
        # Single-file mode
        input_path = args.input or os.path.join(os.environ.get("BASE_DIR", "."), "data/next_query_keep_dedup_train.jsonl")
        print(f"[Single-file mode] {input_path}")
        records = _load_and_convert(input_path, args.max_turns, args.max_target_chars)

        random.seed(args.seed)
        random.shuffle(records)
        val_n = min(args.val_size, len(records) - 1) if args.val_size else max(1, int(len(records) * args.val_ratio))
        val_records = records[:val_n]
        train_records = records[val_n:]

    print(f"\nTrain: {len(train_records)}, Val: {len(val_records)}")

    # Turn distribution
    turn_dist: dict[int, int] = {}
    for r in train_records:
        n = r["extra_info"]["num_history_turns"]
        turn_dist[n] = turn_dist.get(n, 0) + 1
    print("\nTurn distribution (train):")
    for k in sorted(turn_dist.keys()):
        print(f"  {k} turns: {turn_dist[k]}")

    # Save
    train_path = os.path.join(args.output_dir, "rl_agentic_train.parquet")
    val_path = os.path.join(args.output_dir, "rl_agentic_val.parquet")
    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)
    print(f"\nSaved:\n  {train_path}\n  {val_path}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    df = pd.read_parquet(train_path)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    row = df.iloc[0]
    print(f"prompt roles: {[m['role'] for m in row['prompt']]}")
    print(f"ground_truth: {row['reward_model']['ground_truth'][:80]}")
    interaction_kwargs = row["extra_info"]["interaction_kwargs"]
    print(f"interaction name: {interaction_kwargs['name']}")
    print(f"remaining turns: {len(interaction_kwargs['turns'])}")

    # Print one multi-turn sample's full structure
    for i, row in df.iterrows():
        if len(row["extra_info"]["interaction_kwargs"]["turns"]) > 0:
            print(f"\n--- Multi-turn sample example (index={i}) ---")
            for msg in row["prompt"]:
                content = msg["content"]
                if len(content) > 500:
                    content = content[:500] + "...(truncated)"
                print(f"[{msg['role']}] {content}")
            turns = row["extra_info"]["interaction_kwargs"]["turns"]
            print(f"remaining turns: {len(turns)}")
            for t in turns[:2]:
                fb_str = f" feedback={t['user_feedback']}" if t.get('user_feedback') else ""
                print(f"  turn {t['turn_idx']}: query={t['query'][:50]}{fb_str}")
            if len(turns) > 2:
                print(f"  ... ({len(turns) - 2} more)")
            print(f"ground_truth: {row['reward_model']['ground_truth']}")
            break


if __name__ == "__main__":
    main()
