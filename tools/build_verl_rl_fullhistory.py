"""
Build Full History RL data: convert existing agentic parquet to single-turn prediction format.

Data source: data/rl_agentic_train.parquet and data/rl_agentic_val.parquet
  - Extract full conversation history (first turn parsed from prompt[1], remaining from interaction_kwargs.turns)
  - User profile is extracted from the system prompt
  - ground_truth reuses reward_model.ground_truth

Difference from the agentic version:
  - prompt contains the full conversation history (all turns concatenated)
  - system prompt has no memory management instructions, direct prediction
  - No interaction_kwargs needed (single-turn, no multi-turn interaction)
  - Used for verl standard single-turn GRPO training (verl.trainer.main_ppo)
"""

import argparse
import os
import re

import pandas as pd

from onepred.locale import (
    FULLHISTORY_SYSTEM_PROMPT_TEMPLATE,
    observation_header,
    observation_user_label,
    observation_response_label,
    feedback_label,
    response_truncation_marker,
    fallback_none,
    fullhistory_conversation_header,
    fullhistory_prediction_instruction,
    parse_query_pattern,
    parse_response_pattern,
    parse_feedback_pattern,
    parse_turn_idx_pattern,
)

# ============================================================
# System prompt is now imported from locale
# ============================================================

MAX_RESPONSE_CHARS = 500


# ============================================================
# Extract information from agentic parquet rows (reusing evaluate_baselines.py logic)
# ============================================================

def _parse_first_turn(user_content: str) -> dict:
    """Parse query, response, and feedback from the first turn user message. Supports both Chinese and English formats."""
    query = ""
    response = ""
    feedback = ""

    query_match = re.search(parse_query_pattern, user_content, re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()

    response_match = re.search(parse_response_pattern, user_content, re.DOTALL)
    if response_match:
        response = response_match.group(1).strip()

    feedback_match = re.search(parse_feedback_pattern, user_content, re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    idx_match = re.search(parse_turn_idx_pattern, user_content)
    if idx_match:
        turn_idx = int(idx_match.group(1) or idx_match.group(2))
    else:
        turn_idx = 1

    return {
        "turn_idx": turn_idx,
        "query": query,
        "response": response,
        "user_feedback": feedback,
    }


def extract_all_turns(row: dict) -> list[dict]:
    """Extract full conversation history from an agentic parquet row."""
    first_user_content = row["prompt"][1]["content"]
    first_turn = _parse_first_turn(first_user_content)

    remaining_turns_raw = row["extra_info"]["interaction_kwargs"].get("turns", [])
    if hasattr(remaining_turns_raw, 'tolist'):
        remaining_turns = remaining_turns_raw.tolist()
    elif isinstance(remaining_turns_raw, dict):
        remaining_turns = list(remaining_turns_raw.values())
    else:
        remaining_turns = list(remaining_turns_raw)

    return [first_turn] + remaining_turns


def extract_user_profile(row: dict) -> str:
    """Extract user profile text from the agentic system prompt."""
    system_content = row["prompt"][0]["content"]
    match = re.search(
        r"Current user profile \(for reference only[^)]*\):\n(.+?)\n\n(?:CRITICAL|Rules|You will)",
        system_content, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return fallback_none


# ============================================================
# Formatting & Conversion
# ============================================================

def format_turn_text(turn: dict, max_response_chars: int | None = None) -> str:
    """Format a single turn of conversation."""
    turn_idx = turn.get("turn_idx", 1)
    query = turn.get("query", "")
    response = turn.get("response", "")

    text = f"{observation_header(turn_idx)}\n{observation_user_label}{query}\n{observation_response_label}{response}"

    fb = turn.get("user_feedback", "")
    if fb and fb.strip():
        text += f"\n{feedback_label}{fb.strip()}"

    return text


def convert_row(row: dict) -> dict:
    """Convert an agentic parquet row to Full History single-turn format."""
    all_turns = extract_all_turns(row)
    user_profile = extract_user_profile(row)
    ground_truth = row["reward_model"]["ground_truth"]

    system_prompt = FULLHISTORY_SYSTEM_PROMPT_TEMPLATE.format(
        user_profile=user_profile if user_profile else fallback_none
    )

    turn_texts = [format_turn_text(t) for t in all_turns]
    user_content = fullhistory_conversation_header + "\n\n" + "\n\n".join(turn_texts)
    user_content += "\n\n" + fullhistory_prediction_instruction

    return {
        "data_source": "onepred",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "ability": "next_query_prediction",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "session_id": row["extra_info"].get("session_id", ""),
            "num_history_turns": row["extra_info"].get("num_history_turns", len(all_turns)),
            "all_queries": [t.get("query", "") for t in all_turns],
        },
    }


# ============================================================
# Main flow
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build Full History RL data from agentic parquet")
    parser.add_argument(
        "--train_input",
        type=str,
        default="./data/rl_agentic_train.parquet",
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default="./data/rl_agentic_val.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
    )
    parser.add_argument("--max_response_chars", type=int, default=MAX_RESPONSE_CHARS)
    args = parser.parse_args()

    # Update module-level constant for format_turn_text default
    import tools.build_verl_rl_fullhistory as _self
    _self.MAX_RESPONSE_CHARS = args.max_response_chars

    os.makedirs(args.output_dir, exist_ok=True)

    for split, input_path in [("train", args.train_input), ("val", args.val_input)]:
        print(f"\n{'='*60}")
        print(f"Processing {split}: {input_path}")
        print(f"{'='*60}")

        df = pd.read_parquet(input_path)
        print(f"Input samples: {len(df)}")

        records = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            records.append(convert_row(row))

        print(f"Output samples: {len(records)}")

        output_path = os.path.join(args.output_dir, f"rl_fullhistory_{split}.parquet")
        pd.DataFrame(records).to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

        # Compute prompt length statistics (character count)
        prompt_lens = []
        for r in records:
            total_chars = sum(len(m["content"]) for m in r["prompt"])
            prompt_lens.append(total_chars)
        prompt_lens.sort()
        n = len(prompt_lens)
        print(f"\nPrompt length (char count, response truncated to {args.max_response_chars}):")
        print(f"  P50: {prompt_lens[n // 2]}")
        print(f"  P90: {prompt_lens[int(n * 0.9)]}")
        print(f"  P95: {prompt_lens[int(n * 0.95)]}")
        print(f"  P99: {prompt_lens[int(n * 0.99)]}")
        print(f"  Max: {prompt_lens[-1]}")

        # Turn distribution statistics
        turn_dist: dict[int, int] = {}
        for r in records:
            nt = r["extra_info"]["num_history_turns"]
            turn_dist[nt] = turn_dist.get(nt, 0) + 1
        print(f"\nTurn distribution:")
        for k in sorted(turn_dist.keys()):
            print(f"  {k} turns: {turn_dist[k]}")

        # Print one sample example
        r0 = records[0]
        print(f"\n--- Sample example ---")
        for msg in r0["prompt"]:
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "...(truncated)"
            print(f"[{msg['role']}] {content}")
        print(f"ground_truth: {r0['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
