"""
Baseline evaluation script: single-turn inference evaluation supporting multiple input strategies.

Supported methods:
  - current_turn: only the last turn query+response
  - full_concat: all turns directly concatenated

Consistency with training:
  - Data extraction logic is fully consistent with build_verl_rl_agentic.py
  - Observation format reuses OnePredInteraction._format_observation
  - Prediction extraction reuses OnePredInteraction._extract_prediction
  - Output structured_traces/traces.jsonl format, compatible with compute_metrics.py

Usage:
  python -m tools.evaluate_baselines \
      --method current_turn \
      --data data/rl_agentic_val.parquet \
      --model_path os.environ.get("MODEL_DIR", "./models")/Qwen3-8B \
      --gpu 0
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from uuid import uuid4

import pandas as pd
from vllm import LLM, SamplingParams

from tools.build_verl_rl_agentic import format_user_profile, format_user_feedback
from onepred.locale import (
    FULLHISTORY_SYSTEM_PROMPT_TEMPLATE,
    observation_header,
    observation_user_label,
    observation_response_label,
    feedback_label as locale_feedback_label,
    fullhistory_conversation_header,
    fullhistory_prediction_instruction,
    fallback_none,
    parse_query_pattern,
    parse_response_pattern,
    parse_feedback_pattern,
    parse_turn_idx_pattern,
)


# ============================================================
# System Prompt (single-turn baseline, no memory management instructions)
# ============================================================

BASELINE_SYSTEM_PROMPT_TEMPLATE = FULLHISTORY_SYSTEM_PROMPT_TEMPLATE

# Summary method's summarization system prompt
SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation analysis assistant. "
    "Read the following conversation history and extract key information helpful for predicting the user's next question. "
    "Output a concise summary (no more than 300 words), focusing on: user topic shifts, unresolved questions, and conversation direction."
)


# ============================================================
# Logging: dual-write to stdout + file
# ============================================================

class TeeLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ============================================================
# Data extraction: extract full conversation history from parquet rows
# ============================================================

def extract_all_turns(row: dict) -> list[dict]:
    """Extract full conversation history from a parquet or JSONL row.

    Each returned turn contains:
      - turn_idx: int
      - query: str
      - response: str
      - user_feedback: str (may be empty)
    """
    # JSONL mode: has history field directly
    if "history" in row and "prompt" not in row:
        turns = []
        for t in row["history"]:
            # JSONL's turn_idx starts from 1, while format_turn_text will +1 for display,
            # so here -1 to convert to 0-based for consistency
            raw_idx = t.get("turn_idx", 1)
            turns.append({
                "turn_idx": raw_idx - 1 if raw_idx >= 1 else raw_idx,
                "query": t.get("query", ""),
                "response": t.get("response", ""),
                "user_feedback": format_user_feedback(t.get("user_feedback", "")),
            })
        return turns

    # Parquet mode: first turn parsed from prompt[1]
    first_user_content = row["prompt"][1]["content"]
    first_turn = _parse_first_turn(first_user_content)

    # Remaining turns: from extra_info.interaction_kwargs.turns
    remaining_turns_raw = row["extra_info"]["interaction_kwargs"].get("turns", [])
    # pandas/parquet may deserialize as ndarray or dict, need to convert to list
    if hasattr(remaining_turns_raw, 'tolist'):
        remaining_turns = remaining_turns_raw.tolist()
    elif isinstance(remaining_turns_raw, dict):
        remaining_turns = list(remaining_turns_raw.values())
    else:
        remaining_turns = list(remaining_turns_raw)

    all_turns = [first_turn] + remaining_turns
    return all_turns


def _parse_first_turn(user_content: str) -> dict:
    """Parse query and response from the first turn user message.

    Supports both Chinese and English formats (via bilingual regex from locale.py):
      Chinese: [Turn X] User: ... Response: ... User feedback: ... [System Instruction]
      English: [Turn X] User: ... Response: ... User feedback: ... [System Instruction]
    """
    query = ""
    response = ""
    feedback = ""

    # query: use bilingual regex
    query_match = re.search(parse_query_pattern, user_content, re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()

    # response: use bilingual regex
    response_match = re.search(parse_response_pattern, user_content, re.DOTALL)
    if response_match:
        response = response_match.group(1).strip()

    # feedback: use bilingual regex
    feedback_match = re.search(parse_feedback_pattern, user_content, re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    # Extract turn_idx: supports both [Turn X] formats
    idx_match = re.search(parse_turn_idx_pattern, user_content)
    if idx_match:
        # group(1) is the Chinese number, group(2) is the English number
        turn_num = idx_match.group(1) or idx_match.group(2)
        turn_idx = int(turn_num) - 1
    else:
        turn_idx = 0

    return {
        "turn_idx": turn_idx,
        "query": query,
        "response": response,
        "user_feedback": feedback,
    }


def extract_user_profile(row: dict) -> str:
    """Extract user profile text from the system prompt."""
    # JSONL mode: has user_profile field directly
    if "user_profile" in row and "prompt" not in row:
        profile = format_user_profile(row["user_profile"])
        return profile if profile else fallback_none

    system_content = row["prompt"][0]["content"]
    match = re.search(
        r"Current user profile \(for reference only[^)]*\):\n(.+?)\n\n(?:CRITICAL|You will|Each candidate)",
        system_content, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return fallback_none


# ============================================================
# Prompt construction
# ============================================================

def format_turn_text(turn: dict) -> str:
    """Format a single turn of conversation as text (consistent with OnePredInteraction._format_observation)."""
    turn_idx = turn.get("turn_idx", 0) + 1
    query = turn.get("query", "")
    response = turn.get("response", "")

    text = f"{observation_header(turn_idx)}\n{observation_user_label}{query}\n{observation_response_label}{response}"

    feedback = turn.get("user_feedback", "")
    if feedback and feedback.strip():
        text += f"\n{locale_feedback_label}{feedback.strip()}"

    return text


def build_prompt_current_turn(row: dict) -> list[dict]:
    """Method A: take only the last turn."""
    all_turns = extract_all_turns(row)
    user_profile = extract_user_profile(row)
    system_prompt = BASELINE_SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile)

    last_turn = all_turns[-1]
    user_content = format_turn_text(last_turn) + "\n\n" + fullhistory_prediction_instruction

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_prompt_last_k(row: dict, k: int) -> list[dict]:
    """Method B/C: take last k turns."""
    all_turns = extract_all_turns(row)
    user_profile = extract_user_profile(row)
    system_prompt = BASELINE_SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile)

    selected = all_turns[-k:] if len(all_turns) > k else all_turns
    turn_texts = [format_turn_text(t) for t in selected]
    recent_header = "Below is the recent conversation history:"
    user_content = recent_header + "\n\n" + "\n\n".join(turn_texts)
    user_content += "\n\n" + fullhistory_prediction_instruction

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_prompt_full_concat(row: dict) -> list[dict]:
    """Method D: all turns directly concatenated."""
    all_turns = extract_all_turns(row)
    user_profile = extract_user_profile(row)
    system_prompt = BASELINE_SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile)

    turn_texts = [format_turn_text(t) for t in all_turns]
    user_content = fullhistory_conversation_header + "\n\n" + "\n\n".join(turn_texts)
    user_content += "\n\n" + fullhistory_prediction_instruction

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_prompt_summary_stage1(row: dict) -> list[dict]:
    """Method E stage 1: summary prompt."""
    all_turns = extract_all_turns(row)
    turn_texts = [format_turn_text(t) for t in all_turns]
    full_history = "\n\n".join(turn_texts)

    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": full_history},
    ]


def build_prompt_summary_stage2(row: dict, summary: str) -> list[dict]:
    """Method E stage 2: summary + last turn -> prediction."""
    all_turns = extract_all_turns(row)
    user_profile = extract_user_profile(row)
    system_prompt = BASELINE_SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile)

    last_turn = all_turns[-1]
    summary_label = "[Conversation Summary]"
    latest_label = "[Latest Turn]"
    user_content = (
        f"{summary_label}\n{summary}\n\n"
        f"{latest_label}\n"
        f"{observation_user_label}{last_turn.get('query', '')}\n"
        f"{observation_response_label}{last_turn.get('response', '')}\n\n"
        f"{fullhistory_prediction_instruction}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# Prediction extraction (reusing OnePredInteraction logic)
# ============================================================

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_prediction(raw_response: str) -> str:
    """Extract <prediction> content from model output (consistent with OnePredInteraction._extract_prediction)."""
    content_clean = _strip_think(raw_response)
    match = re.search(r"<prediction>(.*?)</prediction>", content_clean, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in content_clean.split("\n") if line.strip()]
    return lines[-1] if lines else content_clean


# ============================================================
# Inference
# ============================================================

def run_single_turn_inference(
    llm: LLM,
    tokenizer,
    messages: list[dict],
    sampling_params: SamplingParams,
    enable_thinking: bool = True,
) -> str:
    """Single-turn inference, returns raw model output."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def run_batch_inference(
    llm: LLM,
    tokenizer,
    all_messages: list[list[dict]],
    sampling_params: SamplingParams,
    enable_thinking: bool = True,
    batch_size: int = 64,
) -> list[str]:
    """Batch inference, returns list of raw model outputs."""
    all_prompts = []
    for messages in all_messages:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        all_prompts.append(prompt_text)

    # Filter overlong prompts: tokenize to check length, skip those exceeding max_model_len
    max_len = llm.llm_engine.model_config.max_model_len
    valid_indices = []
    for idx, prompt in enumerate(all_prompts):
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > max_len:
            print(f"  [SKIP] sample {idx}: prompt length {len(token_ids)} > max_model_len {max_len}")
        else:
            valid_indices.append(idx)
    print(f"  [Filter] {len(valid_indices)}/{len(all_prompts)} samples within max_model_len")

    valid_prompts = [all_prompts[i] for i in valid_indices]
    valid_results = []
    for i in range(0, len(valid_prompts), batch_size):
        batch = valid_prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        for output in outputs:
            valid_results.append(output.outputs[0].text.strip())
        print(f"  [{min(i + batch_size, len(valid_prompts))}/{len(valid_prompts)}] batches done")

    # Reconstruct full results with empty strings for skipped samples
    results = [""] * len(all_prompts)
    for i, valid_idx in enumerate(valid_indices):
        results[valid_idx] = valid_results[i]

    return results


# ============================================================
# Main flow
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OnePred baseline evaluation (single-turn inference)")

    parser.add_argument("--method", type=str, required=True,
                        choices=["current_turn", "last_3", "last_5", "full_concat", "summary"],
                        help="Baseline method")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model path")
    parser.add_argument("--data", type=str, default="data/rl_agentic_val.parquet",
                        help="Evaluation data parquet")

    # Inference parameters (consistent with evaluate.py)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="vLLM batch size for generation")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"outputs/baseline-{args.method}-{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    trace_dir = os.path.join(args.output_dir, "structured_traces")
    os.makedirs(trace_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "eval.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Args] {vars(args)}")

    # ---- Load data ----
    print(f"[Data] {args.data}")
    if args.data.endswith(".jsonl"):
        rows = []
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        is_jsonl = True
    else:
        df = pd.read_parquet(args.data)
        rows = [df.iloc[i].to_dict() for i in range(len(df))]
        is_jsonl = False
    print(f"[Data] Total samples: {len(rows)}")

    # ---- Load model ----
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[GPU] Using GPU {args.gpu}")

    print(f"[vLLM] Loading model (tp={args.tp})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    print(f"[vLLM] Model loaded")

    # ---- Build prompts ----
    print(f"\n[Prompt] Building method={args.method} ...")
    t_start = time.time()

    summaries: list[str] = []  # only used for summary method

    if args.method == "current_turn":
        all_messages = [build_prompt_current_turn(row) for row in rows]
    elif args.method == "last_3":
        all_messages = [build_prompt_last_k(row, k=3) for row in rows]
    elif args.method == "last_5":
        all_messages = [build_prompt_last_k(row, k=5) for row in rows]
    elif args.method == "full_concat":
        all_messages = [build_prompt_full_concat(row) for row in rows]
    else:  # summary
        all_messages = [build_prompt_summary_stage1(row) for row in rows]

    print(f"[Prompt] Building complete, elapsed {time.time() - t_start:.1f}s")

    # ---- Inference ----
    if args.method == "summary":
        # Stage 1: Summary
        print(f"\n[Inference] Stage 1 - Summary ({len(all_messages)} samples) ...")
        t_infer = time.time()
        summary_responses = run_batch_inference(
            llm, tokenizer, all_messages, sampling_params,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
        )
        summaries = [_strip_think(r) for r in summary_responses]
        print(f"[Inference] Stage 1 complete, elapsed {time.time() - t_infer:.1f}s")

        # Stage 2: Prediction
        all_messages_stage2 = [
            build_prompt_summary_stage2(row, summary)
            for row, summary in zip(rows, summaries)
        ]
        print(f"\n[Inference] Stage 2 - Prediction ({len(all_messages_stage2)} samples) ...")
        t_infer2 = time.time()
        raw_responses = run_batch_inference(
            llm, tokenizer, all_messages_stage2, sampling_params,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
        )
        print(f"[Inference] Stage 2 complete, elapsed {time.time() - t_infer2:.1f}s")
    else:
        print(f"\n[Inference] Single-turn inference ({len(all_messages)} samples) ...")
        t_infer = time.time()
        raw_responses = run_batch_inference(
            llm, tokenizer, all_messages, sampling_params,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
        )
        print(f"[Inference] Complete, elapsed {time.time() - t_infer:.1f}s")

    # ---- Extract prediction and save traces ----
    print(f"\n[Post-processing] Extracting predictions & saving traces ...")
    trace_path = os.path.join(trace_dir, "traces.jsonl")
    results = []

    with open(trace_path, "w", encoding="utf-8") as f:
        for i, (row, raw_resp) in enumerate(zip(rows, raw_responses)):
            all_turns = extract_all_turns(row)
            ground_truth = row.get("target", "") if is_jsonl else row["reward_model"]["ground_truth"]
            prediction = extract_prediction(raw_resp)
            model_response_clean = _strip_think(raw_resp)

            # Build trace format consistent with OnePredInteraction._save_trace
            instance_id = uuid4().hex

            # Build turns array (observation + model_response)
            if args.method == "summary":
                stage2_msgs = build_prompt_summary_stage2(row, summaries[i])
                observation = stage2_msgs[1]["content"]
            elif args.method == "current_turn":
                msgs = build_prompt_current_turn(row)
                observation = msgs[1]["content"]
            elif args.method == "last_3":
                msgs = build_prompt_last_k(row, k=3)
                observation = msgs[1]["content"]
            elif args.method == "last_5":
                msgs = build_prompt_last_k(row, k=5)
                observation = msgs[1]["content"]
            else:  # full_concat
                msgs = build_prompt_full_concat(row)
                observation = msgs[1]["content"]

            trace_record = {
                "instance_id": instance_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "reward": 0.0,  # No reward computed during evaluation, unified by compute_metrics.py
                "total_turns": len(all_turns),
                "method": args.method,
                "turns": [{
                    "turn": 0,
                    "observation": observation,
                    "model_response": model_response_clean,
                }],
            }

            f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            results.append(trace_record)

    print(f"[Post-processing] Saved {len(results)} traces -> {trace_path}")

    # ---- Summary ----
    has_pred = sum(1 for r in results if r["prediction"].strip())
    print(f"\n{'='*60}")
    print(f"Baseline evaluation complete: {args.method}")
    print(f"{'='*60}")
    print(f"  Samples: {len(results)}")
    print(f"  Has prediction output: {has_pred}/{len(results)} ({100*has_pred/len(results):.1f}%)")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Traces: {trace_path}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "method": args.method,
        "model": args.model_path,
        "data": args.data,
        "num_samples": len(results),
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "has_prediction_rate": has_pred / len(results),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Next step] python -m tools.compute_metrics {trace_path} --output_dir {args.output_dir}")

    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
