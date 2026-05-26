"""
Evaluation script: multi-turn interactive inference + LLM Judge scoring.

The inference pipeline directly reuses the training modules to ensure full consistency with RecurrentMemoryAgentLoop.

Supports --start/--end slice parameters for parallel runner splitting.

Usage:
  # Single-process evaluation
  python -m tools.evaluate \
      --model_path /path/to/hf_model \
      --data data/eval_test_1000.parquet

  # Parallel mode (called by run_eval_parallel.sh)
  python -m tools.evaluate \
      --model_path /path/to/hf_model \
      --data data/eval_test_1000.parquet \
      --start 0 --end 125 --gpu 0 \
      --output_dir outputs/eval_xxx/shard_0
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from vllm import LLM, SamplingParams

# ============================================================
# Directly reuse training modules (ensuring fully consistent inference pipeline)
# ============================================================
from onepred.recurrent_agent_loop import (
    MEMORY_USER_TEMPLATE,
    _extract_memory,
    MAX_MEMORY_CHARS,
    MAX_MEMORY_TOKENS,
)
from onepred.interaction import (
    OnePredInteraction,
    MAX_RESPONSE_CHARS,
)
from onepred.judges.llm_judge_multi import llm_judge_score


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
# Helper functions: directly call training module implementations
# ============================================================

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ============================================================
# FSDP Checkpoint Merging
# ============================================================

def merge_fsdp_checkpoint(ckpt_dir: str, target_dir: str) -> str:
    """Merge FSDP sharded checkpoint into HuggingFace format."""
    if os.path.exists(os.path.join(target_dir, "config.json")):
        has_weights = any(
            f.endswith((".safetensors", ".bin"))
            for f in os.listdir(target_dir)
            if "model" in f
        )
        if has_weights:
            print(f"[Merge] Already have merged model: {target_dir}, skipping merge")
            return target_dir

    print(f"[Merge] Starting FSDP checkpoint merge...")
    print(f"  Input: {ckpt_dir}")
    print(f"  Output: {target_dir}")

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=ckpt_dir,
        hf_model_config_path=os.path.join(ckpt_dir, "huggingface"),
        target_dir=target_dir,
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

    print(f"[Merge] Done! Model saved to: {target_dir}")
    return target_dir


# ============================================================
# Multi-turn inference (reusing training module functions, consistent with RecurrentMemoryAgentLoop.run)
# ============================================================

def run_multiturn_inference(
    llm: LLM,
    tokenizer,
    row: dict,
    sampling_params: SamplingParams,
    enable_thinking: bool = True,
) -> dict:
    """Execute full multi-turn interactive inference for a single sample.

    Inference logic is fully consistent with RecurrentMemoryAgentLoop.run():
    - Each turn constructs fresh_messages = [system, user(memory + observation)]
    - Memory is extracted and truncated by recurrent_agent_loop._extract_memory
    - Observations are formatted by OnePredInteraction._format_observation
    - Predictions are extracted by OnePredInteraction._extract_prediction
    """
    system_content = row["prompt"][0]["content"]
    first_user_content = row["prompt"][1]["content"]
    remaining_turns = row["extra_info"]["interaction_kwargs"]["turns"]
    ground_truth = row["reward_model"]["ground_truth"]
    total_remaining = len(remaining_turns)

    # Reuse OnePredInteraction static methods
    interaction_cls = OnePredInteraction

    memory = ""
    initial_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": first_user_content},
    ]

    per_turn_responses = []
    has_memory_per_turn = []
    has_prediction_per_turn = []

    # Create temporary interaction instance for formatting observations
    _interaction = interaction_cls.__new__(interaction_cls)

    for turn_idx in range(total_remaining + 1):
        # -- Construct fresh prompt (consistent with RecurrentMemoryAgentLoop) --
        if turn_idx == 0:
            fresh_messages = list(initial_messages)
        else:
            turn_data = remaining_turns[turn_idx - 1]
            remaining_after = total_remaining - turn_idx
            # Reuse OnePredInteraction._format_observation
            observation = _interaction._format_observation(turn_data, remaining_after)

            # Reuse recurrent_agent_loop.MEMORY_USER_TEMPLATE
            user_content = MEMORY_USER_TEMPLATE.format(
                memory=memory,
                observation=observation,
            )
            fresh_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

        # -- tokenize & generate --
        prompt_text = tokenizer.apply_chat_template(
            fresh_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
        raw_response = outputs[0].outputs[0].text.strip()
        model_response = _strip_think(raw_response)

        # -- Extract memory (reusing recurrent_agent_loop._extract_memory) --
        memory = _extract_memory(raw_response)

        # Token-level truncation (consistent with RecurrentMemoryAgentLoop)
        mem_token_ids = tokenizer.encode(memory, add_special_tokens=False)
        if len(mem_token_ids) > MAX_MEMORY_TOKENS:
            mem_token_ids = mem_token_ids[:MAX_MEMORY_TOKENS]
            memory = tokenizer.decode(mem_token_ids, skip_special_tokens=True).strip() + "...(truncated)"

        # Format check
        text_clean = _strip_think(raw_response)
        has_mem = "<memory>" in text_clean and "</memory>" in text_clean
        has_pred = "<prediction>" in text_clean and "</prediction>" in text_clean

        has_memory_per_turn.append(has_mem)
        has_prediction_per_turn.append(has_pred)
        per_turn_responses.append(model_response)

    # -- Extract prediction and memory (reusing OnePredInteraction static methods) --
    final_messages = [{"role": "assistant", "content": per_turn_responses[-1]}]
    prediction = interaction_cls._extract_prediction(final_messages)
    final_memory = interaction_cls._extract_memory(final_messages)

    return {
        "ground_truth": ground_truth,
        "prediction": prediction,
        "final_memory": final_memory,
        "final_response": per_turn_responses[-1],
        "total_turns": total_remaining + 1,
        "has_memory_per_turn": has_memory_per_turn,
        "has_prediction_per_turn": has_prediction_per_turn,
        "session_id": row.get("extra_info", {}).get("session_id", ""),
    }


# ============================================================
# LLM Judge scoring (directly calling the llm_judge_score used in training)
# ============================================================

def score_with_llm_judge(result: dict) -> dict:
    score = llm_judge_score(result["prediction"], result["ground_truth"],
                            memory=result.get("final_memory", ""))
    result["score"] = score if score is not None else 0.5
    return result


# ============================================================
# Summary metrics
# ============================================================

def print_summary(results: list[dict], model_path: str, data_path: str, has_scores: bool):
    print(f"\n{'='*70}")
    print("Evaluation Results Summary")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Samples: {len(results)}")

    memory_ok = sum(all(r["has_memory_per_turn"]) for r in results)
    pred_ok = sum(r["has_prediction_per_turn"][-1] for r in results)
    pred_leak = sum(
        any(r["has_prediction_per_turn"][:-1]) if len(r["has_prediction_per_turn"]) > 1 else False
        for r in results
    )

    print(f"\n--- Format compliance ---")
    print(f"  All turns have <memory>: {memory_ok}/{len(results)} ({100*memory_ok/len(results):.1f}%)")
    print(f"  Final turn has <prediction>: {pred_ok}/{len(results)} ({100*pred_ok/len(results):.1f}%)")
    print(f"  Intermediate turn leaks <prediction>: {pred_leak}/{len(results)} ({100*pred_leak/len(results):.1f}%)")

    if has_scores:
        scores = [r.get("score", 0.5) for r in results]
        avg_score = sum(scores) / len(scores)
        score_dist = Counter(scores)

        print(f"\n--- LLM Judge Score ---")
        print(f"  Average: {avg_score:.4f}")
        print(f"  Score distribution:")
        for s in sorted(score_dist.keys()):
            count = score_dist[s]
            pct = 100 * count / len(scores)
            bar = "█" * int(pct / 2)
            print(f"    {s:.2f}: {count:3d} ({pct:5.1f}%) {bar}")

        turn_scores: dict[int, list[float]] = {}
        for r in results:
            t = r["total_turns"]
            turn_scores.setdefault(t, []).append(r.get("score", 0.5))

        if len(turn_scores) > 1:
            print(f"\n  Grouped by turns:")
            for t in sorted(turn_scores.keys()):
                s_list = turn_scores[t]
                print(f"    {t:2d} turns: avg={sum(s_list)/len(s_list):.4f} (n={len(s_list)})")

        return {"avg_score": avg_score, "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
                "turn_avg_scores": {str(t): sum(s)/len(s) for t, s in sorted(turn_scores.items())}}
    return {}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OnePred model evaluation")

    # Model source
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--ckpt_dir", type=str, default=None,
                             help="FSDP checkpoint actor/ directory (auto-merge)")
    model_group.add_argument("--model_path", type=str, default=None,
                             help="HF model path")

    # Data
    parser.add_argument("--data", type=str, default="data/eval_test_1000.parquet",
                        help="Evaluation data parquet (generated by build_eval_testset.py)")
    parser.add_argument("--start", type=int, default=None, help="Sample start index (for parallel slicing)")
    parser.add_argument("--end", type=int, default=None, help="Sample end index (for parallel slicing)")

    # Inference parameters
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (default 1, single GPU is enough for 8B model)")
    parser.add_argument("--gpu", type=int, default=None, help="Specify GPU id (for parallel mode)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                        help="Enable thinking mode (default on, consistent with training)")
    parser.add_argument("--no_thinking", action="store_true", help="Disable thinking mode")

    # Scoring
    parser.add_argument("--judge_workers", type=int, default=8)
    parser.add_argument("--skip_judge", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False

    # ---- Output directory & logging ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"outputs/eval_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "eval.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Args] {vars(args)}")

    # ---- Model path ----
    if args.ckpt_dir:
        target_dir = os.path.join(args.ckpt_dir, "merged_hf")
        model_path = merge_fsdp_checkpoint(args.ckpt_dir, target_dir)
    else:
        model_path = args.model_path
    print(f"\n[Model] {model_path}")

    # ---- Load data ----
    print(f"[Data] {args.data}")
    df = pd.read_parquet(args.data)
    print(f"[Data] Total samples: {len(df)}")

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(df)
    df = df.iloc[start:end]
    print(f"[Data] Evaluation range: [{start}, {end}), total {len(df)} samples")

    # ---- Load model ----
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[GPU] Using GPU {args.gpu}")

    print(f"[vLLM] Loading model (tp={args.tp})...")
    llm = LLM(
        model=model_path,
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

    # ---- Multi-turn inference ----
    print(f"\n[Inference] Starting...")
    results = []
    t_start = time.time()

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        n_turns = len(row["extra_info"]["interaction_kwargs"]["turns"]) + 1

        result = run_multiturn_inference(
            llm, tokenizer, row, sampling_params,
            enable_thinking=args.enable_thinking,
        )
        result["sample_idx"] = start + i
        results.append(result)

        elapsed = time.time() - t_start
        avg_time = elapsed / (i + 1)
        eta = avg_time * (len(df) - i - 1)

        if (i + 1) % 10 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] avg {avg_time:.1f}s/sample | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    print(f"\n[Inference] Done! Total time: {time.time() - t_start:.1f}s")

    # ---- LLM Judge scoring ----
    if not args.skip_judge:
        print(f"\n[Scoring] LLM Judge (workers={args.judge_workers})...")
        t_judge = time.time()

        with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
            futures = {executor.submit(score_with_llm_judge, r): i for i, r in enumerate(results)}
            done_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  [WARN] Sample {idx} scoring error: {e}")
                    results[idx]["score"] = 0.5
                done_count += 1
                if done_count % 20 == 0 or done_count == len(results):
                    print(f"  Scoring progress: {done_count}/{len(results)}")

        print(f"[Scoring] Done! Elapsed: {time.time() - t_judge:.1f}s")

    # ---- Save results ----
    result_path = os.path.join(args.output_dir, "results.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            record = {
                "sample_idx": r["sample_idx"],
                "session_id": r.get("session_id", ""),
                "total_turns": r["total_turns"],
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
                "final_memory": r["final_memory"],
                "score": r.get("score"),
                "has_memory_all": all(r["has_memory_per_turn"]),
                "has_prediction_final": r["has_prediction_per_turn"][-1],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[Output] Results: {result_path}")

    # ---- Summary ----
    score_summary = print_summary(results, model_path, args.data, not args.skip_judge)

    summary = {
        "model": model_path,
        "data": args.data,
        "range": [start, end],
        "num_samples": len(results),
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        **score_summary,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Output] Summary: {summary_path}")
    print(f"[Output] Log: {log_path}")
    print(f"\n[Done] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
