"""
Evaluation post-processing: compute LLM Judge V1 / ROUGE-L / BLEU / Reranker / Exact Match from structured_traces.

Each sample's prediction is split into 3 candidates; metrics are computed per candidate and max is taken (consistent with judge reward).

Usage:
  # Full metrics (with Reranker + LLM Judge V1)
  python -m tools.compute_metrics \
      outputs/eval-xxx/structured_traces/traces.jsonl \
      --reranker_model os.environ.get("MODEL_DIR", "./models")/Qwen3-Reranker-0.6B \
      --output_dir outputs/eval-xxx/

  # Skip LLM Judge (no API needed)
  python -m tools.compute_metrics \
      outputs/eval-xxx/structured_traces/traces.jsonl \
      --skip_llm_judge \
      --output_dir outputs/eval-xxx/

  # ROUGE/BLEU only (no GPU, no API needed)
  python -m tools.compute_metrics \
      outputs/eval-xxx/structured_traces/traces.jsonl \
      --skip_reranker --skip_llm_judge \
      --output_dir outputs/eval-xxx/
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba
import sacrebleu
from rouge_score import rouge_scorer


# ============================================================
# Candidate splitting (reusing llm_judge_v3 logic)
# ============================================================

def split_candidates(prediction_text: str) -> list[str]:
    """Split prediction text into 3 candidate questions."""
    # Handle literal \n (model sometimes outputs backslash+n instead of actual newline)
    text = prediction_text.replace("\\n", "\n")
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    candidates = []
    for line in lines:
        cleaned = re.sub(r"^(\d+[.、:：)\]]\s*|（\d+）\s*|-\s*)", "", line).strip()
        if cleaned:
            candidates.append(cleaned)
    while len(candidates) < 3:
        candidates.append("")
    return candidates[:3]


# ============================================================
# Chinese tokenizer (for ROUGE-L)
# ============================================================

class ChineseTokenizer:
    """jieba tokenizer for use with rouge_scorer."""

    def tokenize(self, text: str) -> list[str]:
        return list(jieba.cut(text))


# ============================================================
# ROUGE-L
# ============================================================

def compute_rouge_l(candidate: str, reference: str, scorer: rouge_scorer.RougeScorer) -> float:
    """Compute ROUGE-L F1 for a single candidate against the reference."""
    if not candidate or not reference:
        return 0.0
    scores = scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure


# ============================================================
# BLEU
# ============================================================

def _jieba_tokenize(text: str) -> str:
    """Tokenize with jieba and join with spaces, for sacrebleu use."""
    return " ".join(jieba.cut(text))


def compute_bleu(candidate: str, reference: str) -> float:
    """Compute sentence BLEU for a single candidate against the reference."""
    if not candidate or not reference:
        return 0.0
    hyp = _jieba_tokenize(candidate)
    ref = _jieba_tokenize(reference)
    result = sacrebleu.sentence_bleu(hyp, [ref], tokenize="none")
    return result.score / 100.0  # Normalize to [0, 1]


# ============================================================
# Exact Match
# ============================================================

def compute_exact_match(candidate: str, reference: str) -> float:
    return 1.0 if candidate.strip() == reference.strip() else 0.0


# ============================================================
# Reranker (Qwen3-Reranker-0.6B)
# ============================================================

class QwenReranker:
    """Qwen3-Reranker wrapper, same calling convention as in training llm_judge_reranker.py."""

    INSTRUCTION = (
        "Determine whether the candidate question expresses the same or highly similar user intent as the target question. "
        "If the core purpose of both questions is the same, even if the phrasing differs, they should be judged as related."
    )

    def __init__(self, model_path: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device

        print(f"[Reranker] Loading model: {model_path} -> {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(device).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        print(f"[Reranker] Loading complete")

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch compute relevance scores for (query, document) pairs, same as _batch_score during training."""
        import torch

        prefix = f"<Instruct>: {self.INSTRUCTION}\n"
        tokenizer_pairs = [
            [prefix + f"<Query>: {q}", f"<Document>: {d}"]
            for q, d in pairs
        ]

        inputs = self.tokenizer(
            tokenizer_pairs,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            scores_raw = logits[:, [self.token_false_id, self.token_true_id]].float()
            scores = torch.nn.functional.log_softmax(scores_raw, dim=1)[:, 1].exp()

        return scores.cpu().tolist()


# ============================================================
# LLM Judge V1 (Claude Opus 4.6 API)
# ============================================================

def _extract_memory_from_trace(record: dict) -> str:
    """Extract memory from the last model_response in the trace."""
    turns = record.get("turns", [])
    if not turns:
        return ""
    last_response = turns[-1].get("model_response", "")
    clean = re.sub(r"<think>.*?</think>", "", last_response, flags=re.DOTALL)
    matches = re.findall(r"<memory>(.*?)</memory>", clean, re.DOTALL)
    return matches[-1].strip() if matches else ""


def _extract_previous_queries_from_trace(record: dict) -> str:
    """Extract previous user queries list from observations in the trace."""
    turns = record.get("turns", [])
    queries = []
    for turn in turns:
        obs = turn.get("observation", "")
        match = re.search(r"User: (.+?)(?:\n|$)", obs)
        if match:
            queries.append(match.group(1).strip())
    return "\n".join(queries)


def compute_llm_judge_v1(
    records: list[dict],
    max_workers: int = 30,
) -> list[float | None]:
    """Concurrently call LLM Judge Multi (3-model majority vote, five-level discrete) to score each record."""
    from onepred.judges.llm_judge_multi import llm_judge_score

    print(f"[Metrics] Computing LLM Judge Multi (3-model majority vote, workers={max_workers}) ...")
    t0 = time.time()

    def score_one(idx_rec):
        idx, rec = idx_rec
        pred = rec.get("prediction", "").strip()
        gt = rec.get("ground_truth", "").strip()
        if not pred:
            return idx, 0.0
        memory = _extract_memory_from_trace(rec)
        prev_queries = _extract_previous_queries_from_trace(rec)
        score = llm_judge_score(pred, gt, memory=memory, previous_user_queries=prev_queries)
        return idx, score

    scores = [None] * len(records)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_one, (i, rec)): i
            for i, rec in enumerate(records)
        }
        for future in as_completed(futures):
            try:
                idx, score = future.result()
                scores[idx] = score
            except Exception as e:
                idx = futures[future]
                print(f"  [WARN] record {idx} LLM Judge failed: {e}")
                scores[idx] = None
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                print(f"  [{completed}/{len(records)}] ...")

    valid = [s for s in scores if s is not None]
    print(f"  LLM Judge Multi complete, elapsed {time.time() - t0:.1f}s, "
          f"success {len(valid)}/{len(records)}")

    return scores


# ============================================================
# Embedding similarity (BGE-M3, local inference)
# ============================================================

def compute_embedding_scores(records: list[dict]) -> list[float]:
    """Compute semantic similarity score for each record using BGE-M3 embedding."""
    from onepred.judges.llm_judge_embedding import llm_judge_score as embedding_score

    print("[Metrics] Computing Embedding similarity (BGE-M3) ...")
    t0 = time.time()

    scores = []
    for i, rec in enumerate(records):
        pred = rec.get("prediction", "").strip()
        gt = rec.get("ground_truth", "").strip()
        if not pred:
            scores.append(0.0)
            continue
        memory = _extract_memory_from_trace(rec)
        prev_queries = _extract_previous_queries_from_trace(rec)
        score = embedding_score(pred, gt, memory=memory, previous_user_queries=prev_queries)
        scores.append(score if score is not None else 0.0)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(records)}] ...")

    print(f"  Embedding complete, elapsed {time.time() - t0:.1f}s")
    return scores


# ============================================================
# Main flow
# ============================================================

def load_traces(path: str) -> list[dict]:
    """Load structured_traces jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_all_metrics(
    records: list[dict],
    reranker: QwenReranker | None = None,
    reranker_batch_size: int = 32,
) -> list[dict]:
    """Compute all metrics for each record, returning records with metrics attached."""
    # Initialize ROUGE scorer (jieba tokenization)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=ChineseTokenizer())

    # Pre-tokenize (jieba has cache, run once to warm up)
    print("[Metrics] Computing ROUGE-L / BLEU / Exact Match ...")
    t0 = time.time()

    for i, rec in enumerate(records):
        gt = rec.get("ground_truth", "").strip()
        pred_text = rec.get("prediction", "").strip()
        candidates = split_candidates(pred_text) if pred_text else ["", "", ""]

        # Compute per-candidate scores
        rouge_scores = [compute_rouge_l(c, gt, scorer) for c in candidates]
        bleu_scores = [compute_bleu(c, gt) for c in candidates]
        em_scores = [compute_exact_match(c, gt) for c in candidates]

        rec["candidates"] = candidates
        rec["rouge_l_per_cand"] = rouge_scores
        rec["bleu_per_cand"] = bleu_scores
        rec["em_per_cand"] = em_scores
        rec["rouge_l"] = max(rouge_scores)
        rec["bleu"] = max(bleu_scores)
        rec["exact_match"] = max(em_scores)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(records)}] ...")

    print(f"  ROUGE/BLEU/EM complete, elapsed {time.time() - t0:.1f}s")

    # Reranker
    if reranker is not None:
        print(f"[Metrics] Computing Reranker Score (batch_size={reranker_batch_size}) ...")
        t1 = time.time()

        # Collect all (gt, candidate) pairs
        all_pairs = []
        pair_indices = []  # (record_idx, cand_idx)
        for i, rec in enumerate(records):
            gt = rec.get("ground_truth", "").strip()
            for j, c in enumerate(rec["candidates"]):
                if c:
                    all_pairs.append((gt, c))
                    pair_indices.append((i, j))

        # Initialize reranker scores for each record
        for rec in records:
            rec["reranker_per_cand"] = [0.0, 0.0, 0.0]

        # Batch inference
        for batch_start in range(0, len(all_pairs), reranker_batch_size):
            batch_end = min(batch_start + reranker_batch_size, len(all_pairs))
            batch_pairs = all_pairs[batch_start:batch_end]
            batch_scores = reranker.score_batch(batch_pairs)

            for k, score in enumerate(batch_scores):
                ri, ci = pair_indices[batch_start + k]
                records[ri]["reranker_per_cand"][ci] = score

            if (batch_end) % (reranker_batch_size * 10) == 0 or batch_end == len(all_pairs):
                print(f"  [{batch_end}/{len(all_pairs)} pairs] ...")

        for rec in records:
            rec["reranker"] = max(rec["reranker_per_cand"])

        print(f"  Reranker complete, elapsed {time.time() - t1:.1f}s")
    else:
        for rec in records:
            rec["reranker_per_cand"] = None
            rec["reranker"] = None

    return records


# ============================================================
# Summary report
# ============================================================

def print_summary(records: list[dict], has_reranker: bool, has_llm_judge: bool, has_embedding: bool):
    n = len(records)
    print(f"\n{'='*70}")
    print("Evaluation Metrics Summary")
    print(f"{'='*70}")
    print(f"Samples: {n}")

    # Rollout reward (included in traces)
    rewards = [r["reward"] for r in records if r.get("reward") is not None
               and not (isinstance(r["reward"], float) and r["reward"] != r["reward"])]
    if rewards:
        print(f"\n--- Rollout Reward (from traces) ---")
        print(f"  Mean: {sum(rewards)/len(rewards):.4f} (valid {len(rewards)}/{n})")

    # LLM Judge V1
    if has_llm_judge:
        judge_vals = [r["llm_judge_v1"] for r in records
                      if r.get("llm_judge_v1") is not None]
        if judge_vals:
            print(f"\n--- LLM Judge V1 (Claude Opus 4.6) ---")
            print(f"  Mean: {sum(judge_vals)/len(judge_vals):.4f} (valid {len(judge_vals)}/{n})")

    # Embedding
    if has_embedding:
        emb_vals = [r["embedding"] for r in records]
        print(f"\n--- Embedding Similarity (BGE-M3) ---")
        print(f"  Mean: {sum(emb_vals)/n:.4f}")

    # ROUGE-L
    rouge_vals = [r["rouge_l"] for r in records]
    print(f"\n--- ROUGE-L (max of 3 candidates) ---")
    print(f"  Mean: {sum(rouge_vals)/n:.4f}")

    # BLEU
    bleu_vals = [r["bleu"] for r in records]
    print(f"\n--- BLEU-4 (max of 3 candidates) ---")
    print(f"  Mean: {sum(bleu_vals)/n:.4f}")

    # Exact Match
    em_vals = [r["exact_match"] for r in records]
    print(f"\n--- Exact Match ---")
    print(f"  Hit rate: {sum(em_vals)/n:.4f} ({int(sum(em_vals))}/{n})")

    # Reranker
    if has_reranker:
        reranker_vals = [r["reranker"] for r in records]
        print(f"\n--- Reranker Score (max of 3 candidates) ---")
        print(f"  Mean: {sum(reranker_vals)/n:.4f}")

    # Group by turns
    turn_metrics: dict[int, list[dict]] = {}
    for r in records:
        t = r.get("total_turns", 0) + 1  # total_turns is remaining turns, +1 for total
        turn_metrics.setdefault(t, []).append(r)

    if len(turn_metrics) > 1:
        print(f"\n--- Grouped by turns ---")
        header = f"  {'Turns':>4s}  {'Count':>5s}  {'Rollout':>7s}"
        if has_llm_judge:
            header += f"  {'JudgeV1':>7s}"
        if has_embedding:
            header += f"  {'Embed':>6s}"
        header += f"  {'ROUGE':>6s}  {'BLEU':>6s}  {'EM':>5s}"
        if has_reranker:
            header += f"  {'Rerank':>6s}"
        print(header)

        for t in sorted(turn_metrics.keys()):
            group = turn_metrics[t]
            gn = len(group)
            g_rewards = [r["reward"] for r in group if r.get("reward") is not None
                         and not (isinstance(r["reward"], float) and r["reward"] != r["reward"])]
            g_rollout = sum(g_rewards) / len(g_rewards) if g_rewards else float("nan")
            g_rouge = sum(r["rouge_l"] for r in group) / gn
            g_bleu = sum(r["bleu"] for r in group) / gn
            g_em = sum(r["exact_match"] for r in group) / gn

            line = f"  {t:4d}  {gn:5d}  {g_rollout:7.4f}"
            if has_llm_judge:
                g_judge = [r["llm_judge_v1"] for r in group if r.get("llm_judge_v1") is not None]
                g_judge_avg = sum(g_judge) / len(g_judge) if g_judge else float("nan")
                line += f"  {g_judge_avg:7.4f}"
            if has_embedding:
                g_emb = sum(r["embedding"] for r in group) / gn
                line += f"  {g_emb:6.4f}"
            line += f"  {g_rouge:6.4f}  {g_bleu:6.4f}  {g_em:5.3f}"
            if has_reranker:
                g_reranker = sum(r["reranker"] for r in group) / gn
                line += f"  {g_reranker:6.4f}"
            print(line)

    print(f"{'='*70}")

    # Build summary dict
    summary = {
        "num_samples": n,
        "rollout_reward_avg": sum(rewards) / len(rewards) if rewards else None,
        "rouge_l_avg": sum(rouge_vals) / n,
        "bleu_avg": sum(bleu_vals) / n,
        "exact_match_rate": sum(em_vals) / n,
    }
    if has_llm_judge:
        judge_vals = [r["llm_judge_v1"] for r in records if r.get("llm_judge_v1") is not None]
        summary["llm_judge_v1_avg"] = sum(judge_vals) / len(judge_vals) if judge_vals else None
        summary["llm_judge_v1_valid"] = len(judge_vals)
    if has_embedding:
        emb_vals = [r["embedding"] for r in records]
        summary["embedding_avg"] = sum(emb_vals) / n
    if has_reranker:
        reranker_vals = [r["reranker"] for r in records]
        summary["reranker_avg"] = sum(reranker_vals) / n

    # Group by turns
    summary["turn_metrics"] = {}
    for t in sorted(turn_metrics.keys()):
        group = turn_metrics[t]
        gn = len(group)
        g_rewards = [r["reward"] for r in group if r.get("reward") is not None
                     and not (isinstance(r["reward"], float) and r["reward"] != r["reward"])]
        entry = {
            "count": gn,
            "rollout_reward_avg": sum(g_rewards) / len(g_rewards) if g_rewards else None,
            "rouge_l_avg": sum(r["rouge_l"] for r in group) / gn,
            "bleu_avg": sum(r["bleu"] for r in group) / gn,
            "exact_match_rate": sum(r["exact_match"] for r in group) / gn,
        }
        if has_llm_judge:
            g_judge = [r["llm_judge_v1"] for r in group if r.get("llm_judge_v1") is not None]
            entry["llm_judge_v1_avg"] = sum(g_judge) / len(g_judge) if g_judge else None
        if has_embedding:
            entry["embedding_avg"] = sum(r["embedding"] for r in group) / gn
        if has_reranker:
            entry["reranker_avg"] = sum(r["reranker"] for r in group) / gn
        summary["turn_metrics"][str(t)] = entry

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="OnePred evaluation post-processing: compute LLM Judge V1 / Embedding / ROUGE-L / BLEU / Reranker / EM")
    parser.add_argument("traces_path", type=str,
                        help="Path to structured_traces/traces.jsonl")
    parser.add_argument("--reranker_model", type=str,
                        default=os.path.join(os.environ.get("MODEL_DIR", "./models"), "Qwen3-Reranker-0.6B"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_reranker", action="store_true",
                        help="Skip Reranker")
    parser.add_argument("--skip_llm_judge", action="store_true",
                        help="Skip LLM Judge V1 (no API needed)")
    parser.add_argument("--skip_embedding", action="store_true",
                        help="Skip Embedding similarity (no GPU needed)")
    parser.add_argument("--llm_judge_workers", type=int, default=30,
                        help="LLM Judge V1 concurrent threads (default 30)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same level as traces)")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.dirname(args.traces_path))
        if not args.output_dir:
            args.output_dir = "."
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"[Loading] {args.traces_path}")
    records = load_traces(args.traces_path)
    print(f"[Loading] {len(records)} records")

    # Initialize Reranker
    reranker = None
    if not args.skip_reranker:
        reranker = QwenReranker(args.reranker_model, device=args.device)

    # Compute ROUGE / BLEU / EM / Reranker
    records = compute_all_metrics(records, reranker=reranker, reranker_batch_size=args.batch_size)

    # Compute Embedding similarity
    has_embedding = not args.skip_embedding
    if has_embedding:
        emb_scores = compute_embedding_scores(records)
        for rec, score in zip(records, emb_scores):
            rec["embedding"] = score

    # Compute LLM Judge V1
    has_llm_judge = not args.skip_llm_judge
    if has_llm_judge:
        judge_scores = compute_llm_judge_v1(records, max_workers=args.llm_judge_workers)
        for rec, score in zip(records, judge_scores):
            rec["llm_judge_v1"] = score

    # Summary
    has_reranker = not args.skip_reranker
    summary = print_summary(records, has_reranker, has_llm_judge, has_embedding)

    # Save summary
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] Summary: {summary_path}")

    # Save details
    detail_path = os.path.join(args.output_dir, "metrics_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "ground_truth": r.get("ground_truth", ""),
                "prediction": r.get("prediction", ""),
                "candidates": r.get("candidates", []),
                "total_turns": r.get("total_turns"),
                "reward": r.get("reward"),
                "rouge_l": r["rouge_l"],
                "bleu": r["bleu"],
                "exact_match": r["exact_match"],
                "rouge_l_per_cand": r["rouge_l_per_cand"],
                "bleu_per_cand": r["bleu_per_cand"],
                "em_per_cand": r["em_per_cand"],
            }
            if has_llm_judge:
                row["llm_judge_v1"] = r.get("llm_judge_v1")
            if has_embedding:
                row["embedding"] = r.get("embedding")
            if has_reranker:
                row["reranker"] = r["reranker"]
                row["reranker_per_cand"] = r["reranker_per_cand"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Output] Details: {detail_path}")


if __name__ == "__main__":
    main()
