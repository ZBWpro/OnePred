"""LLM-as-Judge reward (multi-model): Multi-model five-level discrete scoring with majority vote.

Scoring flow:
  1. Rule-based code filtering (repetition/non-question/generic -> direct 0)
  2. Candidates passing the filter are sent to 3 remote LLM models for concurrent scoring {0.0, 0.25, 0.5, 0.75, 1.0}
  3. For each candidate, take the majority vote across models (use median if no majority)
  4. Final reward = max(majority scores of 3 candidates)

Models:
  - Claude Opus 4.6
  - Gemini 3.1 Pro
  - GPT-5

Scoring criteria (five discrete levels):
  1.0:  Intent match, candidate asks about the same thing as the actual next question
  0.75: Correct direction, guessed what the user wants to do but specific focus differs
  0.5:  Topic related, guessed the topic but not what the user will ask next
  0.25: Reasonable question, possible in the conversation context but not directly related to the actual next question
  0.0:  Completely unrelated

NaN handling: Skip model on timeout/failure, use remaining models. Return None if all fail.
"""

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from onepred.locale import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    is_not_question_patterns,
    is_generic_patterns,
    fallback_none,
    fallback_empty,
)

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("LLM_API_KEY", "")
TIMEOUT = 40  # seconds per request
MAX_TOTAL_TIME = 130  # seconds, total retry budget per model

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "claude_opus_4.6",
        "url": os.environ.get("CLAUDE_API_URL", ""),
        "supports_temperature": True,
    },
    {
        "name": "gemini_3.1_pro",
        "url": os.environ.get("GEMINI_API_URL", ""),
        "supports_temperature": True,
    },
    {
        "name": "gpt_5",
        "url": os.environ.get("GPT_API_URL", ""),
        "supports_temperature": False,
    },
]

# ---------------------------------------------------------------------------
# Rule-based filters
# ---------------------------------------------------------------------------

def _is_repeat(candidate: str, previous_queries: str) -> bool:
    """Detect mechanical repetition of historical questions. Includes length ratio limit (>0.8) to reduce false positives."""
    if not previous_queries.strip():
        return False
    c_norm = candidate.strip().lower().replace(" ", "").replace("？", "").replace("?", "")
    for q in previous_queries.strip().split("\n"):
        q_norm = q.strip().lower().replace(" ", "").replace("？", "").replace("?", "")
        if not q_norm:
            continue
        if c_norm == q_norm:
            return True
        if len(c_norm) > 5 and len(q_norm) > 5:
            shorter, longer = sorted([c_norm, q_norm], key=len)
            if shorter in longer and len(shorter) / len(longer) > 0.8:
                return True
    return False


def _is_not_question(candidate: str) -> bool:
    """Detect whether the candidate is a third-person description rather than a user question."""
    for p in is_not_question_patterns:
        if re.search(p, candidate.strip()):
            return True
    return False


def _is_generic(candidate: str) -> bool:
    """Detect whether the candidate is an overly generic question."""
    c = candidate.strip()
    if len(c) < 4:
        return True
    for p in is_generic_patterns:
        if re.search(p, c):
            return True
    return False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


def _parse_judge_output(content: str) -> list[float] | None:
    """Extract five-level discrete scores {0.0, 0.25, 0.5, 0.75, 1.0} using regex."""
    content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Remove markdown code block wrapping
    content_clean = re.sub(r"^```(?:json)?\s*", "", content_clean)
    content_clean = re.sub(r"\s*```\s*$", "", content_clean)

    scores = []
    for key in ("candidate_1", "candidate_2", "candidate_3"):
        match = re.search(rf'"{key}".*?"score"\s*:\s*([\d.]+)', content_clean, re.DOTALL)
        if not match:
            return None
        try:
            raw = float(match.group(1))
        except ValueError:
            return None
        # Snap to the nearest valid value
        snapped = min(_VALID_SCORES, key=lambda v: abs(v - raw))
        if abs(snapped - raw) > 0.13:
            return None
        scores.append(snapped)

    return scores


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _truncate_to_first_question(text: str) -> str:
    """Truncate to the first question mark, preventing a single candidate from containing multiple sub-questions."""
    qmarks = [m.start() for m in re.finditer(r"[？?]", text)]
    if len(qmarks) >= 2:
        return text[:qmarks[0] + 1]
    return text


def _split_candidates(prediction_text: str) -> list[str]:
    """Split prediction text into 3 candidate questions."""
    # Handle literal \n (model sometimes outputs backslash+n instead of actual newline)
    text = prediction_text.replace("\\n", "\n")
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    candidates = []
    for line in lines:
        cleaned = re.sub(r"^(\d+[.、:：)\]]\s*|（\d+）\s*|-\s*)", "", line).strip()
        if cleaned:
            candidates.append(_truncate_to_first_question(cleaned))
    while len(candidates) < 3:
        candidates.append("")
    return candidates[:3]


# ---------------------------------------------------------------------------
# Single model call (with retry)
# ---------------------------------------------------------------------------

def _call_single_model(model: dict, user_prompt: str) -> list[float] | None:
    """Call a single model for scoring. Returns a list of 3 candidate scores, or None on failure."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if model["supports_temperature"]:
        data["temperature"] = 0.0

    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(model["url"], headers=headers, json=data, timeout=TIMEOUT)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            scores = _parse_judge_output(content)
            if scores is None:
                logger.warning(
                    "Multi-judge %s output unparseable: %r",
                    model["name"], content[:200],
                )
                return None

            logger.debug("Multi-judge %s: scores=%r", model["name"], scores)
            return scores

        except Exception as e:
            elapsed = time.monotonic() - start
            if elapsed + TIMEOUT >= MAX_TOTAL_TIME:
                logger.warning(
                    "Multi-judge %s gave up after %d attempts (%.0fs): %s",
                    model["name"], attempt, elapsed, e,
                )
                return None
            wait = min(2 ** (attempt - 1), 8)
            logger.warning(
                "Multi-judge %s attempt %d failed (%.0fs elapsed), retrying in %ds: %s",
                model["name"], attempt, elapsed, wait, e,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Public API (same interface as other judges)
# ---------------------------------------------------------------------------

def llm_judge_score(
    prediction: str,
    ground_truth: str,
    memory: str = "",
    previous_user_queries: str = "",
) -> float | None:
    """Multi-model five-level discrete scoring with majority vote.

    Returns:
        float reward in {0.0, 0.25, 0.5, 0.75, 1.0}, or None if all models failed
    """
    pred = prediction.strip()
    gt = ground_truth.strip()

    if not pred:
        return 0.0

    candidates = _split_candidates(pred)

    # Fast path: exact match
    for c in candidates:
        if c == gt:
            return 1.0

    # Rule-based code filtering
    filtered = []
    for c in candidates:
        if not c:
            filtered.append(True)
        elif _is_not_question(c):
            filtered.append(True)
        elif _is_repeat(c, previous_user_queries):
            filtered.append(True)
        elif _is_generic(c):
            filtered.append(True)
        else:
            filtered.append(False)

    # All candidates filtered out
    if all(filtered):
        return 0.0

    # Build prompt
    user_prompt = JUDGE_USER_PROMPT.format(
        memory=memory.strip() if memory else fallback_none,
        previous_user_queries=previous_user_queries.strip() if previous_user_queries else fallback_none,
        ground_truth=gt,
        candidate_1=candidates[0] if candidates[0] else fallback_empty,
        candidate_2=candidates[1] if candidates[1] else fallback_empty,
        candidate_3=candidates[2] if candidates[2] else fallback_empty,
    )

    # Concurrently call 3 models
    model_scores = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {
            executor.submit(_call_single_model, model, user_prompt): model
            for model in MODELS
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                scores = future.result()
                if scores is not None:
                    model_scores.append(scores)
            except Exception as e:
                logger.warning("Multi-judge %s unexpected error: %s", model["name"], e)

    if not model_scores:
        return None

    # For each candidate, take majority vote across models (each model outputs discrete values); filtered candidates are fixed at 0
    final_scores = []
    for i in range(3):
        if filtered[i]:
            final_scores.append(0.0)
        else:
            candidate_scores = [s[i] for s in model_scores]
            # Majority vote: take the value with highest count; use median if no majority
            counts = Counter(candidate_scores)
            most_common_score, most_common_count = counts.most_common(1)[0]
            if most_common_count > 1:
                final_scores.append(most_common_score)
            else:
                final_scores.append(sorted(candidate_scores)[len(candidate_scores) // 2])

    reward = max(final_scores)

    logger.debug(
        "Multi-judge: gt=%r | candidates=%r | n_models=%d | avg_scores=%r | filtered=%r | reward=%.4f",
        gt[:50], [c[:30] for c in candidates],
        len(model_scores), [f"{s:.3f}" for s in final_scores], filtered, reward,
    )

    return reward
