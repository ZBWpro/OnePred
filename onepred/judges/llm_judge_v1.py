"""LLM Judge V1: Single-model scoring via commercial API.

This module provides an alternative judge implementation using a single
commercial LLM endpoint (configured via USE_LLM_JUDGE_API=1).
"""

import os
import re
import requests

from onepred.locale import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    is_not_question_patterns,
    is_generic_patterns,
    fallback_none,
    fallback_empty,
)

API_URL = os.getenv("CLAUDE_API_URL", "")
API_KEY = os.getenv("LLM_API_KEY", "")


def llm_judge_score(
    prediction_text: str,
    ground_truth: str,
    memory: str = "",
    previous_user_queries: str = "",
) -> float | None:
    """Score predictions against ground truth using a single LLM judge.

    Returns a float in {0.0, 0.25, 0.5, 0.75, 1.0} or None on failure.
    """
    if not prediction_text.strip():
        return 0.0

    candidates = [c.strip() for c in re.findall(r"^\d+[.、]\s*(.+)$", prediction_text, re.MULTILINE)]
    if not candidates:
        candidates = [prediction_text.strip()]

    for pattern in is_not_question_patterns + is_generic_patterns:
        if all(re.search(pattern, c) for c in candidates):
            return 0.0

    pad = lambda s: s if s.strip() else fallback_empty
    user_prompt = JUDGE_USER_PROMPT.format(
        memory=pad(memory),
        previous_user_queries=pad(previous_user_queries),
        ground_truth=ground_truth,
        candidate_1=candidates[0] if len(candidates) > 0 else fallback_none,
        candidate_2=candidates[1] if len(candidates) > 1 else fallback_none,
        candidate_3=candidates[2] if len(candidates) > 2 else fallback_none,
    )

    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "default",
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 512,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        import json
        scores_data = json.loads(content)
        scores = []
        for key in ["candidate_1", "candidate_2", "candidate_3"]:
            if key in scores_data:
                scores.append(float(scores_data[key]["score"]))
        return max(scores) if scores else None
    except Exception:
        return None
