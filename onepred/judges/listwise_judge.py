"""Listwise LLM Judge: Comparative ranking of multiple prediction sets.

Scores multiple prediction sets against ground truth using pairwise
comparison via an LLM judge ensemble, returning per-set ranking scores.
"""

import os
import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from onepred.locale import (
    LISTWISE_SYSTEM_PROMPT,
    LISTWISE_USER_PROMPT,
    prediction_block_label,
    fallback_empty,
)

API_URLS = {
    "claude": os.getenv("CLAUDE_API_URL", ""),
    "gemini": os.getenv("GEMINI_API_URL", ""),
    "gpt": os.getenv("GPT_API_URL", ""),
}
API_KEY = os.getenv("LLM_API_KEY", "")


def listwise_judge_score(
    predictions: list[str],
    ground_truth: str,
    previous_queries: str = "",
) -> list[float]:
    """Rank multiple prediction sets and return normalized scores.

    Args:
        predictions: List of prediction texts (one per model/rollout).
        ground_truth: The actual next user query.
        previous_queries: Newline-separated previous user queries.

    Returns:
        List of scores (higher is better), one per prediction set.
    """
    n = len(predictions)
    if n == 0:
        return []

    prediction_blocks = "\n\n".join(
        f"[{prediction_block_label(i+1)}]\n{pred}" for i, pred in enumerate(predictions)
    )

    json_template = json.dumps(
        {prediction_block_label(i+1): {"rank": 1, "reason": "brief"} for i in range(n)},
        indent=2,
    )

    user_prompt = LISTWISE_USER_PROMPT.format(
        n=n,
        previous_queries=previous_queries or fallback_empty,
        ground_truth=ground_truth,
        prediction_blocks=prediction_blocks,
        json_template=json_template,
    )

    def _call_model(url: str) -> dict | None:
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": LISTWISE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_call_model, url): name for name, url in API_URLS.items() if url}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        return [0.0] * n

    all_ranks = []
    for r in results:
        ranks = []
        for i in range(n):
            key = prediction_block_label(i + 1)
            rank = r.get(key, {}).get("rank", n)
            ranks.append(rank)
        all_ranks.append(ranks)

    avg_ranks = [sum(r[i] for r in all_ranks) / len(all_ranks) for i in range(n)]
    scores = [(n - r + 1) / n for r in avg_ranks]
    return scores
