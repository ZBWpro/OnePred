"""
OnePred custom reward function - for GRPO training.

Supports two input formats:
1. Single-turn: solution_str is the prediction text directly
2. Multi-turn (agentic): solution_str contains multiple <memory>...</memory> and <prediction>...</prediction>
   In this case, prediction text is extracted from <prediction> tags

Scoring logic:
  - Extract candidate question list from <prediction>
  - Extract memory context from the last turn's <memory>
  - Send candidate list + memory + ground truth to LLM judge for scoring at once
  - Return 0.0/0.25/0.5/0.75/1.0

Note: In agentic training, the actual reward is computed by onepred_interaction.py;
      this function only serves as a fallback path to maintain logical consistency.
"""

import re

import os
if os.getenv("USE_LLM_JUDGE_API") == "1":
    from onepred.judges.llm_judge_v1 import llm_judge_score
else:
    from onepred.judges.llm_judge_multi import llm_judge_score


def _truncate_to_first_question(text: str) -> str:
    """Truncate to the first question mark, preventing a single candidate from containing multiple sub-questions.

    Example: "How will oil prices move? Will they keep rising? When will they come down?" -> "How will oil prices move?"
    Truncation only triggers when question mark count >= 2.
    """
    qmarks = [m.start() for m in re.finditer(r"[？?]", text)]
    if len(qmarks) >= 2:
        return text[:qmarks[0] + 1]
    return text


def _extract_predictions(text: str) -> list[str]:
    """Extract candidate prediction list from model output. Preferably extracts <prediction> tag content."""
    # Remove <think>...</think> content first to avoid extracting tags from thinking
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"<prediction>(.*?)</prediction>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        # Fallback: take the last non-empty line of text
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return [lines[-1]] if lines else [text.strip()]

    # Handle literal \n (model sometimes outputs backslash+n instead of actual newline)
    content = content.replace("\\n", "\n")

    # Parse numbered list: "1. xxx\n2. yyy\n3. zzz"
    items = re.findall(r"^\d+[.、]\s*(.+)$", content, re.MULTILINE)
    if items:
        return [_truncate_to_first_question(item.strip()) for item in items if item.strip()]

    # Fallback: split by newline
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return [_truncate_to_first_question(l) for l in lines] if lines else [content]


def _extract_memory(text: str) -> str:
    """Extract the last <memory> tag content from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = re.findall(r"<memory>(.*?)</memory>", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """
    Compute reward score.

    Args:
        data_source: Data source identifier (e.g. "onepred")
        solution_str: Model-generated prediction text (single-turn) or multi-turn agent output
        ground_truth: The actual next user question
        extra_info: Additional information
    Returns:
        float: reward score (judge's score for the best candidate)
    """
    preds = _extract_predictions(solution_str)
    memory = _extract_memory(solution_str)
    gt = ground_truth.strip()

    # Combine candidate list into numbered text, call judge once
    pred_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preds) if p.strip())
    if not pred_text:
        return 0.0

    previous_user_queries = ""
    if extra_info and extra_info.get("all_queries"):
        previous_user_queries = "\n".join(q for q in extra_info["all_queries"] if q)

    score = llm_judge_score(pred_text, gt, memory=memory, previous_user_queries=previous_user_queries)
    return score if score is not None else 0.0
