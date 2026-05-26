"""
OnePred Agentic Interaction: Feed conversation history turn by turn, Agent maintains working memory then predicts.

Flow:
  - start_interaction: Initialize instance, store all turns to be fed
  - generate_response: Called after each model generation
      - If remaining turns exist: Feed next turn <query, response>, return should_terminate=False
      - If all turns have been fed: Extract prediction from model output, compute reward, return should_terminate=True

Interaction diagram (n-turn conversation):
  Prompt (system + 1st turn observation)
    -> Agent generates <memory>...</memory>
  Interaction feeds 2nd turn observation
    -> Agent generates <memory>...</memory>
  ...
  Interaction feeds nth turn observation + "[system instruction] please give prediction"
    -> Agent generates <memory>...</memory> <prediction>...</prediction>
  Interaction extracts prediction, computes reward -> terminate
"""

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

import os
if os.getenv("USE_LLM_JUDGE_API") == "1":
    from onepred.judges.llm_judge_v1 import llm_judge_score
else:
    from onepred.judges.llm_judge_multi import llm_judge_score

from onepred.locale import (
    observation_header,
    observation_user_label,
    observation_response_label,
    feedback_label as locale_feedback_label,
    system_instruction,
    response_truncation_marker,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Max characters for response text in each observation turn (0 = no truncation)
MAX_RESPONSE_CHARS = int(os.getenv("ONEPRED_MAX_RESPONSE_CHARS", "0"))

# Trace save directory (set via environment variable)
TRACE_DIR = os.getenv("ONEPRED_TRACE_DIR", "")

# Probability of sampling and printing traces to stdout
TRACE_PRINT_RATE = float(os.getenv("ONEPRED_TRACE_PRINT_RATE", "0.03"))


class OnePredInteraction(BaseInteraction):

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances: dict[str, dict] = {}
        self._trace_buffer: list[dict] = []
        self._trace_lock = threading.Lock()
        self._trace_flush_count = int(os.getenv("ONEPRED_TRACE_FLUSH_EVERY", "1"))

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        turns = kwargs.get("turns", [])
        self._instances[instance_id] = {
            "turns": turns,
            "ground_truth": kwargs.get("ground_truth", ""),
            "all_queries": kwargs.get("all_queries", None),
            "current_turn": 0,
            "total_turns": len(turns),
            # Per-turn trajectory recording
            "trajectory": [],  # list of {"turn": int, "observation": str, "model_response": str}
            "should_print": hash(instance_id) % 100 < int(TRACE_PRINT_RATE * 100),
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """
        Called after each model generation.
        messages is the full conversation history: [system, user(turn1), assistant, user(turn2), assistant, ...]
        """
        inst = self._instances[instance_id]
        cur = inst["current_turn"]
        total = inst["total_turns"]

        # Extract the assistant output just generated
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        # Record current turn trajectory: current observation comes from last feed (or 1st turn in prompt)
        # Here observation is the last user message in messages
        last_user_obs = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_obs = msg.get("content", "")
                break

        inst["trajectory"].append({
            "turn": cur,
            "observation": last_user_obs,
            "model_response": last_assistant,
        })

        if cur >= total:
            # All turns have been fed, model just produced the final result
            prediction = self._extract_prediction(messages)
            has_prediction_tag = self._has_prediction_tag(messages)
            memory = self._extract_memory(messages)
            previous_queries = self._extract_previous_queries(inst)

            skip_pointwise = kwargs.get("skip_pointwise_judge", False)

            if not has_prediction_tag:
                # Final turn missing <prediction> tag -> reward=0 directly, skip judge
                judge_reward = 0.0
                format_reward = 0.0
                reward = 0.0
            elif skip_pointwise:
                # Listwise training mode: skip pointwise judge, reward will be overridden by listwise
                # But still compute format_reward (pure local computation) for listwise reward mixing
                judge_reward = 0.0
                format_reward = self._compute_format_reward(messages)
                reward = 0.0
            else:
                judge_reward = await self._score_predictions(
                    prediction, inst["ground_truth"], memory, previous_queries
                )
                if judge_reward is None:
                    # Judge failed -> mark with NaN, GRPO will skip this sample
                    judge_reward = float("nan")
                    format_reward = float("nan")
                    reward = float("nan")
                else:
                    format_reward = self._compute_format_reward(messages)
                    is_validate = kwargs.get("is_validate", False)
                    fmt_w = float(os.getenv("ONEPRED_FORMAT_REWARD_WEIGHT", "0"))
                    if fmt_w > 0 and not is_validate:
                        reward = (1.0 - fmt_w) * judge_reward + fmt_w * format_reward
                    else:
                        reward = judge_reward

            # Save complete structured trace
            self._save_trace(instance_id, inst, prediction, reward)

            return True, "", reward, {
                "prediction": prediction,
                "has_prediction_tag": has_prediction_tag,
                "judge_reward": judge_reward,
                "format_reward": format_reward,
            }

        # More turns to feed
        turn = inst["turns"][cur]
        inst["current_turn"] = cur + 1
        remaining = total - inst["current_turn"]

        observation = self._format_observation(turn, remaining)

        return False, observation, 0.0, {}

    # ------------------------------------------------------------------
    # Trace saving
    # ------------------------------------------------------------------

    def _save_trace(self, instance_id: str, inst: dict, prediction: str, reward: float):
        """Save a complete structured trace."""
        trace_record = {
            "instance_id": instance_id,
            "ground_truth": inst["ground_truth"],
            "prediction": prediction,
            "reward": reward,
            "total_turns": inst["total_turns"],
            "turns": inst["trajectory"],
            # turns structure:
            # [
            #   {"turn": 0, "observation": "[Turn 1 dialogue]...", "model_response": "<memory>...</memory>"},
            #   {"turn": 1, "observation": "[Turn 2 dialogue]...", "model_response": "<memory>...</memory>"},
            #   ...
            #   {"turn": N, "observation": "[Turn N + system instruction]", "model_response": "<memory>...<prediction>...</prediction>"},
            # ]
        }

        # Sample print to stdout
        if inst.get("should_print"):
            self._print_trace(trace_record)

        # Write to file
        if TRACE_DIR:
            with self._trace_lock:
                self._trace_buffer.append(trace_record)
                if len(self._trace_buffer) >= self._trace_flush_count:
                    self._flush_traces()

    def _flush_traces(self):
        """Flush buffered traces to file."""
        if not TRACE_DIR or not self._trace_buffer:
            return
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, "traces.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for record in self._trace_buffer:
                # NaN reward needs to be replaced with null to ensure valid JSON output
                reward_val = record.get("reward")
                if isinstance(reward_val, float) and reward_val != reward_val:  # NaN check
                    record = {**record, "reward": None}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trace_buffer.clear()

    @staticmethod
    def _print_trace(record: dict):
        """Print a trace to stdout for log inspection."""
        print(f"\n{'='*70}")
        print(f"[TRACE] instance={record['instance_id'][:8]} | "
              f"turns={record['total_turns']+1} | "
              f"reward={record['reward']} | "
              f"gt={record['ground_truth'][:80]}")
        print(f"{'='*70}")
        for step in record["turns"]:
            t = step["turn"]
            obs = step["observation"]
            resp = step["model_response"]
            # Truncate for display
            obs_short = obs[:150].replace("\n", "\\n")
            resp_short = resp[:200].replace("\n", "\\n")
            print(f"  Turn {t}:")
            print(f"    [ENV → model] {obs_short}")
            print(f"    [model → ENV] {resp_short}")
        print(f"  Prediction: {record['prediction'][:100]!r}")
        print(f"  GroundTruth: {record['ground_truth'][:100]!r}")
        print(f"  Reward: {record['reward']}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _format_observation(self, turn: dict, remaining: int) -> str:
        """Format a turn <query, response, user_feedback> into environment observation text."""
        turn_idx = turn.get("turn_idx", 1)
        query = turn.get("query", "")
        response = turn.get("response", "")

        if MAX_RESPONSE_CHARS > 0 and len(response) > MAX_RESPONSE_CHARS:
            response = response[:MAX_RESPONSE_CHARS] + response_truncation_marker

        observation = f"{observation_header(turn_idx)}\n{observation_user_label}{query}\n{observation_response_label}{response}"

        feedback = turn.get("user_feedback", "")
        if feedback and feedback.strip():
            observation += f"\n{locale_feedback_label}{feedback.strip()}"

        if remaining == 0:
            observation += system_instruction

        return observation

    # ------------------------------------------------------------------
    # Extraction and scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_format_reward(messages: list[dict[str, Any]]) -> float:
        """Check format compliance of all assistant outputs, return [0, 1]."""
        score = 1.0

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            return 0.0

        # --- Check degenerate repetition patterns (return 0 immediately) ---
        for msg in assistant_msgs:
            content = msg.get("content", "")
            if re.search(r"(<memory>){3,}", content) or re.search(r"(<prediction>){3,}", content):
                return 0.0

        # --- Check final turn: must have <memory> and <prediction> ---
        final = assistant_msgs[-1].get("content", "")
        final_clean = re.sub(r"<think>.*?</think>", "", final, flags=re.DOTALL)
        if not re.search(r"<memory>.*?</memory>", final_clean, re.DOTALL):
            score -= 0.3
        if not re.search(r"<prediction>.*?</prediction>", final_clean, re.DOTALL):
            score -= 0.4

        # --- Check intermediate turns: should have <memory>, should not have <prediction> ---
        for msg in assistant_msgs[:-1]:
            content = msg.get("content", "")
            content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            if re.search(r"<prediction>", content_clean):
                score -= 0.1
            if not re.search(r"<memory>.*?</memory>", content_clean, re.DOTALL):
                score -= 0.05

        return max(0.0, score)

    async def _score_predictions(
        self, prediction_text: str, ground_truth: str, memory: str = "", previous_queries: str = ""
    ) -> float:
        """Send the full prediction text and ground truth to the judge for scoring at once."""
        if not prediction_text.strip():
            return 0.0
        score = await asyncio.to_thread(
            llm_judge_score, prediction_text, ground_truth, memory, previous_queries
        )
        return score  # None if judge failed

    @staticmethod
    def _extract_previous_queries(inst: dict) -> str:
        """Extract all historical user queries from instance data, newline-separated.

        Preferably uses all_queries (complete list including history[0]),
        falls back to extracting from turns for backward compatibility with old data.
        """
        all_queries = inst.get("all_queries", None)
        if all_queries is not None:
            return "\n".join(q for q in all_queries if q)

        # Backward compatibility with old data: turns only contains history[1:]
        queries = []
        for turn in inst.get("turns", []):
            query = turn.get("query", "")
            if query:
                queries.append(query)
        return "\n".join(queries)

    @staticmethod
    def _extract_memory(messages: list[dict[str, Any]]) -> str:
        """Extract the last <memory> tag content from the last assistant message."""
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        matches = re.findall(r"<memory>(.*?)</memory>", content_clean, re.DOTALL)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _has_prediction_tag(messages: list[dict[str, Any]]) -> bool:
        """Check whether a <prediction> tag exists in the last assistant message."""
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return bool(re.search(r"<prediction>.*?</prediction>", content_clean, re.DOTALL))

    @staticmethod
    def _extract_prediction(messages: list[dict[str, Any]]) -> str:
        """Extract <prediction> tag content from the last assistant message."""
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        # Remove <think>...</think> content first to avoid extracting <prediction> tags from thinking
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        match = re.search(r"<prediction>(.*?)</prediction>", content_clean, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = [line.strip() for line in content_clean.split("\n") if line.strip()]
        return lines[-1] if lines else content_clean

    async def finalize_interaction(self, instance_id: str, **kwargs):
        self._instances.pop(instance_id, None)
        # Ensure remaining buffered data is flushed
        if TRACE_DIR:
            with self._trace_lock:
                self._flush_traces()
