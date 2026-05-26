"""
RecurrentMemoryAgentLoop (Approach B): Recurrent memory Agent Loop with per-turn independent training.

Core changes (relative to Approach A):
  - run() returns AgentLoopOutput with extra_fields containing:
      per_turn_conversations: list[list[dict]] -- per-turn independent full conversations
      per_turn_reward: float -- final turn reward (scalar)
  - prompt_ids / response_ids / response_mask only contain data from the final turn
    (for compatibility with AgentLoopWorker._agent_loop_postprocess)
  - During training, each turn is expanded into independent (prompt, response) samples,
    handled by RecurrentAgentLoopManager for expansion and tokenization

Each turn only lets the model see system_prompt + memory + current observation; memory is the sole cross-turn information bottleneck.
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from onepred.locale import (
    MEMORY_USER_TEMPLATE,
    truncation_marker,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Maximum memory characters (character-level safety fallback)
MAX_MEMORY_CHARS = 4500

# Maximum memory tokens (precise token-level truncation, executed by tokenizer in agent loop)
MAX_MEMORY_TOKENS = 1500


def _extract_memory(text: str) -> str:
    """Extract <memory>...</memory> content from model output, truncate if too long."""
    # Remove <think>...</think> content first to avoid extracting <memory> tags from thinking
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"<memory>(.*?)</memory>", text_clean, re.DOTALL)
    if match:
        mem = match.group(1).strip()
    else:
        # Fallback: model did not output <memory> tag, take text outside of think
        mem = text_clean.strip()
    if len(mem) > MAX_MEMORY_CHARS:
        mem = mem[:MAX_MEMORY_CHARS] + truncation_marker
    return mem


@register("recurrent_memory_agent")
class RecurrentMemoryAgentLoop(AgentLoopBase):
    """Context-independent recurrent memory Agent Loop (Approach B).

    For each generation turn, constructs a fresh prompt = [system, user(memory + observation)],
    with only the memory text serving as the cross-turn information carrier.

    The extra_fields in the output contain per-turn independent conversations for RecurrentAgentLoopManager to expand.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns

        # Initialize interaction
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = initialize_interactions_from_config(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # 1. Get initial messages from kwargs
        initial_messages = list(kwargs["raw_prompt"])

        # Extract system message and 1st turn user observation
        system_msg = None
        first_user_msg = None
        for msg in initial_messages:
            if msg["role"] == "system":
                system_msg = msg
            elif msg["role"] == "user":
                first_user_msg = msg

        if system_msg is None:
            raise ValueError("initial messages must contain a system message")
        if first_user_msg is None:
            raise ValueError("initial messages must contain a user message")

        # 2. Initialize interaction
        request_id = uuid4().hex
        interaction: Optional[BaseInteraction] = None
        interaction_kwargs: dict[str, Any] = {}

        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found. "
                    f"Available: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            # Pass skip_pointwise_judge flag from manager to interaction
            if kwargs.get("skip_pointwise_judge"):
                interaction_kwargs["skip_pointwise_judge"] = True
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # 3. Initialize state
        memory = ""
        reward_score: Optional[float] = None
        turn_scores: list[float] = []
        metrics: dict[str, Any] = {}
        interaction_extra_metrics: dict[str, Any] = {}
        assistant_turns = 0
        user_turns = 0

        # Per-turn full conversations [system, user, assistant]
        per_turn_conversations: list[list[dict]] = []

        # Maintain messages_for_interaction (interaction needs full history to extract assistant output)
        messages_for_interaction = list(initial_messages)

        # 4. Multi-turn loop
        terminated = False
        current_observation = None  # will be set after first interaction response

        while not terminated:
            # -- Construct fresh prompt for this turn --
            if assistant_turns == 0:
                fresh_messages = list(initial_messages)
            else:
                user_content = MEMORY_USER_TEMPLATE.format(
                    memory=memory,
                    observation=current_observation,
                )
                fresh_messages = [
                    system_msg,
                    {"role": "user", "content": user_content},
                ]

            # -- Tokenize fresh prompt --
            fresh_prompt_ids = await self.apply_chat_template(fresh_messages)
            # Truncate to max prompt length (keep the tail to preserve user message)
            if len(fresh_prompt_ids) > self.prompt_length:
                fresh_prompt_ids = fresh_prompt_ids[-self.prompt_length:]

            # -- Adjust max_tokens --
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = min(
                turn_sampling_params.get("max_tokens", self.response_length),
                self.response_length,
            )

            # -- Generate --
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=fresh_prompt_ids,
                    sampling_params=turn_sampling_params,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            assistant_turns += 1

            # -- Decode assistant text --
            gen_token_ids = output.token_ids
            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
            )
            memory = _extract_memory(assistant_text)

            # Token-level memory truncation (precise control within MAX_MEMORY_TOKENS)
            mem_token_ids = self.tokenizer.encode(memory, add_special_tokens=False)
            if len(mem_token_ids) > MAX_MEMORY_TOKENS:
                mem_token_ids = mem_token_ids[:MAX_MEMORY_TOKENS]
                memory = self.tokenizer.decode(mem_token_ids, skip_special_tokens=True).strip() + truncation_marker

            # -- Record this turn's full conversation --
            turn_conversation = list(fresh_messages) + [{"role": "assistant", "content": assistant_text}]
            per_turn_conversations.append(turn_conversation)

            # -- Check max turn limit --
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                reward_score = 0.0
                break

            # -- Call interaction to get next turn observation --
            if interaction is not None:
                messages_for_interaction.append({"role": "assistant", "content": assistant_text})

                (
                    should_terminate,
                    observation_text,
                    reward,
                    extra_metrics,
                ) = await interaction.generate_response(
                    request_id, messages_for_interaction, **interaction_kwargs
                )
                if extra_metrics:
                    interaction_extra_metrics = extra_metrics

                user_turns += 1

                if reward is not None:
                    turn_scores.append(reward)

                if should_terminate:
                    reward_score = reward
                    terminated = True
                else:
                    messages_for_interaction.append({"role": "user", "content": observation_text})
                    current_observation = observation_text
            else:
                terminated = True

            if self.max_user_turns and user_turns >= self.max_user_turns:
                if reward_score is None:
                    reward_score = 0.0
                break

        # 5. Construct AgentLoopOutput
        # prompt_ids / response_ids / response_mask use data from the final turn
        # (for compatibility with AgentLoopWorker._agent_loop_postprocess padding)
        if per_turn_conversations:
            final_conv = per_turn_conversations[-1]
            # prompt = messages before assistant
            final_prompt_msgs = [m for m in final_conv if m["role"] != "assistant"]
            final_prompt_ids = await self.apply_chat_template(final_prompt_msgs)

            # response = tokenize full conversation, then strip prompt
            final_full_ids = await self.apply_chat_template(final_conv)
            final_response_ids = final_full_ids[len(final_prompt_ids):]
            final_response_mask = [1] * len(final_response_ids)

            # Truncate prompt to max length (keep tail)
            if len(final_prompt_ids) > self.prompt_length:
                final_prompt_ids = final_prompt_ids[-self.prompt_length:]
        else:
            # should not happen, but provide fallback
            final_prompt_ids = await self.apply_chat_template(initial_messages)
            final_response_ids = []
            final_response_mask = []

        # Build previous_queries from all_queries for listwise reward
        all_queries = interaction_kwargs.get("all_queries") or []
        previous_queries_str = "\n".join(q for q in all_queries if q)

        output = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[:self.response_length],
            response_mask=final_response_mask[:self.response_length],
            response_logprobs=None,
            reward_score=reward_score,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": turn_scores,
                "tool_rewards": [],
                "per_turn_conversations": per_turn_conversations,
                "per_turn_reward": reward_score if reward_score is not None else 0.0,
                # Listwise reward fields (for second-stage training)
                "prediction": interaction_extra_metrics.get("prediction", "") if interaction_extra_metrics.get("has_prediction_tag", True) else "",
                "format_reward": interaction_extra_metrics.get("format_reward", 0.0),
                "final_memory": memory,
                "ground_truth": interaction_kwargs.get("ground_truth", ""),
                "previous_queries": previous_queries_str,
            },
        )

        # Ensure interaction resource cleanup and trace flush
        if interaction is not None:
            await interaction.finalize_interaction(request_id)

        return output
