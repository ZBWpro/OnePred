"""
RecurrentAgentLoopManager: custom AgentLoopManager for Approach B.

Overrides ``generate_sequences()`` to:
1. Call the parent's agent loop execution (which invokes RecurrentMemoryAgentLoop).
2. Extract ``per_turn_conversations`` from each sample's extra_fields.
3. Expand every turn into an independent (prompt, response) batch row.
4. Construct a new DataProto with ``sample_index`` and ``final_mask`` tensors.

The expanded batch is returned directly to RecurrentRayPPOTrainer.fit(),
which handles reward computation, advantage, and actor update.
"""

import logging
import os

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager

from verl.utils.tokenizer import normalize_token_ids

from onepred.recurrent_utils import create_position_ids, pad_tensor_list_to_length

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RecurrentAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that expands per-turn conversations into independent batch rows."""

    def _get_tokenizer(self):
        """Lazy-load and cache the tokenizer (avoid reloading from disk every step)."""
        if not hasattr(self, "_cached_tokenizer") or self._cached_tokenizer is None:
            from transformers import AutoTokenizer
            model_path = self.config.actor_rollout_ref.model.path
            self._cached_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return self._cached_tokenizer

    def _expand_turns(self, base_output: DataProto) -> tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Expand the base output (1 row per sample) into (1 row per turn).

        Each row becomes an independent (prompt, response) for training.
        Returns:
            expanded: DataProto with shape (total_turns, seq_len)
            final_mask: BoolTensor of shape (total_turns,)
            sample_index: LongTensor of shape (total_turns,) mapping each turn to original sample
        """
        tokenizer = self._get_tokenizer()

        prompt_length = self.config.data.max_prompt_length
        response_length = self.config.data.max_response_length

        all_prompt_ids = []
        all_response_ids = []
        all_response_masks = []
        sample_index_list = []
        final_mask_list = []

        # Non-tensor fields to propagate (per original sample)
        original_non_tensor = base_output.non_tensor_batch

        # Expanded non-tensor batch
        expanded_non_tensor_keys = {k for k in original_non_tensor.keys()
                                     if k not in ("per_turn_conversations", "per_turn_reward")}
        expanded_non_tensor = {k: [] for k in expanded_non_tensor_keys}

        apply_chat_template_kwargs = {}
        enable_thinking = self.config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", None)
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        bsz = len(base_output)
        for i in range(bsz):
            per_turn_convs = original_non_tensor["per_turn_conversations"][i]
            if per_turn_convs is None or len(per_turn_convs) == 0:
                # Fallback: treat the whole sample as 1 turn
                per_turn_convs = [None]

            n_turns = len(per_turn_convs)

            for t, conv in enumerate(per_turn_convs):
                is_final = (t == n_turns - 1)

                if conv is not None:
                    # Split conversation into prompt messages and full messages
                    prompt_msgs = [m for m in conv if m["role"] != "assistant"]
                    full_msgs = conv

                    # Tokenize prompt
                    p_ids_raw = tokenizer.apply_chat_template(
                        prompt_msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                        **apply_chat_template_kwargs,
                    )
                    p_ids = normalize_token_ids(p_ids_raw)
                    # Tokenize full conversation
                    f_ids_raw = tokenizer.apply_chat_template(
                        full_msgs,
                        add_generation_prompt=False,
                        tokenize=True,
                        **apply_chat_template_kwargs,
                    )
                    f_ids = normalize_token_ids(f_ids_raw)
                    r_ids = f_ids[len(p_ids):]
                    r_mask = [1] * len(r_ids)
                else:
                    # Fallback: use the base output's prompt/response for this sample
                    p_ids = base_output.batch["prompts"][i].cpu().tolist()
                    # Remove padding (left-padded with pad_token_id)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    while p_ids and p_ids[0] == pad_id:
                        p_ids = p_ids[1:]
                    r_ids = base_output.batch["responses"][i].cpu().tolist()
                    r_mask_tensor = base_output.batch["response_mask"][i].cpu().tolist()
                    # Remove right padding
                    while r_ids and r_ids[-1] == pad_id:
                        r_ids = r_ids[:-1]
                        r_mask_tensor = r_mask_tensor[:-1]
                    r_mask = r_mask_tensor

                # Truncate response to max_response_length
                r_ids = r_ids[:response_length]
                r_mask = r_mask[:response_length]

                all_prompt_ids.append(torch.tensor(p_ids, dtype=torch.long))
                all_response_ids.append(torch.tensor(r_ids, dtype=torch.long))
                all_response_masks.append(torch.tensor(r_mask, dtype=torch.long))

                sample_index_list.append(i)
                final_mask_list.append(is_final)

                # Propagate non-tensor fields
                for k in expanded_non_tensor_keys:
                    expanded_non_tensor[k].append(original_non_tensor[k][i])

        # Pad to uniform lengths
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        prompt_ids_padded, prompt_attn_mask = pad_tensor_list_to_length(
            all_prompt_ids,
            pad_token_id=pad_id,
            max_length=prompt_length,
            left_pad=True,
            return_mask=True,
        )
        response_ids_padded, response_attn_mask = pad_tensor_list_to_length(
            all_response_ids,
            pad_token_id=pad_id,
            max_length=response_length,
            left_pad=False,
            return_mask=True,
        )
        response_mask_padded = pad_tensor_list_to_length(
            all_response_masks,
            pad_token_id=0,
            max_length=response_length,
            left_pad=False,
        )
        # response_mask should be masked by response_attn_mask
        response_mask_padded = response_mask_padded * response_attn_mask.long()

        input_ids = torch.cat([prompt_ids_padded, response_ids_padded], dim=1)
        attention_mask = torch.cat([prompt_attn_mask.long(), response_attn_mask.long()], dim=1)
        position_ids = create_position_ids(attention_mask)

        sample_index = torch.tensor(sample_index_list, dtype=torch.long)
        final_mask = torch.tensor(final_mask_list, dtype=torch.bool)

        # Build rm_scores for final turns (reward placed at last valid response position)
        total_turns = len(all_prompt_ids)
        rm_scores = torch.zeros(total_turns, response_length, dtype=torch.float32)
        for idx in range(total_turns):
            if final_mask[idx]:
                orig_i = sample_index[idx].item()
                reward_val = original_non_tensor.get("per_turn_reward", np.zeros(bsz))[orig_i]
                if reward_val is None:
                    reward_val = 0.0
                # Place reward at last valid response token
                valid_len = response_attn_mask[idx].sum().item()
                if valid_len > 0:
                    rm_scores[idx, valid_len - 1] = float(reward_val)

        batch = TensorDict(
            {
                "prompts": prompt_ids_padded,
                "responses": response_ids_padded,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask_padded,
                "rm_scores": rm_scores,
                "final_mask": final_mask,
                "sample_index": sample_index,
            },
            batch_size=total_turns,
        )

        non_tensor = {}
        for k, v in expanded_non_tensor.items():
            arr = np.empty(len(v), dtype=object)
            arr[:] = v
            non_tensor[k] = arr

        expanded = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor,
            meta_info=base_output.meta_info,
        )

        return expanded, final_mask, sample_index

    def _apply_listwise_reward(self, base_output: DataProto):
        """Apply listwise reward: compare all rollouts for the same prompt together.

        Groups rollouts by prompt (consecutive n_rollouts rows share the same prompt),
        calls the listwise judge for each group, and overwrites per_turn_reward.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from onepred.judges.listwise_judge import listwise_judge_score

        n_rollouts = self.config.actor_rollout_ref.rollout.n
        non_tensor = base_output.non_tensor_batch
        n_samples = len(base_output)
        n_groups = n_samples // n_rollouts

        if n_groups == 0:
            return

        if n_samples % n_rollouts != 0:
            logger.warning(
                "Listwise reward: n_samples=%d not divisible by n_rollouts=%d, "
                "trailing %d samples will keep reward=0.0",
                n_samples, n_rollouts, n_samples % n_rollouts,
            )

        # Check required fields are present
        required_fields = ["prediction", "ground_truth", "previous_queries"]
        missing = [f for f in required_fields if f not in non_tensor]
        if missing:
            logger.warning("Listwise reward: missing fields %s, skipping", missing)
            return

        # Prepare groups
        groups = []
        for g in range(n_groups):
            start = g * n_rollouts
            end = start + n_rollouts
            predictions = [non_tensor["prediction"][i] for i in range(start, end)]
            gt = non_tensor["ground_truth"][start]
            prev_q = non_tensor["previous_queries"][start]
            groups.append((start, end, predictions, gt, prev_q))

        # Call listwise judge for each group (concurrent across groups)
        with ThreadPoolExecutor(max_workers=min(n_groups, 4)) as executor:
            futures = {}
            for g_idx, (start, end, predictions, gt, prev_q) in enumerate(groups):
                future = executor.submit(listwise_judge_score, predictions, gt, prev_q)
                futures[future] = (g_idx, start, end)

            for future in as_completed(futures):
                g_idx, start, end = futures[future]
                try:
                    rewards = future.result()
                    if rewards is None:
                        # All models failed → all 0
                        rewards = [0.0] * n_rollouts
                except Exception as e:
                    logger.warning("Listwise judge failed for group %d: %s", g_idx, e)
                    rewards = [0.0] * n_rollouts

                # Overwrite per_turn_reward (mix listwise ranking with format reward)
                for i, r in enumerate(rewards):
                    fmt_r = non_tensor.get("format_reward", np.zeros(n_samples))[start + i]
                    if fmt_r is None:
                        fmt_r = 0.0
                    non_tensor["per_turn_reward"][start + i] = 0.9 * r + 0.1 * float(fmt_r)

        logger.info(
            "Listwise reward applied: %d groups, %d rollouts/group",
            n_groups, n_rollouts,
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Override to expand per-turn conversations after generation.

        Returns a DataProto where each row is an independent (prompt, response) turn,
        with ``final_mask`` and ``sample_index`` in batch tensors.
        """
        is_validate = prompts.meta_info.get("validate", False)

        # Pass validate flag to interaction so it can adjust reward calculation
        bsz = len(prompts)
        if is_validate:
            for i in range(bsz):
                prompts.non_tensor_batch["extra_info"][i]["interaction_kwargs"]["is_validate"] = True

        # Tell interaction to skip pointwise judge during training when listwise is enabled
        if os.getenv("USE_LISTWISE_REWARD") == "1" and not is_validate:
            for i in range(bsz):
                prompts.non_tensor_batch["extra_info"][i]["interaction_kwargs"]["skip_pointwise_judge"] = True

        # Call parent's generate_sequences to run agent loops
        base_output = super().generate_sequences(prompts)

        # Skip expansion during validation — _validate() expects batch size to stay the same
        if is_validate:
            return base_output

        # Check if per_turn_conversations are available
        if ("per_turn_conversations" not in base_output.non_tensor_batch
                or base_output.non_tensor_batch["per_turn_conversations"][0] is None):
            # Fallback: no expansion needed (e.g., single-turn)
            return base_output

        # Apply listwise reward if enabled (overwrites per_turn_reward before expansion)
        if os.getenv("USE_LISTWISE_REWARD") == "1":
            self._apply_listwise_reward(base_output)

        # Expand turns
        expanded, final_mask, sample_index = self._expand_turns(base_output)

        return expanded
