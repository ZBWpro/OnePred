"""
RecurrentRayPPOTrainer: subclass of RayPPOTrainer for Approach B.

Overrides ``fit()`` to handle the expanded multi-turn batch:
  - No repeat+union (the expanded batch from RecurrentAgentLoopManager already contains all turns)
  - Disable balance_batch (sample_index ordering must be preserved)
  - Reward computed only on final_mask=True rows
  - 1D GRPO advantage broadcast via sample_index
  - Graceful DP padding before actor update
  - Unpad for metrics collection
"""

import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from onepred.recurrent_utils import (
    compute_1D_grpo_advantage,
    final_batch,
    graceful_padding,
    indexing_proto,
)


_logger = logging.getLogger(__name__)


class RecurrentRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with Approach B multi-turn training support."""

    def fit(self):
        """Training loop with per-turn independent training.

        Key differences from base RayPPOTrainer.fit():
        1. generate_sequences returns expanded batch with sample_index + final_mask
        2. No batch.repeat() + batch.union() — the expanded batch IS the training batch
        3. Reward extracted only from final_mask rows
        4. 1D GRPO advantage broadcast to all turns via sample_index
        5. Graceful DP padding before actor update
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # Validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # Add uid to batch (per original sample)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                is_last_step = self.global_steps >= self.total_training_steps

                # Save uids for GRPO grouping (before batch is overwritten by expanded output)
                n_rollouts = self.config.actor_rollout_ref.rollout.n
                repeated_uids = np.repeat(batch.non_tensor_batch["uid"], n_rollouts)

                with marked_timer("step", timing_raw):
                    # ===== GENERATION =====
                    with marked_timer("gen", timing_raw, color="red"):
                        # Repeat for n rollouts
                        gen_batch_output = gen_batch.repeat(
                            repeat_times=n_rollouts, interleave=True
                        )
                        gen_batch_output.meta_info["global_steps"] = self.global_steps

                        # RecurrentAgentLoopManager.generate_sequences returns expanded batch
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()

                        if "timing" in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info.pop("timing"))

                    # The expanded batch has sample_index and final_mask
                    final_mask = gen_batch_output.batch["final_mask"]
                    sample_index = gen_batch_output.batch["sample_index"]

                    # The expanded batch IS our training batch
                    batch = gen_batch_output

                    if "response_mask" not in batch.batch.keys():
                        from verl.trainer.ppo.ray_trainer import compute_response_mask
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # NOTE: Do NOT balance batch — it would break sample_index ordering

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ===== REWARD =====
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Extract reward from rm_scores (already placed by RecurrentAgentLoopManager)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # ===== OLD LOG PROBS =====
                    # Pad batch for divisibility before log_prob computation
                    batch_for_logprob, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch_for_logprob)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch_for_logprob.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch_for_logprob = batch_for_logprob.union(old_log_prob)

                    # Unpad back
                    batch_for_logprob = unpad_dataproto(batch_for_logprob, pad_size)
                    # Transfer old_log_probs to batch
                    batch.batch["old_log_probs"] = batch_for_logprob.batch["old_log_probs"]

                    # ===== REF LOG PROBS =====
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            batch_for_ref, ref_pad_size = pad_dataproto_to_divisor(
                                batch, self.actor_rollout_wg.world_size
                            )
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch_for_ref)
                            batch_for_ref = unpad_dataproto(batch_for_ref.union(ref_log_prob), ref_pad_size)
                            batch.batch["ref_log_prob"] = batch_for_ref.batch["ref_log_prob"]

                    # ===== ADVANTAGE =====
                    with marked_timer("adv", timing_raw, color="brown"):
                        # Extract reward only from final turns
                        reward_batch = final_batch(batch, final_mask, sample_index)

                        assert len(reward_batch) == len(repeated_uids), (
                            f"final_batch size ({len(reward_batch)}) != repeated_uids size ({len(repeated_uids)}). "
                            f"Each trajectory must have exactly one final turn."
                        )

                        final_reward_tensor = reward_batch.batch["rm_scores"]

                        # Mask out NaN-reward trajectories (judge failures) from training
                        final_reward_scalar = final_reward_tensor.sum(dim=-1)  # (num_traj,)
                        nan_traj_mask = torch.isnan(final_reward_scalar)       # True = skip
                        if nan_traj_mask.any():
                            n_nan = nan_traj_mask.sum().item()
                            _logger.warning(
                                "Skipping %d/%d NaN-reward trajectories in advantage computation",
                                n_nan, len(final_reward_scalar),
                            )
                            # Zero response_mask for all turns belonging to NaN trajectories
                            nan_turn_mask = nan_traj_mask[sample_index]  # broadcast to all turns
                            batch.batch["response_mask"][nan_turn_mask] = 0

                        # Compute 1D GRPO advantage on final turns only
                        # (must run BEFORE NaN cleanup so NaN samples are properly skipped)
                        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        advantage_scalar = compute_1D_grpo_advantage(
                            token_level_rewards=final_reward_tensor,
                            index=repeated_uids,
                            use_adv=norm_adv,
                        )

                        # Clean NaN from reward tensor AFTER advantage computation
                        # to prevent metric pollution downstream
                        if nan_traj_mask.any():
                            final_reward_tensor[nan_traj_mask] = 0

                        # Broadcast advantage to all turns via sample_index
                        advantage_scalar = advantage_scalar[sample_index]

                        # Tile to response_length and mask
                        response_length = batch.batch["responses"].size(-1)
                        eos_mask = batch.batch["response_mask"]
                        advantages = advantage_scalar.unsqueeze(-1).tile([1, response_length]) * eos_mask
                        batch.batch["advantages"] = advantages
                        batch.batch["returns"] = advantages

                        # Broadcast trajectory reward to all turns, placing at each
                        # turn's own last valid response position.
                        # (1) scalar reward per trajectory
                        reward_scalar = final_reward_tensor.sum(dim=-1)       # (num_traj,)
                        reward_per_turn = reward_scalar[sample_index]         # (total_turns,)
                        # (2) place at each turn's last valid response token
                        token_level_scores = torch.zeros_like(
                            batch.batch["response_mask"], dtype=torch.float32
                        )
                        valid_lens = batch.batch["response_mask"].sum(dim=-1).long()
                        has_tokens = valid_lens > 0
                        row_idx = torch.arange(len(token_level_scores))[has_tokens]
                        col_idx = (valid_lens[has_tokens] - 1)
                        token_level_scores[row_idx, col_idx] = reward_per_turn[has_tokens]
                        batch.batch["token_level_scores"] = token_level_scores
                        batch.batch["token_level_rewards"] = token_level_scores

                    # ===== UPDATE ACTOR =====
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Graceful DP padding
                        wsz = self.actor_rollout_wg.world_size
                        if len(batch) % wsz != 0:
                            padding_index, no_padding_mask = graceful_padding(len(batch), wsz)
                            batch = indexing_proto(batch, padding_index)
                            batch.batch["attention_mask"][~no_padding_mask, :] = 0
                            batch.batch["response_mask"][~no_padding_mask, :] = 0
                            batch.batch["no_padding_mask"] = no_padding_mask
                            batch.meta_info["padded"] = True
                        else:
                            batch.batch["no_padding_mask"] = torch.ones(len(batch), dtype=torch.bool)

                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # Validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })

                # Remove padding for metrics
                if batch.meta_info.get("padded", False):
                    batch = indexing_proto(batch, batch.batch["no_padding_mask"])

                # Recurrent-specific metrics
                metrics["recurrent/total_turns"] = len(final_mask)
                metrics["recurrent/num_samples"] = final_mask.sum().item()
                metrics["recurrent/avg_turns_per_sample"] = len(final_mask) / max(final_mask.sum().item(), 1)

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
