"""
RecurrentDataParallelPPOActor: overrides update_policy() for Approach B.

Changes from the base DataParallelPPOActor.update_policy():
  1. Check for ``no_padding_mask`` in data.batch; if present, use indexing_proto
     to drop padded empty samples before training.
  2. Use ``td_split`` instead of ``batch.split`` for mini-batch / micro-batch
     splitting to support unequal chunk sizes.
  3. Token-weighted gradient accumulation for variable-sized batches.

Reference: MemAgent/verl/workers/actor/dp_actor.py:236-406
"""

import logging
import os

import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

from onepred.recurrent_utils import indexing_proto, td_split

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RecurrentDataParallelPPOActor(DataParallelPPOActor):
    """DataParallelPPOActor with recurrent multi-turn support."""

    @staticmethod
    def _grad_acc_mode(loss_agg_mode: str) -> str:
        """Determine gradient accumulation mode from loss aggregation mode."""
        if loss_agg_mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"):
            return loss_agg_mode.split("-")[0]
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "response_mask",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # Check for padding mask (recurrent mode)
        padded = "no_padding_mask" in data.batch
        if padded:
            proto = data.select(batch_keys=select_keys)
            # Drop empty padding samples to avoid impacting sequence-level averaging loss
            batch = indexing_proto(proto, data.batch["no_padding_mask"]).batch
        else:
            batch = data.select(batch_keys=select_keys).batch

        # Split into mini-batches
        if padded:
            num_mini_batches = self.config.train_batch_size // self.config.ppo_mini_batch_size
            dataloader = td_split(batch, num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                mini_batch = mini_batch_data

                # Split into micro-batches
                if padded:
                    num_micro_batches = -(-len(mini_batch) // self.config.ppo_micro_batch_size_per_gpu)
                    micro_batches = td_split(mini_batch, num_micro_batches)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # Total token count for this mini-batch (for token-weighted grad accumulation)
                mini_batch_token_nums = mini_batch["response_mask"].sum()

                for micro_data in micro_batches:
                    if isinstance(micro_data, DataProto):
                        micro_data = {
                            **micro_data.batch.to(torch.cuda.current_device()),
                            **micro_data.non_tensor_batch,
                        }
                    else:
                        micro_data = micro_data.to(torch.cuda.current_device())

                    response_mask = micro_data["response_mask"]
                    old_log_prob = micro_data["old_log_probs"]
                    advantages = micro_data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    outputs = self._forward_micro_batch(
                        micro_batch=micro_data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    # verl's _forward_micro_batch returns a dict with "log_probs" and optional "entropys"
                    log_prob = outputs["log_probs"]
                    entropy = outputs.get("entropys") if calculate_entropy else None

                    from verl.trainer.ppo.core_algos import compute_policy_loss
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_data["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # Token-weighted gradient accumulation for variable batch sizes
                    acc_mode = self._grad_acc_mode(loss_agg_mode)
                    if acc_mode == "seq":
                        loss = policy_loss * (len(micro_data) / len(mini_batch))
                    elif acc_mode == "token":
                        loss = policy_loss * (response_mask.sum().item() / mini_batch_token_nums.item())
                    else:
                        raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_mode}")

                    loss.backward()

                    step_data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, step_data)

                grad_norm = self._optimizer_step()
                step_data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, step_data)
        self.actor_optimizer.zero_grad()
        return metrics
