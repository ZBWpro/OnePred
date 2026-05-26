"""
Utility functions for Approach B recurrent memory agent training.

Adapted from MemAgent/recurrent/utils.py to avoid runtime dependency on MemAgent.
"""

from collections import defaultdict
from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------

def graceful_padding(bsz: int, group_nums: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create index mapping for padding batches not evenly divisible by group_nums (world_size).

    Returns:
        padding_index: index tensor where -1 marks padding slots (mapped to last real sample).
        no_padding_mask: bool tensor, True for real samples, False for padding.
    """
    group_size = bsz // group_nums + 1
    remainder = bsz % group_nums
    if not remainder:
        return torch.arange(bsz), torch.ones(bsz, dtype=torch.bool)

    no_padding_mask = torch.tensor(
        [1 if i // group_size < remainder or i % group_size else 0
         for i in range(group_nums * group_size)],
        dtype=torch.int,
    )
    padding_index = torch.cumsum(no_padding_mask, dim=0) - 1
    padding_index[~no_padding_mask.bool()] = -1
    return padding_index, no_padding_mask.bool()


def pad_tensor_list_to_length(
    response: List[torch.LongTensor],
    pad_token_id: int,
    max_length: int = None,
    left_pad: bool = True,
    return_mask: bool = False,
):
    """Pad a list of 1-D LongTensors to uniform 2-D tensor.

    ~20x faster than verl's ``pad_2d_list_to_length`` because it works with 1-D tensors.
    """
    response_length = max(len(t) for t in response)
    target_length = max(max_length, response_length) if max_length is not None and max_length > response_length else response_length
    full_long = lambda length, v: torch.full((length,), fill_value=v, dtype=response[0].dtype)

    if left_pad:
        padded = [torch.cat([full_long(target_length - len(t), pad_token_id), t]) for t in response]
    else:
        padded = [torch.cat([t, full_long(target_length - len(t), pad_token_id)]) for t in response]
    padded = torch.stack(padded)

    if return_mask:
        mask = torch.full(padded.shape, True, dtype=torch.bool)
        if left_pad:
            for i, t in enumerate(response):
                mask[i, :target_length - len(t)] = False
        else:
            for i, t in enumerate(response):
                pad_len = target_length - len(t)
                if pad_len > 0:
                    mask[i, -pad_len:] = False
        return padded, mask
    return padded


def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create position ids from attention mask."""
    return torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, min=0)


# ---------------------------------------------------------------------------
# DataProto indexing
# ---------------------------------------------------------------------------

def indexing_proto(proto: DataProto, indices) -> DataProto:
    """Fancy-index a DataProto by indices (tensor, list, or ndarray)."""
    if isinstance(indices, torch.Tensor):
        cpu_indices = indices.cpu()
        if indices.device != proto.batch.device:
            indices = indices.to(proto.batch.device)
    else:
        cpu_indices = indices
    return DataProto.from_dict(
        tensors={k: v[indices] for k, v in proto.batch.items()},
        non_tensors={k: v[cpu_indices] for k, v in proto.non_tensor_batch.items()},
        meta_info=proto.meta_info,
    )


def td_split(td: TensorDict, sections: int) -> list[TensorDict]:
    """Split TensorDict along dim-0 into ``sections`` chunks (supports unequal sizes)."""
    if len(td) < sections:
        raise ValueError(f"len(td)={len(td)} < sections={sections}")
    tensors_split = {k: torch.tensor_split(v, sections) for k, v in td.items()}
    return [TensorDict.from_dict({k: v[i] for k, v in tensors_split.items()}) for i in range(sections)]


# ---------------------------------------------------------------------------
# Index helpers for final_batch
# ---------------------------------------------------------------------------

def reverse_indices(tensor: torch.Tensor) -> torch.Tensor:
    """Given a permutation tensor, return its inverse permutation."""
    unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    assert len(unique) == len(tensor), "Input tensor has duplicated elements."
    indices = torch.scatter_reduce(
        torch.zeros_like(unique, dtype=torch.long, device=tensor.device),
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(0), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    return indices


def final_batch(batch: DataProto, final_mask: torch.Tensor, sample_index: torch.Tensor) -> DataProto:
    """Extract final-turn rows from an expanded batch, reordered to match original input order."""
    final_output = indexing_proto(batch, final_mask)
    final_index = sample_index[final_mask]
    final_output.reorder(reverse_indices(final_index))
    return final_output


# ---------------------------------------------------------------------------
# GRPO advantage
# ---------------------------------------------------------------------------

def compute_1D_grpo_advantage(
    token_level_rewards: torch.Tensor,
    index,
    epsilon: float = 1e-6,
    use_adv: bool = True,
) -> torch.Tensor:
    """Compute 1-D scalar GRPO advantage per sample (not per token).

    Groups by ``index`` (uid), normalizes within each group.
    Samples with NaN reward are skipped (advantage=0, not included in group stats).

    Returns:
        advantages: shape ``(bsz,)``
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        nan_mask = torch.isnan(scores)
        for i in range(bsz):
            if not nan_mask[i]:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # Single sample cannot be compared within group, set advantage to 0 (skip this sample)
                id2mean[idx] = id2score[idx][0]
                if use_adv:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                if use_adv:
                    id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score for prompt index: {idx}")
        for i in range(bsz):
            if nan_mask[i]:
                scores[i] = 0.0
            elif index[i] not in id2mean:
                # All samples in this group were NaN — skip
                scores[i] = 0.0
            elif use_adv:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    return scores
