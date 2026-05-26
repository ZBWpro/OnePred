# Verl Patch Note

This is verl v0.8.0.dev with one patch applied.

## Patch: `verl/utils/config.py`

The `validate_config` function is modified to multiply `train_batch_size` by `rollout.n` before passing it to `actor_config.validate()`. This fixes a validation issue when GRPO's `rollout_n` parameter expands the effective batch size.

Without this patch, verl incorrectly rejects valid configurations where `train_batch_size * rollout_n` exceeds the per-device capacity, even though the actual per-step batch is correctly sharded.
