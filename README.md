# OnePred

Reinforcement learning for next-query prediction in multi-turn conversations, using recursive memory as an information bottleneck.

## Method

Given a multi-turn conversation history, OnePred predicts what the user will ask next. The core design is a **recursive memory agent loop** trained with GRPO:

1. At each turn, the model receives only `system_prompt + <memory> + current_observation`
2. The model generates an updated `<memory>` block (its sole cross-turn context)
3. At the final turn, the model additionally outputs `<prediction>` with candidate next queries
4. A multi-model LLM judge (Claude / Gemini / GPT majority vote) scores the predictions as reward

The memory acts as an information bottleneck: the model must learn *what to remember* because it cannot access raw history at later turns.

## Directory Structure

```
OnePred/
├── onepred/            # Core Python package (agent loop, trainer, reward, judges)
│   ├── judges/         # LLM judge implementations (multi-model, listwise, embedding)
│   ├── interaction.py  # Agentic interaction (feeds turns, computes reward)
│   ├── locale.py       # Prompts, templates, and format strings
│   ├── recurrent_agent_loop.py      # Recursive memory agent loop
│   ├── recurrent_generation_manager.py  # Per-turn batch expansion
│   ├── recurrent_ray_trainer.py     # Multi-turn GRPO trainer
│   ├── recurrent_dp_actor.py        # DataParallel actor override
│   ├── recurrent_utils.py           # Shared utilities
│   ├── reward_fn.py    # Custom reward function for verl
│   └── main_ppo_recurrent.py  # Training entry point
├── tools/              # Data processing & offline evaluation scripts
├── scripts/            # Shell launchers (train/ and eval/)
├── configs/            # YAML configs (interaction, agent loop)
├── third_party/verl/   # Patched verl framework (see PATCH_NOTE.md)
├── .env.example        # Environment variable template
└── requirements.txt
```

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install verl (patched local version)
cd third_party/verl && pip install -e . && cd ../..

# 3. Configure environment variables
cp .env.example .env
# Edit .env: fill in WANDB_API_KEY, LLM judge URLs, model/data paths
```

## Data Preparation

Input: JSONL file where each line is a multi-turn dialogue session with ground-truth next queries.

```bash
# Build agentic RL training data (multi-turn interaction format)
python -m tools.build_verl_rl_agentic \
    --input data/raw/train.jsonl \
    --output_dir data/

# Build full-history baseline data
python -m tools.build_verl_rl_fullhistory \
    --input data/raw/train.jsonl \
    --output_dir data/

# Build current-turn baseline data
python -m tools.build_verl_rl_currentturn \
    --input data/raw/train.jsonl \
    --output_dir data/
```

## Training

```bash
# Stage 2: Agentic GRPO (main method — requires Stage 1 checkpoint)
bash scripts/train/run_grpo_agentic.sh

# Stage 1: Full-history GRPO (trains prediction ability)
bash scripts/train/run_grpo_fullhistory.sh

# Ablation: Current-turn baseline
bash scripts/train/run_grpo_currentturn.sh
```

Key environment variables (set in `.env` or export before running):
- `ONEPRED_ROOT`: project root directory
- `MODEL_PATH`: path to base model or Stage 1 checkpoint (Qwen3-8B)
- `MODEL_DIR`: directory containing model weights
- `WANDB_API_KEY`: Weights & Biases API key

## Evaluation

```bash
# Agentic evaluation (reuses training inference pipeline)
bash scripts/eval/run_eval_valonly.sh <checkpoint_path>

# Baseline evaluation (current-turn, sliding-window, full-history)
bash scripts/eval/run_all_baselines.sh

# Compute metrics from evaluation traces
python -m tools.compute_metrics --input_dir outputs/eval_run/
python -m tools.summarize_eval outputs/eval_run/
```

## License

MIT
