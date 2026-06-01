# DPO Training for DOOM MultiVec

This directory contains a complete implementation of Direct Preference Optimization (DPO) for training DOOM-playing policies from LLM rubric scores.

## Overview

DPO trains a policy to prefer high-scoring actions over low-scoring ones, as judged by the LLM rubric. Unlike traditional RL, DPO:
- Requires no reward model training
- Works offline (no environment interaction during training)
- Is stable and sample-efficient for small models

## Architecture

```
ASCII Frame → [Frozen Encoder] → [CLS] Embedding (128-d)
                              ↓
                    [Trainable Actor Head]
                    Linear(128, 64) → GELU → Linear(64, 6)
                              ↓
                       Action Logits
```

- **Encoder**: ModernBERT-Hash (5L, 128d) — frozen, preserves ASCII understanding
- **Actor Head**: Small MLP (~4K parameters) — the only trainable component
- **Reference Policy**: Frozen copy of initial policy for DPO ratio computation

## Complete Workflow

The full DPO training pipeline has **4 steps**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Generate    │ →  │  2. Score with  │ →  │  3. Prepare     │ →  │  4. Train DPO   │
│    Episodes     │    │   LLM Rubric    │    │ Preference Pairs│    │    Policy       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
episodes_generated.json episodes_rubric_scored.json dpo_preferences.json  trained model
```

### Step 1: Generate Episodes

Generate gameplay episodes using your current model:

```bash
python scripts/generate_dpo_data.py --model models/doom-multivec-5L --num-episodes 50 --output data/episodes_generated.json --armed
```

**Output**: `data/episodes_generated.json` with frames, actions, and game states.

### Step 2: Score with LLM Rubric

Send episode frames to an LLM for scoring using your rubric:

```bash
python scripts/score_with_rubric.py --episodes data/episodes_generated.json --output data/episodes_rubric_scored.json --api-key $TRITON_API_KEY --sample-frames 5
```

**Output**: `data/episodes_rubric_scored.json` with added `rubric_score` field (0.0 to 1.0).

### Step 3: Prepare Preference Pairs

Convert scored episodes into win/lose preference pairs for DPO:

```bash
python scripts/prepare_dpo_data.py --episodes data/episodes_rubric_scored.json --output data/dpo_preferences.json --min-margin 0.3
```

**Output**: `data/dpo_preferences.json` with preference pairs (win_action vs lose_action).

### Step 4: Train with DPO

Train the policy using the preference pairs:

```bash
python scripts/train_dpo.py --model models/doom-multivec-5L --data data/dpo_preferences.json --output output/dpo-v1 --epochs 5 --batch-size 32 --lr 5e-5 --beta 0.1 --bf16
```

**Output**: Trained model in `output/dpo-v1/final/`

## Quick Reference: All Commands

```bash
# Complete pipeline
python scripts/generate_dpo_data.py --model models/doom-multivec-5L --num-episodes 50 --output data/episodes_generated.json --armed

export TRITON_API_KEY="your-api-key"

python scripts/score_with_rubric.py --episodes data/episodes_generated.json --output data/episodes_rubric_scored.json

python scripts/prepare_dpo_data.py --episodes data/episodes_rubric_scored.json --output data/dpo_preferences.json --min-margin 0.3

python scripts/train_dpo.py --model models/doom-multivec-5L --data data/dpo_preferences.json --output output/dpo-v1 --epochs 5 --batch-size 32 --lr 5e-5 --beta 0.1
```

## Important: Data File Formats

⚠️ **Each script expects a specific input format. You cannot skip steps.**

| File | Stage | Contains |
|------|-------|----------|
| `episodes_generated.json` | Step 1 output | Raw frames, actions, game states |
| `episodes_rubric_scored.json` | Step 2 output | Above + `rubric_score` field |
| `dpo_preferences.json` | Step 3 output | Preference pairs with win/lose actions |

**Error you'll get if you skip steps:**
```
ValueError: min() arg is an empty sequence  # No preference pairs found
```

## Using the Trained Model

```python
from doom_multivec.model import DPODoomPolicy
import torch

# Load trained policy
policy = DPODoomPolicy.from_pretrained('output/dpo-v1/final')
policy.eval()

# Inference on a frame
with torch.no_grad():
    action, probs = policy.sample_action(
        input_ids, attention_mask, temperature=1.0
    )

# Or get probability distribution
action_probs = policy.predict_action_probs(input_ids, attention_mask)
# {'shoot': 0.45, 'move_forward': 0.30, ...}
```

## Training Arguments for `train_dpo.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `models/doom-multivec-5L` | Base encoder model path |
| `--data` | (required) | Path to **preference pairs** JSON (not raw episodes!) |
| `--output` | `output/dpo-trained` | Output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 5e-5 | Learning rate |
| `--beta` | 0.1 | DPO temperature. Lower = more divergence from reference |
| `--actor-hidden` | 64 | Actor head hidden dimension |
| `--eval-split` | 0.1 | Fraction of data for validation |
| `--bf16` | False | Use bfloat16 mixed precision |
| `--resume` | None | Resume from checkpoint directory |

## Understanding the Beta Parameter

Beta controls how much the policy can diverge from the reference:

- **β = 0.1** (conservative): Policy stays close to reference, small improvements
- **β = 0.5** (moderate): Balanced exploration/exploitation
- **β = 1.0** (aggressive): Large updates, risk of instability

Start with 0.1 and increase if training is stable.

## Data Format Details

### Step 1 Output: Raw Episodes (`episodes_generated.json`)

```json
{
  "episodes": [
    {
      "frames": [
        {
          "text": "ASCII frame content...",
          "input_ids": [2, 45, 67, ...],
          "attention_mask": [1, 1, 1, ...],
          "action_taken": "shoot",
          "action_idx": 0,
          "reward": 1.0
        }
      ],
      "actions": [0, 1, 2, ...],
      "rewards": [1.0, 0.0, ...],
      "total_reward": 5.0,
      "kills": 5,
      "num_frames": 100
    }
  ]
}
```

### Step 2 Output: Rubric-Scored Episodes (`episodes_rubric_scored.json`)

```json
{
  "episodes": [
    {
      "frames": [...],
      "actions": [...],
      "rubric_score": 0.85,           // ← Added by scoring
      "rubric_reasoning": "Good threat management...",
      "total_reward": 5.0
    }
  ]
}
```

### Step 3 Output: Preference Pairs (`dpo_preferences.json`)

```json
{
  "pairs": [
    {
      "input_ids": [2, 45, 67, ...],     // State (frame)
      "attention_mask": [1, 1, 1, ...],
      "win_action": 0,                   // Preferred action
      "lose_action": 3,                  // Dispreferred action
      "win_score": 0.85,                 // Rubric score for winner
      "lose_score": 0.45                 // Rubric score for loser
    }
  ]
}
```

## Output Structure

```
output/dpo-v1/
├── checkpoint-epoch-1/          # Epoch checkpoints
│   ├── model.safetensors        # Full model
│   ├── actor_head.pt            # Actor head only
│   ├── config.json              # Model config
│   ├── optimizer.pt             # Optimizer state
│   └── training_state.json      # Step, history, etc.
├── best/                        # Best model by eval accuracy
│   └── ...
├── final/                       # Final model after training
│   ├── model.safetensors
│   ├── actor_head.pt
│   ├── config.json              # Includes dpo_config
│   └── metadata.json            # Training metadata
├── train_config.json            # Full training args
└── history.json                 # Loss curves
```

## Monitoring Training

Enable Weights & Biases logging:
```bash
python scripts/train_dpo.py ... --wandb --wandb-project doom-dpo
```

Key metrics to watch:
- `train_loss`: Should decrease steadily
- `train_accuracy`: % of pairs where winner is ranked higher (target > 70%)
- `eval_accuracy`: Validation accuracy (watch for overfitting)
- `kl_div`: KL divergence from reference (keep < 10 for stability)

## Resuming Training

```bash
python scripts/train_dpo.py --resume output/dpo-v1/checkpoint-epoch-3 --data data/dpo_preferences.json --output output/dpo-v1-resumed --epochs 5
```

## Tips for Good Results

1. **Quality over quantity**: Better to have 500 high-quality preference pairs than 5000 noisy ones
2. **Score margins matter**: Use `--min-margin 0.3` to ensure clear preferences
3. **Balance actions**: Ensure all 6 actions appear in winning positions
4. **Start conservative**: Use β=0.1, increase if stable
5. **Monitor KL**: If KL diverges (>10), reduce learning rate or increase β

## Troubleshooting

### "ValueError: min() arg is an empty sequence"

**Cause**: You passed raw episodes to `train_dpo.py` instead of preference pairs.

**Fix**: Run all 4 steps in order. The `--data` argument to `train_dpo.py` must be the output of `prepare_dpo_data.py` (with preference pairs), not `generate_dpo_data.py` output.

### "Error initializing VizDoom"

**Cause**: VizDoom not installed or scenario file missing.

**Fix**: 
```bash
pip install vizdoom
# Ensure DOOM scenario files are in the correct location
```

### Loss not decreasing

- Check learning rate (try lower, e.g., 1e-5)
- Verify preference pairs have clear margins
- Ensure encoder is frozen

### Policy collapses to single action

- Increase β (temperature)
- Add more diverse preference pairs
- Check action distribution in training data

### Out of memory

- Reduce batch size
- Enable bf16 with `--bf16`
- Use gradient checkpointing (modify code)

## References

- Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model". NeurIPS 2023.
