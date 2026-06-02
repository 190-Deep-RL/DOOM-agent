"""Train DOOM MultiVec policy using Direct Preference Optimization (DPO).

This script trains a policy from LLM-rubric-scored preference pairs.
The encoder is frozen; only a small actor head is trained.

Usage:
    # Basic training
    python scripts/train_dpo.py --data data/dpo_preferences.json --output output/dpo-v1

    # Full training with options
    python scripts/train_dpo.py \
        --model models/doom-multivec-5L \
        --data data/dpo_preferences.json \
        --output output/dpo-v1 \
        --epochs 5 \
        --batch-size 32 \
        --lr 5e-5 \
        --beta 0.1 \
        --eval-split 0.1 \
        --bf16

    # Resume from checkpoint
    python scripts/train_dpo.py \
        --resume output/dpo-v1/checkpoint-epoch-3 \
        --data data/dpo_preferences.json \
        --output output/dpo-v1-resumed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Ensure package is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doom_multivec.model.dpo_policy import DPODoomPolicy
from doom_multivec.training.dpo_dataset import (
    DPOPreferenceDataset,
    DPOPreferenceSample,
)
from doom_multivec.training.dpo_trainer import DPOTrainer, create_reference_policy
from doom_multivec.training.action_mapping import BASE_ACTIONS


def load_preference_data(
    data_path: str,
    tokenizer=None,
    max_length: int = 1100,
) -> list[DPOPreferenceSample]:
    """Load preference data from JSON file.

    Supports two formats:
    1. Simple format: {"pairs": [...]} with pre-tokenized data
    2. Trajectory format: {"episodes": [...]} with raw text frames
    """
    with open(data_path) as f:
        data = json.load(f)

    samples = []

    # Check format
    if 'pairs' in data:
        # Pre-built preference pairs
        for pair in data['pairs']:
            sample = DPOPreferenceSample(
                input_ids=pair['input_ids'],
                attention_mask=pair['attention_mask'],
                win_action=pair['win_action'],
                lose_action=pair['lose_action'],
                win_score=pair.get('win_score', 1.0),
                lose_score=pair.get('lose_score', 0.0),
                depth_ids=pair.get('depth_ids'),
            )
            samples.append(sample)

    elif 'samples' in data:
        # Prepared dataset from DPOPreferenceDataset.save()
        from doom_multivec.training.dpo_dataset import DPOPreferenceSample

        for s in data['samples']:
            sample = DPOPreferenceSample(
                input_ids=s['input_ids'],
                attention_mask=s['attention_mask'],
                win_action=s['win_action'],
                lose_action=s['lose_action'],
                win_score=s.get('win_score', 1.0),
                lose_score=s.get('lose_score', 0.0),
                depth_ids=s.get('depth_ids'),
            )
            samples.append(sample)

    elif 'episodes' in data or 'trajectories' in data:
        # Raw trajectories with rubric scores
        # Need to build pairs
        from doom_multivec.training.dpo_dataset import DPODataBuilder

        trajectories = data.get('episodes', data.get('trajectories', []))

        builder = DPODataBuilder(
            min_score_margin=0.3,
            max_pairs_per_episode=100,
        )
        samples = builder.build_from_trajectories(
            trajectories, tokenizer=tokenizer, max_length=max_length
        )

    else:
        raise ValueError(
            f"Unknown data format in {data_path}. "
            "Expected 'pairs', 'samples', or 'episodes'/'trajectories' key."
        )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train DOOM MultiVec policy with DPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        '--model', '-m',
        default='models/doom-multivec-5L',
        help='Path to base encoder model',
    )
    parser.add_argument(
        '--actor-hidden',
        type=int,
        default=64,
        help='Hidden size of actor head MLP',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint directory',
    )

    # Data arguments
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to preference data JSON file',
    )
    parser.add_argument(
        '--eval-split',
        type=float,
        default=0.1,
        help='Fraction of data for validation (0-1)',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1100,
        help='Max sequence length',
    )

    # Training arguments
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=3,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='DPO temperature (lower = more divergence from reference)',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay for AdamW',
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Use bfloat16 mixed precision',
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        default='output/dpo-trained',
        help='Output directory for trained model',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=1,
        help='Save checkpoint every N epochs',
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=1,
        help='Evaluate every N epochs',
    )

    # Logging
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project',
        default='doom-multivec-dpo',
        help='W&B project name',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec - DPO Training")
    print("=" * 60)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer if needed
    tokenizer = None
    if args.resume is None:
        from transformers import AutoTokenizer
        print(f"\nLoading tokenizer from {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load preference data
    print(f"\nLoading preference data from {args.data}...")
    samples = load_preference_data(args.data, tokenizer, args.max_length)
    print(f"  Loaded {len(samples)} preference pairs")

    # Print sample statistics
    win_scores = [s.win_score for s in samples]
    lose_scores = [s.lose_score for s in samples]
    margins = [s.win_score - s.lose_score for s in samples]
    print(f"  Win score:  {min(win_scores):.2f} - {max(win_scores):.2f} (avg {sum(win_scores)/len(win_scores):.2f})")
    print(f"  Lose score: {min(lose_scores):.2f} - {max(lose_scores):.2f} (avg {sum(lose_scores)/len(lose_scores):.2f})")
    print(f"  Margin:     {min(margins):.2f} - {max(margins):.2f} (avg {sum(margins)/len(margins):.2f})")

    # Create dataset
    dataset = DPOPreferenceDataset(samples)

    # Train/test split
    if args.eval_split > 0:
        eval_size = int(len(dataset) * args.eval_split)
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"\nDataset split: {train_size} train, {eval_size} eval")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"\nDataset: {len(dataset)} samples (no eval split)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid issues with Windows/Mac
        pin_memory=(device == 'cuda'),
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda'),
        )

    # Create or load policy
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        policy = DPODoomPolicy.from_pretrained(args.resume)
        policy.to(device)
        print(f"  Loaded policy with {policy.count_parameters()['total']:,} params")
    else:
        print(f"\nInitializing policy from {args.model}...")
        policy = DPODoomPolicy(
            encoder_path=args.model,
            actor_hidden=args.actor_hidden,
            num_actions=len(BASE_ACTIONS),
            freeze_encoder=True,
        )
        policy.to(device)

        param_counts = policy.count_parameters()
        print(f"  Encoder: {param_counts['encoder']:,} params (frozen)")
        print(f"  Actor head: {param_counts['actor_head']:,} params (trainable)")
        print(f"  Total: {param_counts['total']:,} params")

    # Create reference policy
    print("\nCreating reference policy...")
    ref_policy = create_reference_policy(policy)

    # Create trainer
    print(f"\nInitializing trainer:")
    print(f"  Beta: {args.beta}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    trainer = DPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        beta=args.beta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    # Resume training state if applicable
    if args.resume:
        state_path = os.path.join(args.resume, 'training_state.json')
        if os.path.exists(state_path):
            trainer.load_checkpoint(args.resume)
            print(f"  Resumed from step {trainer.global_step}")

    # Wandb logging
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    'model': args.model,
                    'beta': args.beta,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'actor_hidden': args.actor_hidden,
                    'train_samples': len(train_dataset),
                    'eval_samples': len(eval_dataset) if eval_dataset else 0,
                }
            )
            print(f"\nW&B logging enabled: {args.wandb_project}")
        except ImportError:
            print("\nWarning: wandb not installed, disabling W&B logging")
            args.wandb = False

    # Train
    os.makedirs(args.output, exist_ok=True)

    # Save training config
    with open(os.path.join(args.output, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nOutput directory: {args.output}")
    print(f"Starting training for {args.epochs} epochs...\n")

    history = trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        num_epochs=args.epochs,
        eval_every=args.eval_every,
        save_path=args.output,
        save_every=args.save_every,
    )

    # Save final model
    final_path = os.path.join(args.output, 'final')
    os.makedirs(final_path, exist_ok=True)
    policy.save_pretrained(final_path, save_encoder=True)

    # Save training history
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save metadata
    metadata = {
        'base_model': args.model,
        'actor_hidden': args.actor_hidden,
        'num_actions': len(BASE_ACTIONS),
        'actions': BASE_ACTIONS,
        'beta': args.beta,
        'final_step': trainer.global_step,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset) if eval_dataset else 0,
    }
    with open(os.path.join(final_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'=' * 60}")

    # Test inference
    print("\nTesting inference...")
    policy.eval()
    with torch.no_grad():
        sample = train_dataset[0]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

        logits = policy(input_ids, attention_mask)['logits']
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        print("Action probabilities on sample:")
        for i, (action, prob) in enumerate(zip(BASE_ACTIONS, probs)):
            print(f"  {action:15s}: {prob:.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
