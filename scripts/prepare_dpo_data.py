"""Prepare DPO training data from rubric-graded DOOM episodes.

This script takes episodes graded by the LLM rubric and constructs
preference pairs for DPO training.

Usage:
    # From rubric-graded trajectories
    python scripts/prepare_dpo_data.py \
        --episodes data/rubric_scored_episodes.json \
        --output data/dpo_preferences.json \
        --min-margin 0.3

    # From trajectory directory (multiple files)
    python scripts/prepare_dpo_data.py \
        --episodes-dir data/rubric_episodes/ \
        --output data/dpo_preferences.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doom_multivec.training.dpo_dataset import (
    DPODataBuilder,
    DPOPreferenceDataset,
)
from transformers import AutoTokenizer


def load_episodes(path: str) -> list[dict]:
    """Load episodes from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
    if 'episodes' in data:
        return data['episodes']
    elif 'trajectories' in data:
        return data['trajectories']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DPO training data from rubric-graded episodes",
    )

    # Input arguments
    parser.add_argument(
        '--episodes',
        type=str,
        help='Path to rubric-graded episodes JSON file',
    )
    parser.add_argument(
        '--episodes-dir',
        type=str,
        help='Directory containing multiple episode JSON files',
    )

    # Processing arguments
    parser.add_argument(
        '--tokenizer',
        default='models/doom-multivec-5L',
        help='Tokenizer to use for encoding frames',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1100,
        help='Max sequence length',
    )
    parser.add_argument(
        '--min-margin',
        type=float,
        default=0.3,
        help='Minimum score margin for preference pairs',
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=10000,
        help='Maximum number of pairs to generate',
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for DPO preference dataset',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec - Prepare DPO Data")
    print("=" * 60)

    # Load episodes
    if args.episodes:
        print(f"\nLoading episodes from {args.episodes}...")
        episodes = load_episodes(args.episodes)
    elif args.episodes_dir:
        print(f"\nLoading episodes from {args.episodes_dir}...")
        episodes = []
        episodes_dir = Path(args.episodes_dir)
        for json_file in episodes_dir.glob('*.json'):
            eps = load_episodes(str(json_file))
            episodes.extend(eps)
            print(f"  Loaded {len(eps)} episodes from {json_file.name}")
    else:
        parser.error("Either --episodes or --episodes-dir must be specified")

    print(f"\nTotal episodes: {len(episodes)}")

    # Check for rubric scores
    scored = [ep for ep in episodes if 'rubric_score' in ep]
    unscored = [ep for ep in episodes if 'rubric_score' not in ep]
    print(f"  With rubric scores: {len(scored)}")
    print(f"  Without rubric scores: {len(unscored)}")

    if len(scored) == 0:
        print("\nError: No episodes have rubric scores!")
        print("Expected 'rubric_score' field in episode dictionaries.")
        return 1

    # Score distribution
    scores = [ep['rubric_score'] for ep in scored]
    print(f"\nScore distribution:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")
    print(f"  Median: {sorted(scores)[len(scores)//2]:.3f}")

    # Bin scores
    low = [s for s in scores if s < 0.33]
    mid = [s for s in scores if 0.33 <= s < 0.67]
    high = [s for s in scores if s >= 0.67]
    print(f"\nScore bins:")
    print(f"  Low (< 0.33):    {len(low)} ({len(low)/len(scores)*100:.1f}%)")
    print(f"  Mid (0.33-0.67): {len(mid)} ({len(mid)/len(scores)*100:.1f}%)")
    print(f"  High (> 0.67):   {len(high)} ({len(high)/len(scores)*100:.1f}%)")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Build preference pairs
    print(f"\nBuilding preference pairs (min_margin={args.min_margin})...")

    builder = DPODataBuilder(
        min_score_margin=args.min_margin,
        max_pairs_per_episode=100,
        balance_by_score=True,
        seed=42,
    )

    samples = builder.build_from_trajectories(
        scored, tokenizer=tokenizer, max_length=args.max_length
    )

    # Limit pairs
    if len(samples) > args.max_pairs:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_pairs)
        print(f"  Sampled down to {len(samples)} pairs")

    print(f"\nGenerated {len(samples)} preference pairs")

    # Analyze pairs
    margins = [s.win_score - s.lose_score for s in samples]
    print(f"\nPair score margins:")
    print(f"  Min: {min(margins):.3f}")
    print(f"  Max: {max(margins):.3f}")
    print(f"  Mean: {sum(margins)/len(margins):.3f}")

    # Action distribution
    win_actions = {}
    lose_actions = {}
    for s in samples:
        win_actions[s.win_action] = win_actions.get(s.win_action, 0) + 1
        lose_actions[s.lose_action] = lose_actions.get(s.lose_action, 0) + 1

    print(f"\nWinning actions:")
    for action_id, count in sorted(win_actions.items()):
        print(f"  Action {action_id}: {count} ({count/len(samples)*100:.1f}%)")

    print(f"\nLosing actions:")
    for action_id, count in sorted(lose_actions.items()):
        print(f"  Action {action_id}: {count} ({count/len(samples)*100:.1f}%)")

    # Save dataset
    print(f"\nSaving to {args.output}...")
    dataset = DPOPreferenceDataset(samples)
    dataset.save(args.output)

    # Also save as simple JSON for inspection
    simple_path = str(args.output).replace('.json', '_simple.json')
    with open(simple_path, 'w') as f:
        json.dump({
            'num_pairs': len(samples),
            'min_margin': args.min_margin,
            'pairs': [
                {
                    'win_action': s.win_action,
                    'lose_action': s.lose_action,
                    'win_score': s.win_score,
                    'lose_score': s.lose_score,
                    'margin': s.win_score - s.lose_score,
                }
                for s in samples[:100]  # First 100 for inspection
            ]
        }, f, indent=2)

    print(f"  Saved {len(samples)} preference pairs")
    print(f"  Summary saved to {simple_path}")

    print("\n" + "=" * 60)
    print("Done! Use this file with train_dpo.py:")
    print(f"  python scripts/train_dpo.py --data {args.output} --output output/dpo-v1")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
