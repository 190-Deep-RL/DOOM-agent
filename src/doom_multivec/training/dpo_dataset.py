"""Dataset utilities for DPO training on rubric-graded DOOM trajectories.

Provides preference pair construction from episodes scored by the LLM rubric.
Pairs are created by comparing actions taken in similar states with different
rubric scores.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class DPOPreferenceSample:
    """Single preference sample for DPO training.

    Attributes:
        input_ids: Tokenized ASCII frame (list of int).
        attention_mask: Attention mask (list of int).
        win_action: Index of preferred action (int).
        lose_action: Index of dispreferred action (int).
        win_score: Rubric score for winning trajectory (float).
        lose_score: Rubric score for losing trajectory (float).
        depth_ids: Optional depth bin IDs (list of int or None).
    """

    def __init__(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        win_action: int,
        lose_action: int,
        win_score: float,
        lose_score: float,
        depth_ids: list[int] | None = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.win_action = win_action
        self.lose_action = lose_action
        self.win_score = win_score
        self.lose_score = lose_score
        self.depth_ids = depth_ids

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'win_action': self.win_action,
            'lose_action': self.lose_action,
            'win_score': self.win_score,
            'lose_score': self.lose_score,
            'depth_ids': self.depth_ids,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'DPOPreferenceSample':
        """Create from dictionary."""
        return cls(
            input_ids=d['input_ids'],
            attention_mask=d['attention_mask'],
            win_action=d['win_action'],
            lose_action=d['lose_action'],
            win_score=d['win_score'],
            lose_score=d['lose_score'],
            depth_ids=d.get('depth_ids'),
        )


class DPOPreferenceDataset(Dataset):
    """PyTorch Dataset for DPO preference pairs.

    Args:
        samples: List of DPOPreferenceSample objects.
    """

    def __init__(self, samples: list[DPOPreferenceSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        result = {
            'input_ids': torch.tensor(sample.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(sample.attention_mask, dtype=torch.long),
            'win_action': torch.tensor(sample.win_action, dtype=torch.long),
            'lose_action': torch.tensor(sample.lose_action, dtype=torch.long),
        }

        if sample.depth_ids is not None and isinstance(sample.depth_ids, list) and len(sample.depth_ids) > 0:
            try:
                result['depth_ids'] = torch.tensor(sample.depth_ids, dtype=torch.long)
            except (TypeError, ValueError):
                # Skip if depth_ids can't be converted to tensor
                pass

        return result

    def save(self, path: str | Path) -> None:
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'samples': [s.to_dict() for s in self.samples],
            'num_samples': len(self.samples),
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> 'DPOPreferenceDataset':
        """Load dataset from disk."""
        with open(path) as f:
            data = json.load(f)

        samples = [DPOPreferenceSample.from_dict(s) for s in data['samples']]
        return cls(samples)


class DPODataBuilder:
    """Builds preference datasets from rubric-graded trajectories.

    Takes trajectory data with per-episode rubric scores and constructs
    preference pairs suitable for DPO training.

    Args:
        min_score_margin: Minimum score difference to create a pair (default 0.3).
        max_pairs_per_episode: Max pairs to extract from single episode (default 100).
        balance_by_score: If True, ensure good mix of score ranges (default True).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        min_score_margin: float = 0.3,
        max_pairs_per_episode: int = 100,
        balance_by_score: bool = True,
        seed: int = 42,
    ):
        self.min_score_margin = min_score_margin
        self.max_pairs_per_episode = max_pairs_per_episode
        self.balance_by_score = balance_by_score
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def build_from_trajectories(
        self,
        trajectories: list[dict[str, Any]],
        tokenizer=None,
        max_length: int = 1100,
    ) -> list[DPOPreferenceSample]:
        """Build preference pairs from trajectories.

        Args:
            trajectories: List of trajectory dicts with keys:
                - 'frames': list of frame data (each with 'text', optional 'depth_bins')
                - 'actions': list of action indices taken
                - 'rubric_score': float score from LLM rubric
            tokenizer: Optional tokenizer for on-the-fly tokenization.
                      If None, assumes frames are already tokenized.
            max_length: Max sequence length for tokenization.

        Returns:
            List of DPOPreferenceSample objects.
        """
        samples = []

        # Sort trajectories by score
        scored_trajs = [(t, t.get('rubric_score', 0.5)) for t in trajectories]
        scored_trajs.sort(key=lambda x: x[1])

        # Split into tiers for pairing
        n = len(scored_trajs)
        low_tier = scored_trajs[:n//3]
        mid_tier = scored_trajs[n//3:2*n//3]
        high_tier = scored_trajs[2*n//3:]

        print(f"Trajectory tiers: {len(low_tier)} low, {len(mid_tier)} mid, {len(high_tier)} high")

        # Create pairs: high vs low, high vs mid, mid vs low
        pairs_to_build = [
            (high_tier, low_tier, "high_vs_low"),
            (high_tier, mid_tier, "high_vs_mid"),
            (mid_tier, low_tier, "mid_vs_low"),
        ]

        for win_tier, lose_tier, pair_type in pairs_to_build:
            tier_samples = self._build_pairs_between_tiers(
                win_tier, lose_tier, tokenizer, max_length, pair_type
            )
            samples.extend(tier_samples)
            print(f"  {pair_type}: {len(tier_samples)} pairs")

        # Shuffle
        self.rng.shuffle(samples)

        return samples

    def _build_pairs_between_tiers(
        self,
        win_tier: list[tuple[dict, float]],
        lose_tier: list[tuple[dict, float]],
        tokenizer,
        max_length: int,
        pair_type: str,
    ) -> list[DPOPreferenceSample]:
        """Build preference pairs between two score tiers."""
        samples = []

        # Limit samples per tier combination
        max_per_combo = self.max_pairs_per_episode * 3

        for win_traj, win_score in win_tier:
            for lose_traj, lose_score in lose_tier:
                score_margin = win_score - lose_score

                if score_margin < self.min_score_margin:
                    continue

                # Find matching frames (similar game state)
                pairs = self._extract_frame_pairs(
                    win_traj, lose_traj, win_score, lose_score, tokenizer, max_length
                )
                samples.extend(pairs)

                if len(samples) >= max_per_combo:
                    break

            if len(samples) >= max_per_combo:
                break

        return samples

    def _extract_frame_pairs(
        self,
        win_traj: dict,
        lose_traj: dict,
        win_score: float,
        lose_score: float,
        tokenizer,
        max_length: int,
    ) -> list[DPOPreferenceSample]:
        """Extract frame-level preference pairs from two trajectories."""
        samples = []

        win_frames = win_traj.get('frames', [])
        lose_frames = lose_traj.get('frames', [])
        win_actions = win_traj.get('actions', [])
        lose_actions = lose_traj.get('actions', [])

        if not win_frames or not lose_frames:
            return samples

        # Sample random frame pairs (could be smarter about matching game state)
        num_pairs = min(10, len(win_frames), len(lose_frames))
        win_indices = self.np_rng.choice(len(win_frames), size=num_pairs, replace=False)
        lose_indices = self.np_rng.choice(len(lose_frames), size=num_pairs, replace=False)

        for wi, li in zip(win_indices, lose_indices):
            win_frame = win_frames[wi]
            lose_frame = lose_frames[li]

            # Use winning frame as the state (could use either)
            state = win_frame

            # Tokenize if needed
            if tokenizer is not None:
                encoded = tokenizer(
                    state['text'],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                )
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                depth_ids = self._encode_depth(state.get('depth_bins'), len(input_ids))
            else:
                input_ids = state.get('input_ids', [])
                attention_mask = state.get('attention_mask', [])
                depth_ids = state.get('depth_ids')

            if not input_ids:
                continue

            # Get actions
            win_action = win_actions[wi] if wi < len(win_actions) else 0
            lose_action = lose_actions[li] if li < len(lose_actions) else 0

            # Skip if same action (no preference signal)
            if win_action == lose_action:
                continue

            sample = DPOPreferenceSample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                win_action=int(win_action),
                lose_action=int(lose_action),
                win_score=win_score,
                lose_score=lose_score,
                depth_ids=depth_ids,
            )
            samples.append(sample)

        return samples

    def _encode_depth(self, depth_bins: list | None, seq_len: int) -> list[int] | None:
        """Encode depth bins to depth_ids format."""
        if depth_bins is None:
            return None

        # Pad or truncate to seq_len
        if len(depth_bins) < seq_len:
            depth_bins = depth_bins + [0] * (seq_len - len(depth_bins))
        else:
            depth_bins = depth_bins[:seq_len]

        return depth_bins

    def build_from_preference_file(
        self,
        preference_file: str | Path,
    ) -> list[DPOPreferenceSample]:
        """Load pre-built preference pairs from a JSON file.

        Expected format:
        {
            "pairs": [
                {
                    "input_ids": [...],
                    "attention_mask": [...],
                    "win_action": 0,
                    "lose_action": 3,
                    "win_score": 0.85,
                    "lose_score": 0.45,
                    "depth_ids": [...]  // optional
                },
                ...
            ]
        }
        """
        with open(preference_file) as f:
            data = json.load(f)

        samples = [DPOPreferenceSample.from_dict(p) for p in data.get('pairs', [])]
        return samples


def create_preference_pairs_from_rubric_scores(
    episodes: list[dict],
    score_key: str = 'rubric_score',
    min_margin: float = 0.3,
) -> list[dict]:
    """Simple utility to create preference pairs from scored episodes.

    Args:
        episodes: List of episodes, each with 'frames' and score_key.
        score_key: Key for rubric score in episode dict.
        min_margin: Minimum score difference for a valid pair.

    Returns:
        List of preference pair dicts.
    """
    pairs = []

    for i, ep1 in enumerate(episodes):
        for ep2 in episodes[i+1:]:
            score1 = ep1.get(score_key, 0.5)
            score2 = ep2.get(score_key, 0.5)

            margin = abs(score1 - score2)
            if margin < min_margin:
                continue

            # Determine winner
            if score1 > score2:
                win_ep, lose_ep = ep1, ep2
                win_score, lose_score = score1, score2
            else:
                win_ep, lose_ep = ep2, ep1
                win_score, lose_score = score2, score1

            # Create frame-level pairs (simplified: just first frame)
            # In practice, you'd want to match similar game states
            if win_ep.get('frames') and lose_ep.get('frames'):
                pair = {
                    'win_frame': win_ep['frames'][0],
                    'lose_frame': lose_ep['frames'][0],
                    'win_action': win_ep.get('actions', [0])[0] if win_ep.get('actions') else 0,
                    'lose_action': lose_ep.get('actions', [0])[0] if lose_ep.get('actions') else 0,
                    'win_score': win_score,
                    'lose_score': lose_score,
                    'margin': margin,
                }
                pairs.append(pair)

    return pairs
