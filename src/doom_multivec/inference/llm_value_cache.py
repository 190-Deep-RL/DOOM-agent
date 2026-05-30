"""LLM-based value cache shared by BoN and beam search planners.

The cache amortizes expensive LLM evaluations across many rollouts:

1. Planners record `(state_hash, action_idx)` for every visited step.
2. With probability `sample_rate` (or on explicit `rate_trajectory` calls)
   the cache fires one LLM rating on a trajectory's frames and broadcasts
   the resulting `[0, 1]` score back onto every visited (state, action)
   pair with a recency-decay (Monte-Carlo credit assignment).
3. Future rollouts that revisit cached pairs look up the EMA-blended
   value cheaply (no API call).

This converts a per-rollout LLM expense into a per-region-of-state-space
expense; the cache densifies over the episode and across episodes if
persisted to disk.

Design notes:
- State key = stable md5 of the ASCII frame, so saved caches survive
  process restarts (Python's built-in str hash is salted per-process).
- Values are kept in `[0, 1]` via an exponential moving average.
- Rating is fail-soft: on any LLM error the rating returns the prior
  default (0.5) and the trajectory is not credited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from doom_multivec.inference.llm_eval import (
    _extract_text_from_response,
    query_llm_with_frames,
)


def ascii_state_hash(ascii_frame: str) -> str:
    """Stable 16-char hash of an ASCII frame (md5 prefix).

    Deterministic across processes, unlike Python's builtin str hash.
    """
    return hashlib.md5(ascii_frame.encode('utf-8')).hexdigest()[:16]


def parse_llm_score(text: str) -> Optional[float]:
    """Pull a [0, 1] float from an LLM text response.

    Mirrors the regex MCTSAgent._evaluate_with_llm uses, then falls back
    to a positive-word heuristic. Returns None if nothing parseable.
    """
    if not isinstance(text, str):
        return None
    numbers = re.findall(r'0\.\d+|1\.0', text)
    if numbers:
        try:
            return max(0.0, min(1.0, float(numbers[0])))
        except ValueError:
            pass
    positive_words = ['good', 'excellent', 'well', 'strong', 'positive']
    if any(word in text.lower() for word in positive_words):
        return 0.7
    return None


@dataclass
class CacheEntry:
    value: float = 0.5
    visits: int = 0


class LLMValueCache:
    """Sparse `(state_hash, action_idx) -> EMA float` table.

    Args:
        api_key: Triton API key. If None, the cache silently disables
            actual LLM calls (lookups still work; rate_trajectory becomes
            a no-op returning 0.5). Useful for dry-run smoke tests.
        prompt: Rubric text sent with each evaluation.
        sample_rate: Probability that `maybe_rate_trajectory` actually
            fires an LLM call. Each call labels ~`trajectory_length`
            entries, so effective LLM cost per rollout is
            sample_rate * 1, not sample_rate * trajectory_length.
        ema_alpha: Step size for the EMA update (closer to 1.0 = more
            weight on the newest rating).
        credit_decay: Per-step multiplier when broadcasting the LLM
            rating backwards along a trajectory. `1.0` = uniform credit;
            `< 1.0` = recent steps closer to the rated endpoint get more
            weight. Mirrors classic n-step return discounting.
        default_value: What `lookup` returns on a miss when callers ask
            for a fallback rather than `None`.
        llm_model: Model name passed to the Triton chat endpoint.
        llm_max_tokens: Cap on response length.
        verbose: Print parsed LLM responses for debugging.
    """

    def __init__(
        self,
        api_key: Optional[str],
        prompt: str,
        sample_rate: float = 0.05,
        ema_alpha: float = 0.3,
        credit_decay: float = 0.95,
        default_value: float = 0.5,
        llm_model: str = "api-gemma-4-26b",
        llm_max_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.api_key = api_key
        self.prompt = prompt
        self.sample_rate = sample_rate
        self.ema_alpha = ema_alpha
        self.credit_decay = credit_decay
        self.default_value = default_value
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.verbose = verbose

        self._table: Dict[Tuple[str, int], CacheEntry] = {}
        self._lock = threading.Lock()
        # Stats exposed to drivers.
        self.stats = {
            'llm_calls': 0,
            'llm_failures': 0,
            'lookups': 0,
            'hits': 0,
            'updates': 0,
        }

    # ---------- public read path ----------

    def lookup(self, state_hash: str, action_idx: int) -> Optional[float]:
        """Return the EMA-blended value for a (state, action) pair, or None."""
        self.stats['lookups'] += 1
        entry = self._table.get((state_hash, action_idx))
        if entry is None:
            return None
        self.stats['hits'] += 1
        return entry.value

    def lookup_or_default(self, state_hash: str, action_idx: int) -> float:
        v = self.lookup(state_hash, action_idx)
        return self.default_value if v is None else v

    def trajectory_value(
        self,
        trajectory: List[Tuple[str, int]],
        require_min_hits: int = 1,
    ) -> Optional[float]:
        """Mean cached value over a (state, action) trajectory.

        Returns None if fewer than `require_min_hits` of the steps are
        actually cached (so callers can decide to skip blending rather
        than fall back to the prior).
        """
        vals = []
        for sh, a in trajectory:
            v = self.lookup(sh, a)
            if v is not None:
                vals.append(v)
        if len(vals) < require_min_hits:
            return None
        return float(np.mean(vals))

    # ---------- public write path ----------

    def should_rate(self) -> bool:
        """Coin-flip whether to spend an LLM call this rollout."""
        if self.api_key is None or self.sample_rate <= 0.0:
            return False
        return random.random() < self.sample_rate

    def maybe_rate_trajectory(
        self,
        frames: List[np.ndarray],
        trajectory: List[Tuple[str, int]],
    ) -> Optional[float]:
        """LLM-rate `frames` and broadcast the score onto `trajectory`.

        Returns the rating, or None if rating was skipped or failed.
        Only fires the API call when `should_rate()` returns True.
        """
        if not self.should_rate():
            return None
        return self.rate_trajectory(frames, trajectory)

    def rate_trajectory(
        self,
        frames: List[np.ndarray],
        trajectory: List[Tuple[str, int]],
    ) -> Optional[float]:
        """Force an LLM rating + cache update. Returns the rating or None."""
        if self.api_key is None or not frames:
            return None

        self.stats['llm_calls'] += 1
        try:
            resp = query_llm_with_frames(
                frames=frames,
                prompt=self.prompt,
                api_key=self.api_key,
                model=self.llm_model,
                max_tokens=self.llm_max_tokens,
            )
            text = _extract_text_from_response(resp)
            rating = parse_llm_score(text if isinstance(text, str) else str(text))
            if self.verbose:
                print(f"[LLMValueCache] response={text!r}  parsed={rating}")
        except Exception:
            logging.exception("LLMValueCache: LLM call failed")
            self.stats['llm_failures'] += 1
            return None

        if rating is None:
            self.stats['llm_failures'] += 1
            return None

        self._broadcast(rating, trajectory)
        return rating

    def _broadcast(self, rating: float, trajectory: List[Tuple[str, int]]) -> None:
        """Update cache entries along `trajectory` with credit decay.

        Steps closer to the end of the trajectory (= what the LLM
        actually saw) get full credit; earlier steps decay by
        credit_decay^(distance_from_end).
        """
        if not trajectory:
            return
        end = len(trajectory) - 1
        with self._lock:
            for i, (sh, a) in enumerate(trajectory):
                discount = self.credit_decay ** (end - i)
                target = rating * discount + self.default_value * (1.0 - discount)
                entry = self._table.setdefault((sh, a), CacheEntry(value=self.default_value))
                entry.value = (1.0 - self.ema_alpha) * entry.value + self.ema_alpha * target
                entry.visits += 1
                self.stats['updates'] += 1

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        with self._lock:
            serial = {
                'entries': [
                    {'state_hash': sh, 'action': a, 'value': e.value, 'visits': e.visits}
                    for (sh, a), e in self._table.items()
                ],
                'stats': self.stats,
            }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(serial, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            serial = json.load(f)
        with self._lock:
            for e in serial.get('entries', []):
                self._table[(e['state_hash'], int(e['action']))] = CacheEntry(
                    value=float(e['value']),
                    visits=int(e.get('visits', 1)),
                )

    def __len__(self) -> int:
        return len(self._table)
