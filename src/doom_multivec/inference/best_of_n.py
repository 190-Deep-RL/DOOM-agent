"""Best-of-N policy rollouts for DOOM action selection.

Stateless planner: at each frame, sample N action sequences from the model
policy (with temperature), simulate each forward H frames in VizDoom, score
by sign(Δkills) + 2·sign(Δhealth) + 2·sign(Δarmor), and retain the first
`retention_count` actions of the best sampled sequence.

Mirrors the current MCTSAgent in three ways so results are directly comparable:
- Composite action selection (4 base + 7 composite buttons = 11 actions).
- Root-replay state restoration: rollouts always restart from a single root
  snapshot rather than re-snapshotting mid-rollout (avoids the VizDoom
  save/load bleed-through when a rollout ends the episode).
- 4-tuple `get_action()` return shape matching MCTSAgent.get_action().
"""

import math
import os
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from doom_multivec.inference.llm_value_cache import LLMValueCache, ascii_state_hash


@dataclass
class RolloutResult:
    action_sequence: Tuple[int, ...]
    score: float
    rule_score: float = 0.0
    llm_score: Optional[float] = None  # mean cached value over trajectory, if any
    rollout_kills: int = 0
    rollout_damage: float = 0.0


class BestOfNAgent:
    """Best-of-N (predictive sampling / MPC) planner.

    Args:
        model: DoomMultiVecClassifier for policy evaluation.
        tokenizer: Transformer tokenizer for ASCII frames.
        converter: AsciiConverter for frame preprocessing.
        num_rollouts: Number of rollouts per decision (default: 25).
        rollout_depth: Frames per rollout (default: 20).
        retention_count: Number of leading actions from the best rollout to
            keep and replay before planning again (default: 1).
        temperature: Softmax temperature for policy sampling (default: 0.7).
        use_composite_moves: Expand action space to include 7 composite
            two-button moves (default: True).
        composite_logit_weights: Per-component weight when summing into a
            composite logit (default: [1.0, 1.0, 1.0, 1.0]).
        device: Torch device for model inference.
        frame_skip: Frames per action (default: 4).
        temp_dir: Directory for the root snapshot file.
    """

    ACTION_NAMES = ['shoot', 'move_forward', 'turn_left', 'turn_right']
    ACTION_TO_BUTTONS = {
        'shoot': [1, 0, 0, 0],
        'move_forward': [0, 1, 0, 0],
        'turn_left': [0, 0, 1, 0],
        'turn_right': [0, 0, 0, 1],
    }
    COMPOSITE_MOVES = {
        'move_forward+turn_left': [0, 1, 1, 0],
        'move_forward+turn_right': [0, 1, 0, 1],
        'move_forward+shoot': [1, 1, 0, 0],
        'turn_left+shoot': [1, 0, 1, 0],
        'turn_right+shoot': [1, 0, 0, 1],
        'shoot+move_forward+turn_left': [1, 1, 1, 0],
        'shoot+move_forward+turn_right': [1, 1, 0, 1],
    }
    BASE_NUM_ACTIONS = 4

    def __init__(
        self,
        model,
        tokenizer,
        converter,
        num_rollouts: int = 25,
        rollout_depth: int = 20,
        retention_count: int = 10,
        temperature: float = 0.1,
        use_composite_moves: bool = True,
        composite_logit_weights: Optional[List[float]] = None,
        device: str = 'cpu',
        frame_skip: int = 4,
        temp_dir: Optional[str] = None,
        llm_cache: Optional[LLMValueCache] = None,
        llm_blend: float = 0.3,
        llm_frames_per_rating: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.converter = converter
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.retention_count = max(1, int(retention_count))
        self.temperature = temperature
        self.use_composite_moves = use_composite_moves
        self.composite_logit_weights = composite_logit_weights or [50.0, 0.7, 5.0, 5.0]
        self.device = device
        self.frame_skip = frame_skip
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._save_counter = 0
        self.current_game = None
        self._root_save_path: Optional[str] = None
        self._retained_actions = deque()

        self.llm_cache = llm_cache
        self.llm_blend = llm_blend
        self.llm_frames_per_rating = llm_frames_per_rating

        self._build_action_mappings()
        self.num_actions = len(self.action_names)

        # Diagnostics exposed for driver UIs and benchmark sinks.
        self.last_action_stats: Dict[int, Dict[str, float]] = {}
        self._benchmark_sink: Dict[str, Dict[str, float]] = {}

    def _build_action_mappings(self) -> None:
        self.action_names: List[str] = list(self.ACTION_NAMES)
        self.action_to_buttons: Dict[str, List[int]] = dict(self.ACTION_TO_BUTTONS)
        if self.use_composite_moves:
            self.action_names.extend(self.COMPOSITE_MOVES.keys())
            self.action_to_buttons.update(self.COMPOSITE_MOVES)
            self.composite_action_components = {
                4: [1, 2],  # move_forward + turn_left
                5: [1, 3],  # move_forward + turn_right
                6: [0, 1],  # move_forward + shoot
                7: [0, 2],  # turn_left + shoot
                8: [0, 3],  # turn_right + shoot
                9: [0, 1, 2],  # shoot+move_forward+turn_left
                10: [0, 1, 3],  # shoot+move_forward+turn_right
            }
        else:
            self.composite_action_components = {}

    def set_game(self, game) -> None:
        self.current_game = game

    def reset(self) -> None:
        self._save_counter = 0
        self.last_action_stats = {}
        self._benchmark_sink = {}
        self._retained_actions.clear()
        self._cleanup_root()

    def advance_root(self, action_taken: int) -> None:
        return

    def _cleanup_root(self) -> None:
        if self._root_save_path and os.path.exists(self._root_save_path):
            try:
                os.remove(self._root_save_path)
            except OSError:
                pass
        self._root_save_path = None

    def _get_save_path(self) -> str:
        self._save_counter += 1
        return os.path.join(self.temp_dir, f'bon_save_{id(self)}_{self._save_counter}.zds')

    def _record_benchmark(self, key: str, elapsed_seconds: float) -> None:
        entry = self._benchmark_sink.setdefault(key, {'total_ms': 0.0, 'count': 0.0})
        entry['total_ms'] += elapsed_seconds * 1000.0
        entry['count'] += 1.0

    # ---------- model interface ----------

    def _ascii_from_game(self) -> Tuple[str, Optional[list]]:
        ascii_frame, depth_bins, _ = self._ascii_and_screen_from_game()
        return ascii_frame, depth_bins

    def _ascii_and_screen_from_game(self) -> Tuple[str, Optional[list], Optional[np.ndarray]]:
        state_obj = self.current_game.get_state()
        if state_obj is None:
            return '', None, None
        screen = state_obj.screen_buffer
        depth = state_obj.depth_buffer if hasattr(state_obj, 'depth_buffer') else None
        if screen.ndim == 3:
            gray = np.mean(screen, axis=2).astype(np.uint8)
        else:
            gray = screen
        if depth is not None:
            ascii_frame, depth_bins = self.converter.convert_with_depth(
                gray, depth.astype(np.float32), num_bins=16
            )
        else:
            ascii_frame = self.converter.convert_simple(gray)
            depth_bins = None
        return ascii_frame, depth_bins, screen

    def _prepare_model_input(self, ascii_frame: str, depth_bins: Optional[list]):
        encoded = self.tokenizer(
            ascii_frame,
            return_tensors='pt',
            max_length=1100,
            padding='max_length',
            truncation=True,
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        depth_ids = None
        if depth_bins is not None:
            no_depth = 16
            d = [no_depth]
            d.extend(depth_bins[:input_ids.shape[1] - 2])
            while len(d) < input_ids.shape[1]:
                d.append(no_depth)
            depth_ids = torch.tensor([d[:input_ids.shape[1]]], dtype=torch.long).to(self.device)
        return input_ids, attention_mask, depth_ids

    def _expand_composite_logits(self, base_logits: torch.Tensor) -> torch.Tensor:
        if not self.use_composite_moves:
            return base_logits
        expanded = base_logits.clone()
        for _, comps in self.composite_action_components.items():
            comp_logit = sum(self.composite_logit_weights[i] * base_logits[i] for i in comps)
            expanded = torch.cat([expanded, comp_logit.unsqueeze(0)])
        return expanded

    def _policy_probs(self) -> Optional[np.ndarray]:
        ascii_frame, depth_bins = self._ascii_from_game()
        if not ascii_frame:
            return None
        input_ids, attention_mask, depth_ids = self._prepare_model_input(ascii_frame, depth_bins)
        with torch.no_grad():
            result = self.model(input_ids, attention_mask, depth_ids=depth_ids)
            logits = result['logits'][0]
            logits = self._expand_composite_logits(logits)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:self.num_actions]

    def _sample_action(self) -> int:
        probs = self._policy_probs()
        if probs is None:
            return int(np.random.randint(self.num_actions))
        if self.temperature != 1.0:
            probs = np.power(probs, 1.0 / max(self.temperature, 1e-6))
            probs = probs / probs.sum()
        return int(np.random.choice(self.num_actions, p=probs))

    # ---------- rollout machinery ----------

    def _read_metrics(self) -> Tuple[float, float, float, float]:
        import vizdoom
        return (
            self.current_game.get_game_variable(vizdoom.GameVariable.HEALTH),
            self.current_game.get_game_variable(vizdoom.GameVariable.ARMOR),
            self.current_game.get_game_variable(vizdoom.GameVariable.KILLCOUNT),
            self.current_game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT),
        )

    def _restore_to_root(self) -> None:
        """Load the root snapshot and reset held buttons.

        Replay pattern from MCTSAgent: only ever load the root (a known-live
        state). Mid-rollout snapshots are never taken, so VizDoom's flaky
        load-after-finished behavior cannot bleed killcount/health into the
        live game.
        """
        self.current_game.load(self._root_save_path)
        self.current_game.make_action([0, 0, 0, 0], 1)

    def _rollout_once(self) -> RolloutResult:
        rt0 = time.perf_counter()
        self._restore_to_root()
        start = self._read_metrics()

        # Trajectory and frame collection for LLM cache. Both are no-ops
        # when llm_cache is None; the only cost is one ascii hash per step.
        trajectory: List[Tuple[str, int]] = []
        llm_frames: List[np.ndarray] = []
        # Evenly spaced frame-capture indices across the rollout.
        capture_indices = set()
        if self.llm_cache is not None and self.llm_frames_per_rating > 0:
            step = max(1, self.rollout_depth // self.llm_frames_per_rating)
            capture_indices = set(range(0, self.rollout_depth, step))

        start_value = self.current_game.get_total_reward()
        for i in range(self.rollout_depth):
            if self.current_game.is_episode_finished():
                break
            ascii_frame, depth_bins, screen = self._ascii_and_screen_from_game()
            if not ascii_frame:
                break
            # Sample using the just-captured policy (avoid a second forward pass).
            probs = self._probs_from_ascii(ascii_frame, depth_bins)
            if self.temperature != 1.0:
                probs = np.power(probs, 1.0 / max(self.temperature, 1e-6))
                probs = probs / probs.sum()
            a = int(np.random.choice(self.num_actions, p=probs))

            trajectory.append((ascii_state_hash(ascii_frame), a))
            if i in capture_indices and screen is not None:
                llm_frames.append(np.array(screen, copy=True))

            self.current_game.make_action(
                self.action_to_buttons[self.action_names[a]],
                self.frame_skip,
            )

        # Read end metrics regardless of whether the episode finished so
        # we can compute precise deltas (kills/damage) for this rollout.
        end = self._read_metrics()
        start_health, start_armor, start_kills, start_damage = start
        end_health, end_armor, end_kills, end_damage = end
        rollout_kills = int(max(0, end_kills - start_kills))
        rollout_damage = float(max(0.0, end_damage - start_damage))

        end_value = self.current_game.get_total_reward()
        rule_score = float(end_value - start_value)
        print(f"Rollout result: {rule_score:.2f}")

        # Cache: query existing entries for a value blend (always free).
        llm_value = None
        if self.llm_cache is not None and trajectory:
            llm_value = self.llm_cache.trajectory_value(trajectory, require_min_hits=1)
            # Probabilistically fire a fresh LLM rating; broadcast onto path.
            self.llm_cache.maybe_rate_trajectory(llm_frames, trajectory)

        # Blend cached LLM value into the score. Cached values are in
        # [0, 1] with 0.5 as the neutral prior, so subtract 0.5 so that a
        # neutral cache contributes zero (preserves rule-based ranking).
        if llm_value is not None:
            score = rule_score + self.llm_blend * (llm_value - 0.5)
        else:
            score = rule_score

        self._record_benchmark('rollout', time.perf_counter() - rt0)
        return RolloutResult(
            action_sequence=tuple(a for _, a in trajectory),
            score=score,
            rule_score=rule_score,
            llm_score=llm_value,
            rollout_kills=rollout_kills,
            rollout_damage=rollout_damage,
        )

    def _probs_from_ascii(self, ascii_frame: str, depth_bins) -> np.ndarray:
        """Policy probs from a precomputed ASCII frame (no game state read)."""
        input_ids, attention_mask, depth_ids = self._prepare_model_input(ascii_frame, depth_bins)
        with torch.no_grad():
            result = self.model(input_ids, attention_mask, depth_ids=depth_ids)
            logits = result['logits'][0]
            logits = self._expand_composite_logits(logits)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:self.num_actions]

    # ---------- public ----------

    def get_action(self) -> Tuple[str, List[int], int, Dict[str, Dict[str, float]], int, float]:
        if self.current_game is None:
            raise RuntimeError("Game not set. Call set_game() first.")

        if self._retained_actions:
            action_idx = int(self._retained_actions.popleft())
            action_name = self.action_names[action_idx]
            buttons = self.action_to_buttons[action_name]
            # No rollout stats available for retained actions; return zeros.
            return action_name, buttons, action_idx, dict(self._benchmark_sink), 0, 0.0

        t0 = time.perf_counter()
        # Root snapshot taken once per decision while game is live.
        self._cleanup_root()
        self._root_save_path = self._get_save_path()
        self.current_game.save(self._root_save_path)

        scores_by_action: Dict[int, list] = defaultdict(list)
        best_result: Optional[RolloutResult] = None
        last_simulation_kills = 0
        last_simulation_damage = 0.0
        try:
            for i in range(self.num_rollouts):
                result = self._rollout_once()
                # Print rollout score as it completes
                try:
                    print(f"Rollout {i+1}/{self.num_rollouts}: score={result.score:.2f} rule={result.rule_score:+.2f} llm={result.llm_score}")
                except Exception:
                    # Best-effort printing; don't crash on formatting
                    print(f"Rollout {i+1}/{self.num_rollouts}: score={result.score}")
                if result.action_sequence:
                    scores_by_action[result.action_sequence[0]].append(result.score)
                if best_result is None or result.score > best_result.score:
                    best_result = result
                # Track last rollout's kills/damage for fallback when rollouts fail
                last_simulation_kills = int(result.rollout_kills)
                last_simulation_damage = float(result.rollout_damage)
        finally:
            # Always restore the live game to the real root, regardless of
            # whether any rollout finished the episode.
            try:
                self._restore_to_root()
            finally:
                self._cleanup_root()

        stats = {}
        for a in range(self.num_actions):
            xs = scores_by_action.get(a, [])
            stats[a] = {
                'count': len(xs),
                'mean': float(np.mean(xs)) if xs else float('-inf'),
            }
        self.last_action_stats = stats

        if best_result is None or not best_result.action_sequence:
            probs = self._policy_probs()
            best_action = int(np.argmax(probs)) if probs is not None else 0
            retained_actions = [best_action]
        else:
            best_action = int(best_result.action_sequence[0])
            retained_actions = list(best_result.action_sequence[: self.retention_count])

        self._retained_actions.extend(retained_actions[1:])

        action_name = self.action_names[best_action]
        buttons = self.action_to_buttons[action_name]

        self._record_benchmark('get_action', time.perf_counter() - t0)
        return action_name, buttons, best_action, dict(self._benchmark_sink), last_simulation_kills, last_simulation_damage
