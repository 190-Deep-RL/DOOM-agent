"""Beam search over action sequences for DOOM action selection.

Stateless planner: at each frame, maintain a beam of B partial action
sequences. At each depth d, expand each beam item by its top-k policy
actions, simulate one step, score by cumulative reward delta +
prior_weight * log-policy, keep top-B. After D steps, return the first
action of the highest-scoring beam.

Mirrors the current MCTSAgent in three ways so results are directly comparable:
- Composite action selection (4 base + 5 composite buttons = 9 actions).
- Root-replay state restoration: a single root snapshot is taken per
  decision, and any beam item's state is reproduced by loading the root
  and replaying its action prefix. No mid-search snapshots, no
  load-after-finished bleed-through.
- 4-tuple `get_action()` return shape matching MCTSAgent.get_action().
"""

import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from doom_multivec.inference.llm_value_cache import LLMValueCache, ascii_state_hash


@dataclass
class BeamItem:
    actions: List[int]
    log_prob: float
    score: float
    metrics: Tuple[float, float, float]  # (health, armor, killcount) at end of sequence
    finished: bool = False
    # Trajectory of (state_hash, action) along this beam, for LLM cache.
    trajectory: List[Tuple[str, int]] = None  # type: ignore
    # One screen frame captured at the tip (None until rated).
    tip_frame: Optional[np.ndarray] = None
    # LLM rating once a leaf eval has fired, or None.
    llm_rating: Optional[float] = None

    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []


class BeamSearchAgent:
    """Beam search planner over short action sequences.

    Args:
        model: DoomMultiVecClassifier for policy evaluation.
        tokenizer: Transformer tokenizer for ASCII frames.
        converter: AsciiConverter for frame preprocessing.
        beam_width: Sequences kept after each pruning step (default: 4).
        beam_depth: Lookahead horizon in actions (default: 8).
        top_k: Children expanded per beam item per step (default: 2).
        prior_weight: Coefficient on cumulative log-policy in the rank
            score (default: 0.5).
        prior_temperature: Sharpening applied to the base 4-way model logits
            before composite expansion (default: 1.0).
        use_composite_moves: Expand action space to include 5 composite
            two-button moves (default: True). When enabled, `top_k` selects
            from 9 actions instead of 4.
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
    }
    BASE_NUM_ACTIONS = 4

    def __init__(
        self,
        model,
        tokenizer,
        converter,
        beam_width: int = 4,
        beam_depth: int = 8,
        top_k: int = 2,
        prior_weight: float = 0.5,
        prior_temperature: float = 1.0,
        use_composite_moves: bool = True,
        composite_logit_weights: Optional[List[float]] = None,
        device: str = 'cpu',
        frame_skip: int = 4,
        temp_dir: Optional[str] = None,
        llm_cache: Optional[LLMValueCache] = None,
        llm_leaf_eval: bool = True,
        llm_blend: float = 0.3,
        llm_cache_blend: float = 0.15,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.converter = converter
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.prior_weight = prior_weight
        self.prior_temperature = prior_temperature
        self.use_composite_moves = use_composite_moves
        self.composite_logit_weights = composite_logit_weights or [1.0, 1.0, 1.0, 1.0]
        self.device = device
        self.frame_skip = frame_skip
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._save_counter = 0
        self.current_game = None
        self._root_save_path: Optional[str] = None

        # llm_leaf_eval forces an LLM call per surviving beam tip every
        # decision; llm_blend weights the resulting per-beam rating into
        # the final reranker; llm_cache_blend weights cached per-step
        # values into intermediate step deltas during expansion.
        self.llm_cache = llm_cache
        self.llm_leaf_eval = llm_leaf_eval
        self.llm_blend = llm_blend
        self.llm_cache_blend = llm_cache_blend

        self._build_action_mappings()
        self.num_actions = len(self.action_names)
        self.top_k = min(top_k, self.num_actions)

        # Diagnostics exposed for driver UIs and benchmark sinks.
        self.last_top_beams: List[BeamItem] = []
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
            }
        else:
            self.composite_action_components = {}

    def set_game(self, game) -> None:
        self.current_game = game

    def reset(self) -> None:
        self._save_counter = 0
        self.last_top_beams = []
        self._benchmark_sink = {}
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
        return os.path.join(self.temp_dir, f'beam_save_{id(self)}_{self._save_counter}.zds')

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
        return self._probs_from_ascii(ascii_frame, depth_bins)

    def _probs_from_ascii(self, ascii_frame: str, depth_bins) -> np.ndarray:
        input_ids, attention_mask, depth_ids = self._prepare_model_input(ascii_frame, depth_bins)
        with torch.no_grad():
            result = self.model(input_ids, attention_mask, depth_ids=depth_ids)
            logits = result['logits'][0]
            logits = self._expand_composite_logits(logits)
            if self.prior_temperature != 1.0:
                logits = logits / self.prior_temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:self.num_actions]

    # ---------- replay-based state restoration ----------

    def _read_metrics(self) -> Tuple[float, float, float]:
        import vizdoom
        return (
            self.current_game.get_game_variable(vizdoom.GameVariable.HEALTH),
            self.current_game.get_game_variable(vizdoom.GameVariable.ARMOR),
            self.current_game.get_game_variable(vizdoom.GameVariable.KILLCOUNT),
        )

    def _step_score(self, start, end) -> float:
        health_d = np.sign(end[0] - start[0])
        armor_d = np.sign(end[1] - start[1])
        kill_d = np.sign(end[2] - start[2])
        return float(kill_d + 2 * health_d + 2 * armor_d)

    def _rank(self, item: BeamItem) -> float:
        base = item.score + self.prior_weight * item.log_prob
        if item.llm_rating is not None:
            base += self.llm_blend * (item.llm_rating - 0.5)
        return base

    def _restore_to(self, action_sequence: List[int]) -> bool:
        """Load root + replay actions. Returns False if episode ended mid-replay."""
        rt0 = time.perf_counter()
        self.current_game.load(self._root_save_path)
        self.current_game.make_action([0, 0, 0, 0], 1)
        finished = False
        for a in action_sequence:
            if self.current_game.is_episode_finished():
                finished = True
                break
            self.current_game.make_action(
                self.action_to_buttons[self.action_names[a]],
                self.frame_skip,
            )
        self._record_benchmark('restore', time.perf_counter() - rt0)
        return not finished

    # ---------- public ----------

    def get_action(self) -> Tuple[str, List[int], int, Dict[str, Dict[str, float]]]:
        if self.current_game is None:
            raise RuntimeError("Game not set. Call set_game() first.")

        t0 = time.perf_counter()
        self._cleanup_root()
        self._root_save_path = self._get_save_path()
        self.current_game.save(self._root_save_path)
        root_metrics = self._read_metrics()

        beam: List[BeamItem] = [BeamItem(
            actions=[],
            log_prob=0.0,
            score=0.0,
            metrics=root_metrics,
            finished=False,
        )]

        try:
            for _ in range(self.beam_depth):
                candidates: List[BeamItem] = []
                for item in beam:
                    if item.finished:
                        candidates.append(item)
                        continue

                    # Restore to this beam-item's state and read its ASCII once.
                    self._restore_to(item.actions)
                    ascii_frame, depth_bins = self._ascii_from_game()
                    if not ascii_frame:
                        candidates.append(item)
                        continue
                    probs = self._probs_from_ascii(ascii_frame, depth_bins)
                    parent_hash = ascii_state_hash(ascii_frame)

                    top_actions = np.argsort(-probs)[:self.top_k]
                    for a in top_actions:
                        a = int(a)
                        # Reach (item.actions + [a]) by replaying from root.
                        self._restore_to(item.actions + [a])
                        finished = self.current_game.is_episode_finished()
                        if finished:
                            end_metrics = item.metrics
                            step_delta = 0.0
                        else:
                            end_metrics = self._read_metrics()
                            step_delta = self._step_score(item.metrics, end_metrics)

                        # Blend cached LLM value into step delta if available.
                        # Centered around 0.5 so a neutral cache contributes zero.
                        if self.llm_cache is not None and self.llm_cache_blend > 0.0:
                            cached = self.llm_cache.lookup(parent_hash, a)
                            if cached is not None:
                                step_delta += self.llm_cache_blend * (cached - 0.5)

                        candidates.append(BeamItem(
                            actions=item.actions + [a],
                            log_prob=item.log_prob + math.log(max(float(probs[a]), 1e-8)),
                            score=item.score + step_delta,
                            metrics=end_metrics,
                            finished=finished,
                            trajectory=item.trajectory + [(parent_hash, a)],
                        ))

                if not candidates:
                    break

                candidates.sort(key=self._rank, reverse=True)
                beam = candidates[:self.beam_width]

            # Optional leaf eval: rate each surviving beam tip with one LLM
            # call per beam, broadcast the rating back onto its trajectory,
            # and update each beam's llm_rating so _rank uses it.
            if (self.llm_cache is not None
                    and self.llm_leaf_eval
                    and self.llm_cache.api_key is not None
                    and self.llm_blend > 0.0):
                for item in beam:
                    if item.finished or not item.actions:
                        continue
                    # Reach the tip and grab a frame for the LLM.
                    self._restore_to(item.actions)
                    _, _, screen = self._ascii_and_screen_from_game()
                    if screen is None:
                        continue
                    rating = self.llm_cache.rate_trajectory(
                        [np.array(screen, copy=True)],
                        item.trajectory,
                    )
                    if rating is not None:
                        item.llm_rating = rating

            beam.sort(key=self._rank, reverse=True)
            self.last_top_beams = beam[:3]

            best = beam[0]
            best_action = best.actions[0] if best.actions else 0
        finally:
            # Restore live game to the real root regardless of beam internals.
            try:
                self.current_game.load(self._root_save_path)
                self.current_game.make_action([0, 0, 0, 0], 1)
            finally:
                self._cleanup_root()

        action_name = self.action_names[best_action]
        buttons = self.action_to_buttons[action_name]

        self._record_benchmark('get_action', time.perf_counter() - t0)
        return action_name, buttons, best_action, dict(self._benchmark_sink)
