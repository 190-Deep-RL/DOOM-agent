"""
Benchmark DOOM agents: MultiVec Classifier vs LLM APIs.

Runs multiple episodes of a scenario and collects metrics:
  - Survival time (steps)
  - Kills
  - Health remaining
  - Actions per second (latency)
  - Action diversity (entropy)

Usage:
  # Benchmark our model:
  python scripts/benchmark.py --agent multivec --model models/doom-multivec-trained --episodes 20

  # Benchmark GPT-4-mini:
  python scripts/benchmark.py --agent gpt4mini --episodes 10

  # Benchmark GPT-5:
  python scripts/benchmark.py --agent gpt5 --episodes 10

  # Benchmark with DPO actor head:
  python scripts/benchmark.py --agent multivec --model models/doom-multivec-5L --actor-head output/dpo-v1/final --episodes 10

  # Compare all:
  python scripts/benchmark.py --agent all --episodes 10
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.ascii.converter import AsciiConverter


# ================================================================
# Metrics
# ================================================================
def compute_metrics(episodes):
    """Compute aggregate metrics from episode results."""
    if not episodes:
        return {}

    survival = [ep['steps'] for ep in episodes]
    kills = [ep['kills'] for ep in episodes]
    health = [ep['health_remaining'] for ep in episodes]
    armor = [ep['armor_remaining'] for ep in episodes]
    damage = [ep['damage_dealt'] for ep in episodes]
    latencies = []
    for ep in episodes:
        latencies.extend(ep['latencies'])
    all_actions = Counter()
    for ep in episodes:
        all_actions.update(ep['action_counts'])

    # Action diversity: entropy of action distribution
    total_actions = sum(all_actions.values())
    if total_actions > 0:
        probs = np.array([all_actions[a] / total_actions for a in all_actions])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0

    return {
        'episodes': len(episodes),
        'avg_survival_steps': np.mean(survival),
        'max_survival_steps': max(survival),
        'avg_kills': np.mean(kills),
        'total_kills': sum(kills),
        'avg_health_remaining': np.mean(health),
        'avg_armor_remaining': np.mean(armor),
        'total_damage_dealt': sum(damage),
        'avg_damage_dealt': np.mean(damage),
        'avg_latency_ms': np.mean(latencies) if latencies else 0,
        'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
        'action_diversity_entropy': entropy,
        'action_distribution': dict(all_actions.most_common()),
    }


# ================================================================
# DOOM Setup
# ================================================================
def setup_game(scenario='defend_the_center', match_visual=False, visible=False):
    game = vizdoom.DoomGame()
    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
        'deathmatch': vizdoom.scenarios_path + '/deathmatch.cfg',
    }
    game.load_config(scenarios.get(scenario, scenario))
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_window_visible(visible)
    game.set_mode(vizdoom.Mode.PLAYER)

    if match_visual:
        # Match play_doom_visual.py settings exactly (training data was recorded with these)
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        game.set_render_hud(True)
        game.set_episode_timeout(2100)  # ~60 seconds
    else:
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
        game.set_render_hud(False)
        game.set_episode_timeout(4200)

    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)
    game.add_available_button(vizdoom.Button.TURN_LEFT)
    game.add_available_button(vizdoom.Button.TURN_RIGHT)

    game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vizdoom.GameVariable.ARMOR)
    game.add_available_game_variable(vizdoom.GameVariable.DAMAGECOUNT)

    game.init()
    game.set_seed(np.random.randint(0, 100000))
    return game


ACTION_NAMES = ['shoot', 'move_forward', 'turn_left', 'turn_right']
ACTION_BUTTONS = {
    'shoot':        [1, 0, 0, 0],
    'move_forward': [0, 1, 0, 0],
    'turn_left':    [0, 0, 1, 0],
    'turn_right':   [0, 0, 0, 1],
}


# ================================================================
# Agent: MultiVec Classifier (with optional DPO actor head)
# ================================================================
class MultiVecAgent:
    def __init__(self, model_path, actor_head_path=None):
        import torch
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.converter = AsciiConverter(width=40, height=25)

        # Check if loading DPO policy (actor head)
        self.actor_head_path = actor_head_path
        if actor_head_path or os.path.exists(os.path.join(model_path, 'actor_head.pt')):
            # Load DPO policy
            from doom_multivec.model.dpo_policy import DPODoomPolicy

            dpo_path = actor_head_path if actor_head_path else model_path
            self.model = DPODoomPolicy.from_pretrained(dpo_path, encoder_path=model_path)
            self.model.eval()
            self.name = f"MultiVec-DPO-{sum(p.numel() for p in self.model.parameters())/1e6:.1f}M"
            self.is_dpo = True
        else:
            # Load standard classifier
            from doom_multivec.model.classifier import DoomMultiVecClassifier

            state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
            num_actions = 4
            for key in state:
                if 'classifier.weight' in key:
                    num_actions = state[key].shape[0]
                    break
            self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
            self.model.load_state_dict(state)
            self.model.eval()
            self.name = f"MultiVec-{sum(p.numel() for p in self.model.parameters())/1e6:.1f}M"
            self.is_dpo = False

    def get_action(self, screen, depth):
        import torch
        from doom_multivec.model.classifier import DoomMultiVecClassifier

        gray = np.mean(screen, axis=2).astype(np.uint8) if screen.ndim == 3 else screen

        if depth is not None:
            ascii_text, depth_bins = self.converter.convert_with_depth(
                gray, depth.astype(np.float32), num_bins=16
            )
        else:
            ascii_text = self.converter.convert_simple(gray)
            depth_bins = None

        encoded = self.tokenizer(ascii_text, return_tensors='pt', max_length=1100,
                                padding='max_length', truncation=True)
        depth_ids = None
        if depth_bins is not None:
            no_depth = 16
            d = [no_depth]
            for k in range(min(len(depth_bins), encoded['input_ids'].shape[1] - 2)):
                d.append(depth_bins[k])
            while len(d) < encoded['input_ids'].shape[1]:
                d.append(no_depth)
            depth_ids = torch.tensor([d[:encoded['input_ids'].shape[1]]], dtype=torch.long)

        with torch.no_grad():
            result = self.model(encoded['input_ids'], encoded['attention_mask'], depth_ids=depth_ids)
            probs = torch.softmax(result['logits'], dim=-1)[0].numpy()

        # Support both DoomMultiVecClassifier and DPODoomPolicy action names
        num_actions = len(probs)
        if hasattr(self.model, 'ACTION_NAMES'):
            action_names = self.model.ACTION_NAMES[:num_actions]
        else:
            action_names = DoomMultiVecClassifier.ACTION_NAMES[:num_actions]

        # Use appropriate button mapping based on number of actions
        action_buttons = ACTION_BUTTONS if num_actions <= 4 else {
            'shoot':        [1, 0, 0, 0, 0, 0],
            'move_forward': [0, 1, 0, 0, 0, 0],
            'turn_left':    [0, 0, 1, 0, 0, 0],
            'turn_right':   [0, 0, 0, 1, 0, 0],
            'strafe_left':  [0, 0, 0, 0, 1, 0],
            'strafe_right': [0, 0, 0, 0, 0, 1],
        }

        sorted_idx = np.argsort(probs)[::-1]
        top_action = action_names[sorted_idx[0]]
        buttons = list(action_buttons[top_action])

        # Shoot boost — exact same logic as play_doom_visual.py
        shoot_idx = action_names.index('shoot') if 'shoot' in action_names else -1
        if shoot_idx >= 0 and top_action != 'shoot':
            top_prob = probs[sorted_idx[0]]
            shoot_prob = probs[shoot_idx]
            if shoot_prob > top_prob * 0.75:
                shoot_buttons = action_buttons['shoot']
                buttons = [max(a, b) for a, b in zip(buttons, shoot_buttons)]
                top_action = f"{top_action}+shoot"

        # Combine top-2 if compatible (movement + rotation)
        if len(sorted_idx) > 1 and probs[sorted_idx[1]] > 0.15:
            second_action = action_names[sorted_idx[1]]
            if second_action != 'shoot':
                movement = {'move_forward'}
                rotation = {'turn_left', 'turn_right'}
                top_base = top_action.split('+')[0]
                top_cat = 'move' if top_base in movement else 'rot'
                sec_cat = 'move' if second_action in movement else 'rot'
                if top_cat != sec_cat:
                    sec_buttons = action_buttons[second_action]
                    buttons = [max(a, b) for a, b in zip(buttons, sec_buttons)]
                    top_action = f"{top_action}+{second_action}"

        return top_action, buttons


# ================================================================
# Agent: LLM (OpenAI API)
# ================================================================
class LLMAgent:
    """LLM agent via OpenAI-compatible API (supports OpenAI + OpenRouter)."""

    SYSTEM_PROMPT = """You are an AI agent playing the classic game DOOM. Each turn you receive:
1. The game view as ASCII art (brightness: " .:-=+*#%@", dark to bright)
2. A depth map with the same layout (0=very near, 9=very far)

Use both the ASCII view and depth map to decide your action.

Available actions (respond with one or combine two with '+'):
  shoot, move_forward, turn_left, turn_right

Examples of valid responses:
  shoot
  move_forward
  turn_left+shoot
  move_forward+turn_right

Respond with ONLY your chosen action(s). No explanation."""

    def __init__(self, model_name='gpt-4o-mini', base_url=None, api_key=None):
        from openai import OpenAI
        kwargs = {'timeout': 120.0}
        if base_url:
            kwargs['base_url'] = base_url
        if api_key:
            kwargs['api_key'] = api_key
        self.client = OpenAI(**kwargs)
        self.model_name = model_name
        self.is_reasoning = any(r in model_name for r in ['gpt-5', 'o1', 'o3', 'deepseek-r1', 'qwen3', 'nemotron'])
        # Same resolution as our model for fair comparison
        self.converter = AsciiConverter(width=40, height=25)
        self.name = model_name.split('/')[-1]  # short name for display

    def get_action(self, screen, depth):
        gray = np.mean(screen, axis=2).astype(np.uint8) if screen.ndim == 3 else screen
        ascii_frame = self.converter.convert_simple(gray)

        # Build depth text (0=near, 9=far) matching the ASCII layout
        depth_text = ""
        if depth is not None:
            depth_resized = self.converter._downscale(
                depth.astype(np.float32) if depth.dtype != np.float32 else depth,
                self.converter.height, self.converter.width
            )
            d_min, d_max = depth_resized.min(), depth_resized.max()
            if d_max > d_min:
                depth_norm = (depth_resized - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_resized)
            depth_quantized = np.clip((depth_norm * 10).astype(int), 0, 9)
            rows = []
            for y in range(self.converter.height):
                rows.append(''.join(str(depth_quantized[y, x]) for x in range(self.converter.width)))
            depth_text = '\n'.join(rows)

        user_content = f"View:\n```\n{ascii_frame}\n```"
        if depth_text:
            user_content += f"\n\nDepth (0=near, 9=far):\n```\n{depth_text}\n```"

        try:
            kwargs = {
                'model': self.model_name,
                'messages': [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }
            if self.is_reasoning:
                kwargs['max_completion_tokens'] = 4000
                # Recommended temps per model card
                if 'qwen3' in self.model_name:
                    kwargs['temperature'] = 0.6
                    kwargs['top_p'] = 0.95
                elif 'nemotron' in self.model_name:
                    kwargs['temperature'] = 0.6
                    kwargs['top_p'] = 0.95
                elif 'gpt-5' in self.model_name:
                    pass  # GPT-5 reasoning: no temp control
                else:
                    kwargs['temperature'] = 0.6
            else:
                kwargs['max_tokens'] = 200
                # Recommended temps per model card
                if 'gemini' in self.model_name or 'gemma' in self.model_name:
                    kwargs['temperature'] = 1.0
                    kwargs['top_p'] = 0.95
                elif 'gpt-4o-mini' in self.model_name:
                    kwargs['temperature'] = 0.3
                else:
                    kwargs['temperature'] = 0.7

            response = self.client.chat.completions.create(**kwargs)
            if not response.choices or not response.choices[0].message.content:
                return 'move_forward', ACTION_BUTTONS['move_forward']
            action_text = response.choices[0].message.content.strip().lower()
            # Extract last line (reasoning models may think first)
            lines = [l.strip() for l in action_text.split('\n') if l.strip()]
            if lines:
                action_text = lines[-1]

            # Parse actions
            buttons = [0, 0, 0, 0]
            parsed_actions = []
            for action in ACTION_NAMES:
                if action in action_text:
                    action_buttons = ACTION_BUTTONS[action]
                    buttons = [max(a, b) for a, b in zip(buttons, action_buttons)]
                    parsed_actions.append(action)

            if parsed_actions:
                return '+'.join(parsed_actions), buttons

            return 'move_forward', ACTION_BUTTONS['move_forward']
        except Exception as e:
            return 'move_forward', ACTION_BUTTONS['move_forward']


# ================================================================
# Agent: Random baseline
# ================================================================
class RandomAgent:
    def __init__(self):
        self.name = "Random"
        self.rng = np.random.default_rng(42)

    def get_action(self, screen, depth):
        action = ACTION_NAMES[self.rng.integers(0, len(ACTION_NAMES))]
        return action, ACTION_BUTTONS[action]


# ================================================================
# Agent: MCTS (Monte Carlo Tree Search)
# ================================================================
class MCTSAgentBenchmark:
    """MCTS agent wrapper for benchmarking."""

    def __init__(self, model_path, actor_head_path=None, simulations=25, depth=20, exploration=1.414, batch_size=1):
        from doom_multivec.model.classifier import DoomMultiVecClassifier
        from doom_multivec.model.dpo_policy import DPODoomPolicy
        from doom_multivec.inference.mcts import MCTSAgent
        from transformers import AutoTokenizer
        import torch

        self.converter = AsciiConverter(width=40, height=25)
        self.simulations = simulations
        self.depth = depth
        self.exploration = exploration
        self.batch_size = batch_size

        # Check if loading DPO policy
        if actor_head_path or os.path.exists(os.path.join(model_path, 'actor_head.pt')):
            dpo_path = actor_head_path if actor_head_path else model_path
            self.model = DPODoomPolicy.from_pretrained(dpo_path, encoder_path=model_path)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = self.model.num_actions
            self.is_dpo = True
            suffix = "-DPO" if actor_head_path else ""
            self.name = f"MCTS{suffix}-{simulations}sim-{depth}d"
        else:
            # Load standard classifier
            state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
            num_actions = 4
            for key in state:
                if 'classifier.weight' in key:
                    num_actions = state[key].shape[0]
                    break
            self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
            self.model.load_state_dict(state)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = num_actions
            self.is_dpo = False
            self.name = f"MCTS-{simulations}sim-{depth}d"

        self.mcts_agent = None
        self.game = None

    def set_game(self, game):
        """Set the game and initialize MCTS agent."""
        self.game = game
        from doom_multivec.inference.mcts import MCTSAgent
        self.mcts_agent = MCTSAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            converter=self.converter,
            num_simulations=self.simulations,
            rollout_depth=self.depth,
            exploration_constant=self.exploration,
            num_actions=self.num_actions,
            device='cpu',
            batch_size=self.batch_size,
        )
        self.mcts_agent.set_game(game)

    def reset(self):
        """Reset MCTS agent for new episode."""
        if self.mcts_agent:
            self.mcts_agent.reset()

    def advance_root(self, action_idx):
        """Advance MCTS tree root."""
        if self.mcts_agent:
            self.mcts_agent.advance_root(action_idx)

    def get_action(self, screen, depth):
        """Get action from MCTS agent."""
        if self.mcts_agent is None:
            return 'move_forward', ACTION_BUTTONS['move_forward']

        action_name, buttons, action_idx, _, _, _, _, _ = self.mcts_agent.get_action()
        return action_name, buttons, action_idx


# ================================================================
# Agent: Best-of-N (BoN)
# ================================================================
class BoNAgentBenchmark:
    """Best-of-N agent wrapper for benchmarking."""

    def __init__(self, model_path, actor_head_path=None, num_rollouts=25, rollout_depth=20, temperature=0.1,
                 llm_eval=False, llm_api_key=None, llm_sample_rate=0.05, llm_blend=0.3,
                 llm_cache_path=None, llm_verbose=False):
        from doom_multivec.model.classifier import DoomMultiVecClassifier
        from doom_multivec.model.dpo_policy import DPODoomPolicy
        from doom_multivec.inference.best_of_n import BestOfNAgent
        from doom_multivec.inference.llm_value_cache import LLMValueCache
        from transformers import AutoTokenizer
        from pathlib import Path
        import torch

        self.converter = AsciiConverter(width=40, height=25)
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.temperature = temperature

        # Build LLM cache if enabled
        self.llm_cache = None
        if llm_eval:
            api_key = llm_api_key or os.environ.get('TRITON_API_KEY')
            if api_key is None:
                print("WARNING: BoN LLM eval enabled but no API key found")
            # Default rubric
            default_rubric = Path(__file__).resolve().parent / '..' / 'src' / 'doom_multivec' / 'inference' / 'rubric.txt'
            rubric_text = ""
            if default_rubric.exists():
                rubric_text = default_rubric.read_text()
            self.llm_cache = LLMValueCache(
                api_key=api_key,
                prompt=rubric_text or "Rate the gameplay shown 0.0 (bad) to 1.0 (good).",
                sample_rate=llm_sample_rate,
                ema_alpha=0.3,
                credit_decay=0.95,
                verbose=llm_verbose,
            )
            if llm_cache_path and Path(llm_cache_path).exists():
                self.llm_cache.load(llm_cache_path)
                print(f"Loaded LLM cache from {llm_cache_path} ({len(self.llm_cache)} entries)")

        # Check if loading DPO policy
        if actor_head_path or os.path.exists(os.path.join(model_path, 'actor_head.pt')):
            dpo_path = actor_head_path if actor_head_path else model_path
            self.model = DPODoomPolicy.from_pretrained(dpo_path, encoder_path=model_path)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = self.model.num_actions
            self.is_dpo = True
            suffix = "-DPO" if actor_head_path else ""
            llm_suffix = "-LLM" if llm_eval else ""
            self.name = f"BoN{suffix}{llm_suffix}-{num_rollouts}r-{rollout_depth}d"
        else:
            # Load standard classifier
            state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
            num_actions = 4
            for key in state:
                if 'classifier.weight' in key:
                    num_actions = state[key].shape[0]
                    break
            self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
            self.model.load_state_dict(state)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = num_actions
            self.is_dpo = False
            llm_suffix = "-LLM" if llm_eval else ""
            self.name = f"BoN{llm_suffix}-{num_rollouts}r-{rollout_depth}d"

        self.bon_agent = None
        self.game = None
        self.llm_blend = llm_blend

    def set_game(self, game):
        """Set the game and initialize BoN agent."""
        self.game = game
        from doom_multivec.inference.best_of_n import BestOfNAgent
        self.bon_agent = BestOfNAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            converter=self.converter,
            num_rollouts=self.num_rollouts,
            rollout_depth=self.rollout_depth,
            temperature=self.temperature,
            device='cpu',
            llm_cache=self.llm_cache,
            llm_blend=self.llm_blend,
        )
        self.bon_agent.set_game(game)

    def reset(self):
        """Reset BoN agent for new episode."""
        if self.bon_agent:
            self.bon_agent.reset()

    def advance_root(self, action_idx):
        """Advance BoN tree root."""
        if self.bon_agent:
            self.bon_agent.advance_root(action_idx)

    def get_action(self, screen, depth):
        """Get action from BoN agent."""
        if self.bon_agent is None:
            return 'move_forward', ACTION_BUTTONS['move_forward']

        action_name, buttons, action_idx, _, _, _ = self.bon_agent.get_action()
        return action_name, buttons, action_idx


# ================================================================
# Agent: Beam Search
# ================================================================
class BeamAgentBenchmark:
    """Beam search agent wrapper for benchmarking."""

    def __init__(self, model_path, actor_head_path=None, beam_width=4, beam_depth=8, top_k=2):
        from doom_multivec.model.classifier import DoomMultiVecClassifier
        from doom_multivec.model.dpo_policy import DPODoomPolicy
        from doom_multivec.inference.beam_search import BeamSearchAgent
        from transformers import AutoTokenizer
        import torch

        self.converter = AsciiConverter(width=40, height=25)
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.top_k = top_k

        # Check if loading DPO policy
        if actor_head_path or os.path.exists(os.path.join(model_path, 'actor_head.pt')):
            dpo_path = actor_head_path if actor_head_path else model_path
            self.model = DPODoomPolicy.from_pretrained(dpo_path, encoder_path=model_path)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = self.model.num_actions
            self.is_dpo = True
            suffix = "-DPO" if actor_head_path else ""
            self.name = f"Beam{suffix}-{beam_width}w-{beam_depth}d"
        else:
            # Load standard classifier
            state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
            num_actions = 4
            for key in state:
                if 'classifier.weight' in key:
                    num_actions = state[key].shape[0]
                    break
            self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
            self.model.load_state_dict(state)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.num_actions = num_actions
            self.is_dpo = False
            self.name = f"Beam-{beam_width}w-{beam_depth}d"

        self.beam_agent = None
        self.game = None

    def set_game(self, game):
        """Set the game and initialize Beam agent."""
        self.game = game
        from doom_multivec.inference.beam_search import BeamSearchAgent
        self.beam_agent = BeamSearchAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            converter=self.converter,
            beam_width=self.beam_width,
            beam_depth=self.beam_depth,
            top_k=self.top_k,
            device='cpu',
        )
        self.beam_agent.set_game(game)

    def reset(self):
        """Reset Beam agent for new episode."""
        if self.beam_agent:
            self.beam_agent.reset()

    def advance_root(self, action_idx):
        """Advance Beam tree root."""
        if self.beam_agent:
            self.beam_agent.advance_root(action_idx)

    def get_action(self, screen, depth):
        """Get action from Beam agent."""
        if self.beam_agent is None:
            return 'move_forward', ACTION_BUTTONS['move_forward']

        action_name, buttons, action_idx, _ = self.beam_agent.get_action()
        return action_name, buttons, action_idx


def arming_sequence(game):
    for _ in range(5):
        game.advance_action(1)
    game.send_game_command("give Backpack")
    for _ in range(3):
        game.advance_action(1)
    game.send_game_command("give PlasmaRifle")
    for _ in range(6):
        game.send_game_command("give CellPack")
        game.advance_action(1)
    game.send_game_command("give BlueArmor")
    game.send_game_command("use PlasmaRifle")
    game.advance_action(1)

# ================================================================
# Benchmark Runner
# ================================================================
# ================================================================
# Technique-based Benchmark Runner (for MCTS/BoN/Beam agents)
# ================================================================
def run_benchmark_technique(agent, scenario, episodes, frame_skip=4, realtime=False, armed=False, max_steps=None):
    """Run benchmark for technique-based agents (MCTS, BoN, Beam).

    These agents manage their own game state and require special handling.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {agent.name}")
    print(f"  Scenario: {scenario}")
    print(f"  Episodes: {episodes}")
    if realtime:
        print(f"  Pacing: REAL-TIME (frame_skip={frame_skip})")
    else:
        print(f"  Pacing: as-fast-as-possible")
    print(f"{'='*60}")

    # Setup game using the agent's preferred method
    game = setup_game(scenario, match_visual=realtime, visible=True)
    game.set_episode_timeout(400)
    game.set_window_visible(False)  # Headless for benchmark

    # Set game for the agent
    agent.set_game(game)

    results = []
    frame_interval = frame_skip / 35.0

    for ep in range(episodes):
        game.new_episode()
        agent.reset()

        if armed:
            arming_sequence(game)

        step = 0
        latencies = []
        action_counts = Counter()
        kills = 0
        last_health = 100
        last_armor = 0
        damage_dealt = 0

        while not game.is_episode_finished() and (max_steps is None or step < max_steps):
            frame_start = time.perf_counter()

            try:
                last_health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
                last_armor = game.get_game_variable(vizdoom.GameVariable.ARMOR)
                damage_dealt = game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT)
            except:
                pass

            # Get action from technique agent
            t0 = time.perf_counter()
            action_name, buttons, action_idx = agent.get_action(None, None)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            action_counts[action_name] += 1

            # Execute action
            reward = game.make_action(buttons, frame_skip)
            if reward > 0:
                kills += int(reward)
            step += 1

            # Advance the agent's tree
            agent.advance_root(action_idx)

            # Real-time pacing
            if realtime:
                elapsed = time.perf_counter() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        results.append({
            'steps': step,
            'kills': kills,
            'health_remaining': max(0, last_health),
            'armor_remaining': max(0, last_armor),
            'damage_dealt': damage_dealt,
            'latencies': latencies,
            'action_counts': dict(action_counts),
        })

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Episode {ep+1}: steps={step}, kills={kills}, HP={last_health:.0f}, Armor={last_armor:.0f}, Dmg={damage_dealt:.0f}")

    game.close()
    return results


# ================================================================
# Standard Benchmark Runner
# ================================================================
def run_benchmark(agent, scenario, episodes, frame_skip=4, realtime=False, armed=False, max_steps=None, visual=False):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {agent.name}")
    print(f"  Scenario: {scenario}")
    print(f"  Episodes: {episodes}")
    if realtime:
        print(f"  Pacing: REAL-TIME (frame_skip={frame_skip} -> {frame_skip/35.0*1000:.0f}ms per decision)")
    else:
        print(f"  Pacing: as-fast-as-possible (headless)")
    if visual:
        print("  Display: VISIBLE window enabled")
    print(f"{'='*60}")

    game = setup_game(scenario, match_visual=realtime, visible=True)
    game.set_episode_timeout(400)  # ~12 seconds max per episode to keep benchmarks fast and consistent
    results = []
    frame_interval = frame_skip / 35.0  # seconds per decision at 35 tics/sec

    for ep in range(episodes):
        game.new_episode()
        if armed:
            arming_sequence(game)
        step = 0
        latencies = []
        action_counts = Counter()
        kills = 0
        last_health = 100
        last_armor = 0
        damage_dealt = 0

        while not game.is_episode_finished() and (max_steps is None or step < max_steps):
            frame_start = time.perf_counter()

            state = game.get_state()
            if state is None:
                break

            screen = state.screen_buffer
            depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

            try:
                last_health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
                last_armor = game.get_game_variable(vizdoom.GameVariable.ARMOR)
                damage_dealt = game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT)
            except:
                pass

            t0 = time.perf_counter()
            action_name, buttons = agent.get_action(screen, depth)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            action_counts[action_name] += 1

            reward = game.make_action(buttons, frame_skip)
            # Count only positive rewards as kills (+1 per kill)
            if reward > 0:
                kills += int(reward)
            step += 1

            # Real-time pacing: sleep to match the same temporal dynamics as visual gameplay
            if realtime:
                elapsed = time.perf_counter() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        results.append({
            'steps': step,
            'kills': kills,
            'health_remaining': max(0, last_health),
            'armor_remaining': max(0, last_armor),
            'damage_dealt': damage_dealt,
            'latencies': latencies,
            'action_counts': dict(action_counts),
        })

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Episode {ep+1}: steps={step}, kills={kills}, HP={last_health:.0f}, Armor={last_armor:.0f}, Dmg={damage_dealt:.0f}")

    game.close()
    return results


def print_comparison(all_results):
    """Print comparison table."""
    print(f"\n{'='*102}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*102}")
    print(f"{'Agent':>25s} | {'Avg Surv':>8s} | {'Max Surv':>8s} | {'Avg Kill':>8s} | {'Tot Kill':>8s} | {'Avg HP':>6s} | {'Avg Armor':>9s} | {'Avg Dmg':>8s} | {'Lat ms':>7s} | {'Entropy':>7s}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")

    for agent_name, metrics in all_results.items():
        print(f"{agent_name:>25s} | {metrics['avg_survival_steps']:>8.1f} | {metrics['max_survival_steps']:>8.0f} | "
              f"{metrics['avg_kills']:>8.1f} | {metrics['total_kills']:>8.0f} | "
              f"{metrics['avg_health_remaining']:>6.1f} | {metrics['avg_armor_remaining']:>9.1f} | "
              f"{metrics['avg_damage_dealt']:>8.1f} | {metrics['avg_latency_ms']:>7.1f} | "
              f"{metrics['action_diversity_entropy']:>7.2f}")

    print(f"{'='*102}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark DOOM agents')
    parser.add_argument('--agent', default='all',
                        choices=['multivec', 'mcts', 'mcts-dpo', 'bon', 'bon-dpo',
                                 'beam', 'beam-dpo', 'gpt4mini', 'gpt5', 'random', 'openrouter', 'all'])
    parser.add_argument('--model', default='models/doom-multivec-trained')
    parser.add_argument('--scenario', default='deathmatch')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--steps', type=int, default=None,
                        help='Max steps per episode')
    parser.add_argument('--armed', action='store_true',
                        help='Start MCTS agent with plasma rifle, armor, and full ammo')
    parser.add_argument('--realtime', action='store_true',
                        help='Enable real-time pacing (sleep between frames like visual gameplay)')
    parser.add_argument('--visual', action='store_true',
                        help='Show the VizDoom game window during benchmarking')
    parser.add_argument('--output', default='benchmark_results.json')
    parser.add_argument('--actor-head',
                        help='Path to DPO-trained actor head (e.g., output/dpo-v1/final)')

    # MCTS-specific arguments
    parser.add_argument('--mcts-simulations', type=int, default=25,
                        help='MCTS: Number of simulations per decision (default: 25)')
    parser.add_argument('--mcts-depth', type=int, default=20,
                        help='MCTS: Rollout depth in frames (default: 20)')
    parser.add_argument('--mcts-exploration', type=float, default=1.414,
                        help='MCTS: UCB1 exploration constant (default: sqrt(2))')
    parser.add_argument('--mcts-batch-size', type=int, default=1,
                        help='MCTS: Batch size for parallel simulations (default: 1)')

    # BoN-specific arguments
    parser.add_argument('--bon-rollouts', type=int, default=25,
                        help='BoN: Number of rollouts per decision (default: 25)')
    parser.add_argument('--bon-depth', type=int, default=20,
                        help='BoN: Rollout depth in frames (default: 20)')
    parser.add_argument('--bon-temperature', type=float, default=0.1,
                        help='BoN: Policy sampling temperature (default: 0.1)')
    parser.add_argument('--bon-llm-eval', action='store_true',
                        help='BoN: Enable LLM-based evaluation')
    parser.add_argument('--bon-llm-api-key', type=str, default=None,
                        help='BoN: API key for LLM eval (defaults to TRITON_API_KEY env var)')
    parser.add_argument('--bon-llm-sample-rate', type=float, default=0.05,
                        help='BoN: Per-rollout probability of LLM call (default: 0.05)')
    parser.add_argument('--bon-llm-blend', type=float, default=0.3,
                        help='BoN: Weight of LLM value in score (default: 0.3)')
    parser.add_argument('--bon-llm-cache-path', type=str, default=None,
                        help='BoN: Path to load/save LLM cache')
    parser.add_argument('--bon-llm-verbose', action='store_true',
                        help='BoN: Print LLM responses')

    # Beam-specific arguments
    parser.add_argument('--beam-width', type=int, default=4,
                        help='Beam: Number of sequences kept (default: 4)')
    parser.add_argument('--beam-depth', type=int, default=8,
                        help='Beam: Lookahead horizon in actions (default: 8)')
    parser.add_argument('--beam-top-k', type=int, default=2,
                        help='Beam: Children expanded per beam item (default: 2)')

    args = parser.parse_args()

    all_results = {}

    openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
    openrouter_url = 'https://openrouter.ai/api/v1'

    agents_to_run = []
    technique_agents = []  # Agents that need run_benchmark_technique

    if args.agent in ('multivec', 'all'):
        agents_to_run.append(('MultiVec', MultiVecAgent(args.model, actor_head_path=args.actor_head)))
    if args.agent in ('mcts', 'all'):
        agent = MCTSAgentBenchmark(
            args.model,
            simulations=args.mcts_simulations,
            depth=args.mcts_depth,
            exploration=args.mcts_exploration,
            batch_size=args.mcts_batch_size
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('mcts-dpo', 'all'):
        if not args.actor_head:
            print("Warning: --mcts-dpo requires --actor-head. Using model path for DPO.")
        agent = MCTSAgentBenchmark(
            args.model,
            actor_head_path=args.actor_head,
            simulations=args.mcts_simulations,
            depth=args.mcts_depth,
            exploration=args.mcts_exploration,
            batch_size=args.mcts_batch_size
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('bon', 'all'):
        agent = BoNAgentBenchmark(
            args.model,
            num_rollouts=args.bon_rollouts,
            rollout_depth=args.bon_depth,
            temperature=args.bon_temperature,
            llm_eval=args.bon_llm_eval,
            llm_api_key=args.bon_llm_api_key,
            llm_sample_rate=args.bon_llm_sample_rate,
            llm_blend=args.bon_llm_blend,
            llm_cache_path=args.bon_llm_cache_path,
            llm_verbose=args.bon_llm_verbose,
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('bon-dpo', 'all'):
        if not args.actor_head:
            print("Warning: --bon-dpo requires --actor-head. Using model path for DPO.")
        agent = BoNAgentBenchmark(
            args.model,
            actor_head_path=args.actor_head,
            num_rollouts=args.bon_rollouts,
            rollout_depth=args.bon_depth,
            temperature=args.bon_temperature,
            llm_eval=args.bon_llm_eval,
            llm_api_key=args.bon_llm_api_key,
            llm_sample_rate=args.bon_llm_sample_rate,
            llm_blend=args.bon_llm_blend,
            llm_cache_path=args.bon_llm_cache_path,
            llm_verbose=args.bon_llm_verbose,
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('beam', 'all'):
        agent = BeamAgentBenchmark(
            args.model,
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
            top_k=args.beam_top_k
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('beam-dpo', 'all'):
        if not args.actor_head:
            print("Warning: --beam-dpo requires --actor-head. Using model path for DPO.")
        agent = BeamAgentBenchmark(
            args.model,
            actor_head_path=args.actor_head,
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
            top_k=args.beam_top_k
        )
        technique_agents.append((agent.name, agent))
    if args.agent in ('random', 'all'):
        agents_to_run.append(('Random', RandomAgent()))
    if args.agent in ('gpt4mini', 'all'):
        agents_to_run.append(('GPT-4o-mini', LLMAgent('gpt-4o-mini')))
    if args.agent in ('gpt5', 'all'):
        agents_to_run.append(('GPT-5', LLMAgent('gpt-5')))
    if args.agent in ('openrouter', 'all'):
        or_models = [
            'qwen/qwen3.5-27b',
            'nvidia/nemotron-3-super-120b-a12b',
            'google/gemini-3.1-flash-lite-preview',
        ]
        for m in or_models:
            agents_to_run.append((m.split('/')[-1], LLMAgent(m, base_url=openrouter_url, api_key=openrouter_key)))

    # Run standard agents
    for name, agent in agents_to_run:
        try:
            results = run_benchmark(
                agent,
                args.scenario,
                args.episodes,
                args.frame_skip,
                args.realtime,
                args.armed,
                args.steps,
                args.visual,
            )
            metrics = compute_metrics(results)
            
            # Attach raw episode data so experiment_utils.py can log individual episodes
            metrics['raw_episodes'] = [
                {
                    'steps': r['steps'],
                    'kills': r['kills'],
                    'health_remaining': r['health_remaining'],
                    'armor_remaining': r['armor_remaining'],
                    'damage_dealt': r['damage_dealt'],
                    'avg_latency': float(np.mean(r['latencies'])) if r['latencies'] else 0.0
                } for r in results
            ]

            all_results[agent.name] = metrics
            print(f"\n  {agent.name}: avg_survival={metrics['avg_survival_steps']:.1f}, "
                  f"avg_kills={metrics['avg_kills']:.1f}, "
                  f"avg_armor={metrics['avg_armor_remaining']:.1f}, "
                  f"avg_damage={metrics['avg_damage_dealt']:.1f}, "
                  f"avg_latency={metrics['avg_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")

    # Run technique agents (MCTS, BoN, Beam)
    for name, agent in technique_agents:
        try:
            results = run_benchmark_technique(
                agent,
                args.scenario,
                args.episodes,
                args.frame_skip,
                args.realtime,
                args.armed,
                args.steps,
            )
            metrics = compute_metrics(results)

            # Attach raw episode data
            metrics['raw_episodes'] = [
                {
                    'steps': r['steps'],
                    'kills': r['kills'],
                    'health_remaining': r['health_remaining'],
                    'armor_remaining': r['armor_remaining'],
                    'damage_dealt': r['damage_dealt'],
                    'avg_latency': float(np.mean(r['latencies'])) if r['latencies'] else 0.0
                } for r in results
            ]

            all_results[agent.name] = metrics
            print(f"\n  {agent.name}: avg_survival={metrics['avg_survival_steps']:.1f}, "
                  f"avg_kills={metrics['avg_kills']:.1f}, "
                  f"avg_armor={metrics['avg_armor_remaining']:.1f}, "
                  f"avg_damage={metrics['avg_damage_dealt']:.1f}, "
                  f"avg_latency={metrics['avg_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")

    print_comparison(all_results)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
