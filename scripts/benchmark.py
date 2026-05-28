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
# DOOM Setup
# ================================================================
def setup_game(scenario='defend_the_center', match_visual=False):
    game = vizdoom.DoomGame()
    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
    }
    game.load_config(scenarios.get(scenario, scenario))
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_window_visible(False)
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

    game.init()
    return game


ACTION_NAMES = ['shoot', 'move_forward', 'turn_left', 'turn_right']
ACTION_BUTTONS = {
    'shoot':        [1, 0, 0, 0],
    'move_forward': [0, 1, 0, 0],
    'turn_left':    [0, 0, 1, 0],
    'turn_right':   [0, 0, 0, 1],
}


# ================================================================
# Agent: MultiVec Classifier
# ================================================================
class MultiVecAgent:
    def __init__(self, model_path):
        import torch
        from doom_multivec.model.classifier import DoomMultiVecClassifier
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
        num_actions = 4
        for key in state:
            if 'classifier.weight' in key:
                num_actions = state[key].shape[0]
                break
        self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
        self.model.load_state_dict(state)
        self.model.eval()
        self.converter = AsciiConverter(width=40, height=25)
        self.name = f"MultiVec-{sum(p.numel() for p in self.model.parameters())/1e6:.1f}M"

    def get_action(self, screen, depth):
        import torch
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

        action_names = ACTION_NAMES[:len(probs)]
        sorted_idx = np.argsort(probs)[::-1]
        top_action = action_names[sorted_idx[0]]
        buttons = list(ACTION_BUTTONS[top_action])

        # Shoot boost — exact same logic as play_doom_visual.py
        shoot_idx = action_names.index('shoot') if 'shoot' in action_names else -1
        if shoot_idx >= 0 and top_action != 'shoot':
            top_prob = probs[sorted_idx[0]]
            shoot_prob = probs[shoot_idx]
            if shoot_prob > top_prob * 0.75:
                shoot_buttons = ACTION_BUTTONS['shoot']
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
                    sec_buttons = ACTION_BUTTONS[second_action]
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
# agent : MCTS agent
# ================================================================
class MCTSBenchmarkAgent :

    def __init__(self, model_path, num_simulations = 25, rollout_depth = 20, exploration_constant = 1.414, use_puct = True, rollout_temperature = 0.1, prior_temperature = 0.1, batch_size = 1, use_llm_eval = False, llm_sampling_rate = 4) :
        
        import torch
        from doom_multivec.model.classifier import DoomMultiVecClassifier
        from transformers import AutoTokenizer

        self.use_llm_eval = use_llm_eval
        self.llm_sampling_rate = llm_sampling_rate

        # load model weights from disk — same pattern as MultiVecAgent
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        state = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
        num_actions = 4

        for key in state :

            if 'classifier.weight' in key :

                num_actions = state[key].shape[0]
                break
        
        self.model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
        self.model.load_state_dict(state)
        self.model.eval()
        self.num_actions = num_actions

        # store MCTS configuration — can't build MCTSAgent yet because the game doesn't exist until setup() is called
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.exploration_constant = exploration_constant
        self.use_puct = use_puct
        self.rollout_temperature = rollout_temperature
        self.prior_temperature = prior_temperature
        self.batch_size = batch_size

        # frame converter — turns raw pixels into ASCII tokens the model can read
        self.converter = AsciiConverter(width = 40, height = 25)

        # placeholders — filled in by setup()
        self.game = None
        self.mcts_agent = None
        self.armed = False

        # stores the action index from the last get_action() call so on_step_end() knows which branch to advance without needing it passed in
        self.last_action_idx = None

        # name shown in the benchmark comparison table
        self.name = f"MCTS-sim{num_simulations}-depth{rollout_depth}-c{exploration_constant}"

    def setup(self, scenario = 'defend_the_center', armed = False) :

        # store armed so on_episode_start() can use it
        self.armed = armed

        # create the game — settings match play_doom_mcts.py exactly so behavior is consistent with what the team has been testing
        game = vizdoom.DoomGame()
        
        scenarios = {

            'basic':             vizdoom.scenarios_path + '/basic.cfg',
            'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
            'deadly_corridor':   vizdoom.scenarios_path + '/deadly_corridor.cfg',
            'deathmatch':        vizdoom.scenarios_path + '/deathmatch.cfg',

        }

        game.load_config(scenarios.get(scenario, scenario))
        game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        game.set_depth_buffer_enabled(True)
        game.set_window_visible(False)  # headless — no window during benchmarking
        game.set_mode(vizdoom.Mode.PLAYER)
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        game.set_render_hud(True)
        game.set_episode_timeout(2100)
        game.clear_available_buttons()
        game.add_available_button(vizdoom.Button.ATTACK)
        game.add_available_button(vizdoom.Button.MOVE_FORWARD)
        game.add_available_button(vizdoom.Button.TURN_LEFT)
        game.add_available_button(vizdoom.Button.TURN_RIGHT)
        game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
        game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
        game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)
        game.add_available_game_variable(vizdoom.GameVariable.ARMOR)
        game.init()
        self.game = game

        # now that the game exists we can build the MCTSAgent and hand it the live game connection it needs
        from doom_multivec.inference.mcts import MCTSAgent as _MCTSAgent

        self.mcts_agent = _MCTSAgent(

            model=self.model,
            tokenizer=self.tokenizer,
            converter=self.converter,
            num_simulations=self.num_simulations,
            rollout_depth=self.rollout_depth,
            exploration_constant=self.exploration_constant,
            num_actions=self.num_actions,
            device='cpu',
            batch_size=self.batch_size,
            use_puct=self.use_puct,
            rollout_temperature=self.rollout_temperature,
            prior_temperature=self.prior_temperature,
            use_composite_moves=True,
            composite_logit_weights=[50.0, 0.7, 0.8, 0.9],
            use_llm_eval = self.use_llm_eval,
            llm_sampling_rate = self.llm_sampling_rate,

        )

        self.mcts_agent.set_game(self.game)

    def on_episode_start(self):
        
        # wipe the tree from the previous episode — old planning is meaningless at the start of a new episode
        self.mcts_agent.reset()

        # if armed mode is on, give the agent plasma rifle + armor + ammo
        # must happen after new_episode() is called, not during setup()
        if self.armed :

            self._arming_sequence()

    def _arming_sequence(self) :

        # copied exactly from play_doom_mcts.py so behavior is identical
        for _ in range(5) :

            self.game.advance_action(1)
        
        self.game.send_game_command("give Backpack")

        for _ in range(3) :

            self.game.advance_action(1)
        
        self.game.send_game_command("give PlasmaRifle")

        for _ in range(6) :

            self.game.send_game_command("give CellPack")
            self.game.advance_action(1)
        
        self.game.send_game_command("give BlueArmor")
        self.game.send_game_command("use PlasmaRifle")
        self.game.advance_action(1)

    def get_action(self, screen, depth) :

        # screen and depth are ignored —
        # MCTS reads directly from the live game connection it already has
        # get_action() returns four values — we discard the fourth (benchmark_times)
        action_name, buttons, action_idx, _ = self.mcts_agent.get_action()

        # store action_idx so on_step_end() knows which branch was taken
        self.last_action_idx = action_idx

        # return only (action_name, buttons) — matching the benchmark's expected interface
        return action_name, buttons

    def on_step_end(self) :

        # tell the tree which branch actually got taken so it can reuse that subtree next frame instead of starting fresh
        if self.last_action_idx is not None :

            self.mcts_agent.advance_root(self.last_action_idx)


# ================================================================
# Benchmark Runner
# ================================================================
def run_benchmark(agent, scenario, episodes, frame_skip=4, realtime=False):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {agent.name}")
    print(f"  Scenario: {scenario}")
    print(f"  Episodes: {episodes}")
    if realtime:
        print(f"  Pacing: REAL-TIME (frame_skip={frame_skip} -> {frame_skip/35.0*1000:.0f}ms per decision)")
    else:
        print(f"  Pacing: as-fast-as-possible (headless)")
    print(f"{'='*60}")

    game = setup_game(scenario, match_visual=realtime)
    results = []
    frame_interval = frame_skip / 35.0  # seconds per decision at 35 tics/sec

    for ep in range(episodes):
        game.new_episode()
        step = 0
        latencies = []
        action_counts = Counter()
        kills = 0
        last_health = 100

        while not game.is_episode_finished():
            frame_start = time.perf_counter()

            state = game.get_state()
            if state is None:
                break

            screen = state.screen_buffer
            depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

            try:
                last_health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
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
            'latencies': latencies,
            'action_counts': dict(action_counts),
        })

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Episode {ep+1}: steps={step}, kills={kills}, HP={last_health:.0f}")

    game.close()
    return results

# ================================================================
# benchmark runner : MCTS
# ================================================================
def run_benchmark_mcts(agent, scenario, episodes, frame_skip = 4, armed = False) :

    print(f"\n{'='*60}")
    print(f"  benchmarking : {agent.name}")
    print(f"  scenario : {scenario}")
    print(f"  episodes : {episodes}")
    print(f"  pacing : as-fast-as-possible (headless)")
    print(f"{'='*60}")

    # setup() creates the game and builds the MCTSAgent internally
    agent.setup(scenario=scenario, armed=armed)
    game = agent.game
    results = []

    for ep in range(episodes) :

        game.new_episode()

        # hook 1 : reset the MCTS tree at the start of each episode
        agent.on_episode_start()
        step = 0
        latencies = []
        action_counts = Counter()
        kills = 0
        last_health = 100

        while not game.is_episode_finished() :

            state = game.get_state()
            
            if state is None :

                break

            screen = state.screen_buffer
            depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

            try :

                last_health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
            except :

                pass

            # ask the agent for an action — MCTS runs its simulations here
            t0 = time.perf_counter()
            action_name, buttons = agent.get_action(screen, depth)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            action_counts[action_name] += 1

            # execute the action in the game
            reward = game.make_action(buttons, frame_skip)

            # hook 2 : update the tree now that we know what actually happened
            agent.on_step_end()

            if reward > 0 :

                kills += int(reward)

            step += 1

        results.append({

            'steps': step,
            'kills': kills,
            'health_remaining': max(0, last_health),
            'latencies': latencies,
            'action_counts': dict(action_counts),

        })

        if (ep + 1) % 5 == 0 or ep == 0 :

            print(f"  episode {ep+1} : steps = {step}, kills = {kills}, HP = {last_health:.0f}")

    game.close()
    return results


def print_comparison(all_results) :

    """Print comparison table."""
    
    print(f"\n{'='*115}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*115}")
    print(f"{'Agent':>28s} | {'Avg Surv ± CI':>15s} | {'Max Surv':>8s} | {'Avg Kill ± CI':>14s} | {'Tot Kill':>8s} | {'Avg HP ± CI':>13s} | {'Lat ms':>7s} | {'Entropy':>7s}")
    print(f"{'-'*28}-+-{'-'*15}-+-{'-'*8}-+-{'-'*14}-+-{'-'*8}-+-{'-'*13}-+-{'-'*7}-+-{'-'*7}")

    for agent_name, metrics in all_results.items() :

        surv_ci = metrics.get('survival_ci_95', 0)
        kill_ci = metrics.get('kills_ci_95', 0)
        hp_ci   = metrics.get('health_ci_95', 0)
        surv_str = f"{metrics['avg_survival_steps']:.1f} ± {surv_ci:.1f}"
        kill_str = f"{metrics['avg_kills']:.1f} ± {kill_ci:.2f}"
        hp_str   = f"{metrics['avg_health_remaining']:.1f} ± {hp_ci:.1f}"

        print(f"{agent_name:>28s} | "
              f"{surv_str:>15s} | "
              f"{metrics['max_survival_steps']:>8.0f} | "
              f"{kill_str:>14s} | "
              f"{metrics['total_kills']:>8.0f} | "
              f"{hp_str:>13s} | "
              f"{metrics['avg_latency_ms']:>7.1f} | "
              f"{metrics['action_diversity_entropy']:>7.2f}")

    print(f"{'='*115}\n")

# ================================================================
# metrics
# ================================================================
def compute_metrics(episodes) :

    if not episodes :

        return {}

    n = len(episodes)
    survival = [ep['steps'] for ep in episodes]
    kills = [ep['kills'] for ep in episodes]
    health = [ep['health_remaining'] for ep in episodes]
    latencies = []

    for ep in episodes :

        latencies.extend(ep['latencies'])
    
    all_actions = Counter()

    for ep in episodes :

        all_actions.update(ep['action_counts'])

    total_actions = sum(all_actions.values())

    if total_actions > 0 :
        
        probs = np.array([all_actions[a] / total_actions for a in all_actions])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else :

        entropy = 0.0

    # 95% confidence interval
    def ci95(data) :

        if len(data) < 2 :

            return 0.0
        
        return 1.96 * np.std(data, ddof=1) / np.sqrt(len(data))

    return {

        'episodes': n,
        'avg_survival_steps': np.mean(survival),
        'survival_ci_95': ci95(survival),
        'max_survival_steps': max(survival),
        'avg_kills': np.mean(kills),
        'kills_ci_95': ci95(kills),
        'total_kills': sum(kills),
        'avg_health_remaining': np.mean(health),
        'health_ci_95': ci95(health),
        'avg_latency_ms': np.mean(latencies) if latencies else 0,
        'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
        'action_diversity_entropy': entropy,
        'action_distribution': dict(all_actions.most_common()),

    }

# ================================================================
# MCTS ablation study
# ================================================================
def run_mcts_ablation(model_path, scenario, episodes, frame_skip = 4, armed = False) :

    # each config varies one parameter while holding the others fixed —
    # this isolates what's actually driving performance differences
    configs = [

        # baseline
        {'num_simulations': 25, 'rollout_depth': 20, 'use_puct': True},

        # vary simulations
        {'num_simulations': 10, 'rollout_depth': 20, 'use_puct': True},
        {'num_simulations': 50, 'rollout_depth': 20, 'use_puct': True},

        # vary depth
        {'num_simulations': 25, 'rollout_depth': 10, 'use_puct': True},
        {'num_simulations': 25, 'rollout_depth': 30, 'use_puct': True},

        # UCB1 vs PUCT — clean binary comparison
        {'num_simulations': 25, 'rollout_depth': 20, 'use_puct': False},

    ]

    all_results = {}

    for config in configs :

        agent = MCTSBenchmarkAgent(model_path, **config)
        print(f"\n  running config : {agent.name}")

        try :

            results = run_benchmark_mcts(agent, scenario, episodes, frame_skip, armed)
            all_results[agent.name] = compute_metrics(results)
        except Exception as e :

            print(f"  config FAILED : {e}")

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Benchmark DOOM agents')
    parser.add_argument('--agent', default='all',
                        choices=['multivec', 'gpt4mini', 'gpt5', 'random',
                                 'openrouter', 'mcts', 'ablation', 'all'])
    parser.add_argument('--model', default='models/doom-multivec-trained')
    parser.add_argument('--scenario', default='defend_the_center')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=25,
                        help='MCTS simulations per frame (mcts/ablation only)')
    parser.add_argument('--depth', type=int, default=20,
                        help='MCTS rollout depth (mcts/ablation only)')
    parser.add_argument('--armed', action='store_true',
                        help='Start MCTS agent with plasma rifle, armor, and full ammo')
    parser.add_argument('--realtime', action='store_true',
                        help='Enable real-time pacing (non-MCTS agents only)')
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()

    all_results = {}

    # --- ablation: run all MCTS configs automatically ---
    if args.agent == 'ablation':
        all_results = run_mcts_ablation(
            args.model, args.scenario, args.episodes,
            args.frame_skip, args.armed
        )

    # --- single MCTS run ---
    elif args.agent == 'mcts':
        agent = MCTSBenchmarkAgent(
            args.model,
            num_simulations=args.simulations,
            rollout_depth=args.depth,
        )
        try:
            results = run_benchmark_mcts(
                agent, args.scenario, args.episodes,
                args.frame_skip, args.armed
            )
            all_results[agent.name] = compute_metrics(results)
        except Exception as e:
            print(f"\n  MCTS FAILED: {e}")

    # --- non-MCTS agents ---
    else:
        openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
        openrouter_url = 'https://openrouter.ai/api/v1'

        agents_to_run = []
        if args.agent in ('multivec', 'all'):
            agents_to_run.append(('MultiVec', MultiVecAgent(args.model)))
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

        for name, agent in agents_to_run:
            try:
                results = run_benchmark(agent, args.scenario, args.episodes, args.frame_skip, args.realtime)
                metrics = compute_metrics(results)
                all_results[agent.name] = metrics
                print(f"\n  {agent.name}: avg_survival={metrics['avg_survival_steps']:.1f}, "
                      f"avg_kills={metrics['avg_kills']:.1f} ± {metrics['kills_ci_95']:.2f}, "
                      f"avg_latency={metrics['avg_latency_ms']:.1f}ms")
            except Exception as e:
                print(f"\n  {name} FAILED: {e}")

    # common output for all agent types
    print_comparison(all_results)

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
