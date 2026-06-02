"""Play DOOM with beam search over action sequences.

At each frame, maintains a beam of B partial action sequences. At each
depth d, expands each item by its top-k policy actions, simulates one
step, scores by cumulative reward delta + prior_weight * log-policy,
keeps top-B. Returns the first action of the highest-scoring beam.

Usage:
  python scripts/play_doom_beam.py --model models/doom-multivec-trained --scenario basic
  python scripts/play_doom_beam.py --model models/doom-multivec-trained --scenario deathmatch --armed \
      --beam-width 4 --beam-depth 8 --top-k 2 --prior-weight 0.5
  python scripts/play_doom_beam.py --model models/doom-multivec-trained --scenario basic --live
"""

import argparse
import os
import sys
import time
from collections import Counter
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.model.classifier import DoomMultiVecClassifier
from doom_multivec.ascii.converter import AsciiConverter
from doom_multivec.inference.beam_search import BeamSearchAgent
from doom_multivec.inference.llm_value_cache import LLMValueCache
from pathlib import Path
from transformers import AutoTokenizer


def maybe_build_llm_cache(args) -> LLMValueCache:
    """Return an LLMValueCache iff --llm-eval is set, otherwise None."""
    if not args.llm_eval:
        return None
    api_key = args.llm_api_key or os.environ.get('TRITON_API_KEY')
    if api_key is None:
        print("WARNING: --llm-eval set but no API key found "
              "(--llm-api-key or TRITON_API_KEY env). Leaf eval will be skipped.")
    rubric_text = ""
    if args.llm_rubric:
        rubric_text = Path(args.llm_rubric).read_text()
    else:
        default_rubric = Path(__file__).resolve().parent.parent / 'src' / 'doom_multivec' / 'inference' / 'rubric.txt'
        if default_rubric.exists():
            rubric_text = default_rubric.read_text()
    cache = LLMValueCache(
        api_key=api_key,
        prompt=rubric_text,
        sample_rate=0.0,  # beam doesn't use sample_rate; it forces leaf calls
        ema_alpha=args.llm_ema_alpha,
        credit_decay=args.llm_credit_decay,
        verbose=args.llm_verbose,
    )
    if args.llm_cache_path and Path(args.llm_cache_path).exists():
        cache.load(args.llm_cache_path)
        print(f"Loaded LLM value cache from {args.llm_cache_path} ({len(cache)} entries)")
    return cache


def setup_doom(scenario='basic', visible=True, armed=False):
    game = vizdoom.DoomGame()

    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'health_gathering': vizdoom.scenarios_path + '/health_gathering.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
        'deathmatch': vizdoom.scenarios_path + '/deathmatch.cfg',
    }
    cfg_path = scenarios.get(scenario, scenario)
    game.load_config(cfg_path)

    game.set_window_visible(visible)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_depth_buffer_enabled(True)

    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)
    game.add_available_button(vizdoom.Button.TURN_LEFT)
    game.add_available_button(vizdoom.Button.TURN_RIGHT)

    game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vizdoom.GameVariable.ARMOR)

    game.set_episode_timeout(2100)
    game.set_mode(vizdoom.Mode.PLAYER)

    game.init()
    return game


def load_model(model_path, device='cpu', actor_head_path=None):
    """Load the trained classifier or DPO policy. Auto-detects num_actions from saved weights.

    Args:
        model_path: Base encoder model path.
        device: Device to load model on.
        actor_head_path: Optional path to DPO-trained actor head.
    """
    from doom_multivec.model.dpo_policy import DPODoomPolicy

    # Check if loading DPO policy (actor head)
    if actor_head_path or os.path.exists(os.path.join(model_path, 'actor_head.pt')):
        # Load DPO policy
        dpo_path = actor_head_path if actor_head_path else model_path
        print(f"Loading DPO policy from {dpo_path}")
        model = DPODoomPolicy.from_pretrained(dpo_path, encoder_path=model_path)
        model.eval()
        model.to(device)

        # Load tokenizer from encoder path
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Determine num_actions from model
        num_actions = model.num_actions
        return model, tokenizer, num_actions

    # Original classifier loading logic
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    state = torch.load(os.path.join(model_path, 'model.pt'), map_location=device)
    num_actions = 4
    for key in state:
        if 'classifier.weight' in key:
            num_actions = state[key].shape[0]
            break

    model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, tokenizer, num_actions


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


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


def format_top_beams(beams, action_names) -> List[str]:
    lines = []
    for i, item in enumerate(beams):
        seq = " > ".join(action_names[a][:6] for a in item.actions)
        lines.append(f"    #{i+1}  score={item.score:+.2f}  logp={item.log_prob:+.2f}  [{seq}]")
    return lines


def run_episode_standard(args, game, agent, num_actions, episode):
    step = 0
    total_reward = 0.0
    action_counter = Counter()
    latencies = []
    enemy_kills = Counter()
    kills_start = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)

    frame_interval = args.frame_skip / 35.0

    while not game.is_episode_finished():
        frame_start = time.perf_counter()

        if game.get_state() is None:
            break

        t0 = time.perf_counter()
        action_name, buttons, action_idx, _bench = agent.get_action()
        decision_time = (time.perf_counter() - t0) * 1000
        latencies.append(decision_time)

        reward = game.make_action(buttons, args.frame_skip)
        total_reward += reward
        if reward > 0:
            enemy_kills[reward] += 1

        action_counter[action_name] += 1
        step += 1

        if step % 10 == 1:
            health = game.get_game_variable(vizdoom.GameVariable.HEALTH) if not game.is_episode_finished() else 0
            kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT) if not game.is_episode_finished() else 0
            top = agent.last_top_beams[0] if agent.last_top_beams else None
            top_str = f"best={top.score:+.2f}" if top else "best=--"
            print(f"  Step {step:4d} | {action_name:22s} | HP={health:.0f} K={kills:.0f} | "
                  f"{decision_time:.0f}ms | {top_str}")

        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    return step, total_reward, action_counter, enemy_kills, latencies, kills_start


def run_episode_live(args, game, agent, num_actions):
    step = 0
    total_reward = 0.0
    action_history: List[str] = []
    latencies = []
    enemy_kills = Counter()
    action_names = agent.action_names
    frame_times = []

    while not game.is_episode_finished():
        if args.steps is not None and step >= args.steps:
            print(f"\nReached max steps ({args.steps})")
            break

        step_start = time.perf_counter()

        if game.get_state() is None:
            break

        health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
        armor = game.get_game_variable(vizdoom.GameVariable.ARMOR)
        kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)

        t0 = time.perf_counter()
        action_name, buttons, action_idx, _bench = agent.get_action()
        decision_time = (time.perf_counter() - t0) * 1000
        latencies.append(decision_time)

        reward = game.make_action(buttons, args.frame_skip)
        total_reward += reward
        if reward > 0:
            enemy_kills[reward] += 1

        action_history.append(action_name)
        step += 1

        frame_time = time.perf_counter() - step_start
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        spm = 60.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        clear_screen()
        print("=" * 70)
        print("  Beam Search DOOM - Live Mode")
        print("=" * 70)
        print(f"  Step: {step}    Action: {action_name} (idx {action_idx})")
        print(f"  HP={health:.0f}  Armor={armor:.0f}  Kills={kills:.0f}  "
              f"Reward(step)={reward:.0f}  Reward(total)={total_reward:.0f}")
        print(f"  Decision: {decision_time:.0f}ms   Steps/min: {spm:.1f}")
        print()
        print(f"  Top beams (depth={args.beam_depth}, width={args.beam_width}, k={args.top_k}):")
        for line in format_top_beams(agent.last_top_beams, action_names):
            print(line)
        print()
        print("  Last 10 actions:")
        for a in action_history[-10:]:
            print(f"    {a}")
        print("=" * 70)

    return step, total_reward, Counter(action_history), enemy_kills, latencies


def main():
    parser = argparse.ArgumentParser(description='Play DOOM with beam search')
    parser.add_argument('--model', default='models/doom-multivec-trained',
                        help='Path to trained model')
    parser.add_argument('--actor-head',
                        help='Path to DPO-trained actor head (e.g., output/dpo-v1/final)')
    parser.add_argument('--scenario', default='defend_the_center',
                        help='DOOM scenario to play')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to play (standard mode only)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Max steps per episode (live mode)')
    parser.add_argument('--beam-width', type=int, default=4,
                        help='Number of sequences kept after each depth step')
    parser.add_argument('--beam-depth', type=int, default=8,
                        help='Lookahead horizon in actions')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Children expanded per beam item per step')
    parser.add_argument('--prior-weight', type=float, default=0.5,
                        help='Coefficient on cumulative log-policy in the rank score')
    parser.add_argument('--prior-temperature', type=float, default=1.0,
                        help='Temperature applied to base model logits before composite expansion')
    parser.add_argument('--no-composite-moves', action='store_true',
                        help='Disable composite (two-button) actions; use base 4 actions only')
    # ---- LLM-value-cache flags ----
    parser.add_argument('--llm-eval', action='store_true',
                        help='Enable LLM-based value caching (leaf eval + cached step blending)')
    parser.add_argument('--llm-api-key', default=None,
                        help='Triton API key (falls back to $TRITON_API_KEY)')
    parser.add_argument('--no-llm-leaf-eval', action='store_true',
                        help='Skip the per-decision LLM call on surviving beam tips (only use cached lookups)')
    parser.add_argument('--llm-blend', type=float, default=0.3,
                        help='Weight of per-beam LLM rating in the final reranker')
    parser.add_argument('--llm-cache-blend', type=float, default=0.15,
                        help='Weight of cached LLM values mixed into intermediate step deltas')
    parser.add_argument('--llm-ema-alpha', type=float, default=0.3,
                        help='EMA step size for cache updates')
    parser.add_argument('--llm-credit-decay', type=float, default=0.95,
                        help='Per-step decay when broadcasting LLM rating backwards along a beam path')
    parser.add_argument('--llm-rubric', default=None,
                        help='Path to a rubric file (defaults to MCTS rubric.txt)')
    parser.add_argument('--llm-cache-path', default=None,
                        help='Load/save cache JSON at this path (persists across episodes)')
    parser.add_argument('--llm-verbose', action='store_true',
                        help='Print parsed LLM responses')
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Frames between decisions')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target display FPS (standard mode only)')
    parser.add_argument('--armed', action='store_true',
                        help='Start with plasma rifle, armor, and full ammo')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--live', action='store_true',
                        help='Live mode: continuous display with steps/minute')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("Loading model...")
    model, tokenizer, num_actions = load_model(args.model, actor_head_path=args.actor_head)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\nStarting DOOM ({args.scenario})...")
    print(f"Beam config: width={args.beam_width}, depth={args.beam_depth}, "
          f"top_k={args.top_k}, prior_weight={args.prior_weight}, "
          f"prior_temperature={args.prior_temperature}, "
          f"composite_moves={not args.no_composite_moves}")
    if args.live:
        print("Mode: LIVE")
    if args.armed:
        print("Armed mode: Starting with plasma rifle, ammo, and armor")

    game = setup_doom(args.scenario, visible=True, armed=args.armed)
    converter = AsciiConverter(width=40, height=25)

    llm_cache = maybe_build_llm_cache(args)

    agent = BeamSearchAgent(
        model=model,
        tokenizer=tokenizer,
        converter=converter,
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
        top_k=args.top_k,
        prior_weight=args.prior_weight,
        prior_temperature=args.prior_temperature,
        use_composite_moves=not args.no_composite_moves,
        device='cpu',
        frame_skip=args.frame_skip,
        llm_cache=llm_cache,
        llm_leaf_eval=not args.no_llm_leaf_eval,
        llm_blend=args.llm_blend,
        llm_cache_blend=args.llm_cache_blend,
    )
    agent.set_game(game)
    print(f"Actions ({agent.num_actions}): {agent.action_names}")
    if llm_cache is not None:
        print(f"LLM eval: leaf_eval={not args.no_llm_leaf_eval}, blend={args.llm_blend}, "
              f"cache_blend={args.llm_cache_blend}, cache_size={len(llm_cache)}")

    if args.live:
        game.new_episode()
        agent.reset()
        if args.armed:
            arming_sequence(game)

        step, total_reward, action_counter, enemy_kills, latencies = run_episode_live(
            args, game, agent, num_actions
        )

        print("\n" + "=" * 70)
        print("  FINAL SUMMARY")
        print("=" * 70)
        print(f"  Total steps: {step}")
        print(f"  Total reward: {total_reward:.0f}")
        print(f"  Final kills: {game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)}")
        print(f"  Final health: {game.get_game_variable(vizdoom.GameVariable.HEALTH)}")
        print(f"  Final armor: {game.get_game_variable(vizdoom.GameVariable.ARMOR)}")
        if latencies:
            print(f"  Avg decision time: {np.mean(latencies):.0f}ms")
        print()
        print("  All Actions:")
        for action, count in action_counter.most_common():
            pct = count / step * 100 if step > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {action:24s}: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        for episode in range(args.episodes):
            game.new_episode()
            agent.reset()
            if args.armed:
                arming_sequence(game)

            print(f"\n{'='*60}\nEpisode {episode + 1}/{args.episodes}\n{'='*60}")
            step, total_reward, action_counter, enemy_kills, latencies, kills_start = (
                run_episode_standard(args, game, agent, num_actions, episode)
            )

            kills_end = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
            kills_this_episode = int(kills_end) - int(kills_start)

            print(f"\n  --- Episode {episode + 1} Summary ---")
            print(f"  Steps: {step}")
            print(f"  Total reward: {total_reward:.0f}")
            print(f"  Kills: {kills_this_episode}")
            if latencies:
                print(f"  Avg decision time: {np.mean(latencies):.0f}ms")

            if enemy_kills:
                print(f"  Enemies killed:")
                for reward_amt in sorted(enemy_kills.keys()):
                    count = enemy_kills[reward_amt]
                    print(f"    Enemy (reward={int(reward_amt)}): {count} kills")

            print(f"  Actions:")
            for action, count in action_counter.most_common():
                pct = count / step * 100 if step > 0 else 0
                bar = '#' * int(pct / 2)
                print(f"    {action:24s}: {count:4d} ({pct:5.1f}%) {bar}")

    if llm_cache is not None:
        print(f"\nLLM cache stats: {llm_cache.stats}  (size={len(llm_cache)})")
        if args.llm_cache_path:
            llm_cache.save(args.llm_cache_path)
            print(f"Saved LLM value cache to {args.llm_cache_path}")

    game.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
