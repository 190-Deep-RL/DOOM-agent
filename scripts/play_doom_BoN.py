"""Play DOOM with Best-of-N policy rollouts (predictive sampling).

At each frame, samples N action sequences from the model policy with
temperature, simulates each forward H frames, and picks the first
action of the sequence(s) with the highest mean score (same reward
function as the MCTS planner).

Usage:
  python scripts/play_doom_BoN.py --model models/doom-multivec-trained --scenario basic
  python scripts/play_doom_BoN.py --model models/doom-multivec-trained --scenario deathmatch --armed \
      --num-rollouts 50 --rollout-depth 20 --temperature 0.7
  python scripts/play_doom_BoN.py --model models/doom-multivec-trained --scenario basic --live
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
from doom_multivec.inference.best_of_n import BestOfNAgent
from transformers import AutoTokenizer


def setup_doom(scenario='basic', visible=True, armed=False):
    """Set up VizDoom with visible window."""
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


def load_model(model_path, device='cpu'):
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


def format_bon_stats(stats: dict, action_names: List[str]) -> List[str]:
    """One line per action: count / mean score."""
    lines = []
    for i, name in enumerate(action_names):
        s = stats.get(i, {'count': 0, 'mean': float('-inf')})
        mean_str = f"{s['mean']:+.2f}" if s['count'] > 0 else "  -- "
        lines.append(f"    {name:24s}: starts={s['count']:3d}  mean={mean_str}")
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
            stats_summary = " ".join(
                f"{agent.action_names[a][:6]}={agent.last_action_stats.get(a, {}).get('count', 0)}"
                for a in range(agent.num_actions)
            )
            print(f"  Step {step:4d} | {action_name:22s} | HP={health:.0f} K={kills:.0f} | "
                  f"{decision_time:.0f}ms | starts: {stats_summary}")

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
        print("  Best-of-N DOOM - Live Mode")
        print("=" * 70)
        print(f"  Step: {step}    Action: {action_name} (idx {action_idx})")
        print(f"  HP={health:.0f}  Armor={armor:.0f}  Kills={kills:.0f}  "
              f"Reward(step)={reward:.0f}  Reward(total)={total_reward:.0f}")
        print(f"  Decision: {decision_time:.0f}ms   Steps/min: {spm:.1f}")
        print()
        print("  Rollout stats (per first-action):")
        for line in format_bon_stats(agent.last_action_stats, action_names):
            print(line)
        print()
        print("  Last 10 actions:")
        for a in action_history[-10:]:
            print(f"    {a}")
        print("=" * 70)

    return step, total_reward, Counter(action_history), enemy_kills, latencies


def main():
    parser = argparse.ArgumentParser(description='Play DOOM with Best-of-N policy rollouts')
    parser.add_argument('--model', default='models/doom-multivec-trained',
                        help='Path to trained model')
    parser.add_argument('--scenario', default='defend_the_center',
                        help='DOOM scenario to play')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to play (standard mode only)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Max steps per episode (live mode)')
    parser.add_argument('--num-rollouts', type=int, default=25,
                        help='Number of rollouts per decision')
    parser.add_argument('--rollout-depth', type=int, default=20,
                        help='Frames per rollout')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Policy sampling temperature (higher = more diverse)')
    parser.add_argument('--prior-temperature', type=float, default=1.0,
                        help='Temperature applied to base model logits before composite expansion')
    parser.add_argument('--no-composite-moves', action='store_true',
                        help='Disable composite (two-button) actions; use base 4 actions only')
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
    model, tokenizer, num_actions = load_model(args.model)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\nStarting DOOM ({args.scenario})...")
    print(f"BoN config: {args.num_rollouts} rollouts x {args.rollout_depth} frames, "
          f"temperature={args.temperature}, prior_temperature={args.prior_temperature}, "
          f"composite_moves={not args.no_composite_moves}")
    if args.live:
        print("Mode: LIVE")
    if args.armed:
        print("Armed mode: Starting with plasma rifle, ammo, and armor")

    game = setup_doom(args.scenario, visible=True, armed=args.armed)
    converter = AsciiConverter(width=40, height=25)

    agent = BestOfNAgent(
        model=model,
        tokenizer=tokenizer,
        converter=converter,
        num_rollouts=args.num_rollouts,
        rollout_depth=args.rollout_depth,
        temperature=args.temperature,
        prior_temperature=args.prior_temperature,
        use_composite_moves=not args.no_composite_moves,
        device='cpu',
        frame_skip=args.frame_skip,
    )
    agent.set_game(game)
    print(f"Actions ({agent.num_actions}): {agent.action_names}")

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

    game.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
