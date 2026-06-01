"""Generate DOOM episodes for rubric scoring and DPO training.

This script runs the current model to generate gameplay trajectories,
which can then be scored by the LLM rubric for DPO training.

Usage:
    # Generate with current model
    python scripts/generate_episodes.py --model models/doom-multivec-5L --num-episodes 50

    # Generate with specific model variant
    python scripts/generate_episodes.py --model output/classifier-best --num-episodes 100 --max-frames 500

    # Save with descriptive name
    python scripts/generate_episodes.py --output data/episodes_baseline_50.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doom_multivec.model import DoomMultiVecClassifier
from doom_multivec.doom.engine import DoomEngine
from doom_multivec.ascii.converter import AsciiConverter
from doom_multivec.training.action_mapping import BASE_ACTIONS, GAMANGEN_ACTIONS
from transformers import AutoTokenizer


def arming_sequence(game):
    """Give starting weapons and armor."""
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


def generate_episode(
    model,
    tokenizer,
    engine,
    converter,
    device,
    max_frames: int = 1000,
    temperature: float = 1.0,
    use_depth: bool = False,
    armed: bool = True,
) -> dict:
    """Generate a single episode.

    Args:
        model: The classifier model.
        tokenizer: Tokenizer for ASCII frames.
        engine: DoomEngine instance (or game object with new_episode/get_state/make_action).
        converter: AsciiConverter instance.
        device: Device for model inference.
        max_frames: Maximum frames per episode.
        temperature: Sampling temperature.
        use_depth: Whether to use depth information.
        armed: Whether to arm the agent with weapons/armor.

    Returns:
        Episode dict with frames, actions, kills, and scores.
    """
    frames = []
    actions = []
    rewards = []
    total_reward = 0

    # Get the underlying game object (DoomEngine wraps it, or use directly)
    if hasattr(engine, 'game'):
        game = engine.game
    else:
        game = engine

    # Start new episode
    game.new_episode()

    # Apply arming sequence if enabled
    if armed:
        arming_sequence(game)

    done = False

    for _ in range(max_frames):
        if done or game.is_episode_finished():
            break

        # Get state
        state = game.get_state()
        if state is None:
            break

        # Extract frame and depth
        screen = state.screen_buffer
        depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

        # Convert to ASCII
        ascii_text = converter.convert_simple(screen)

        # Tokenize
        encoded = tokenizer(
            ascii_text,
            max_length=1100,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Depth bins if available
        depth_ids = None
        if use_depth and depth is not None:
            # Quantize depth to bins
            depth_bins = (depth / 16).astype(np.int32).clip(0, 15)
            # Pad to match sequence length
            depth_ids = torch.zeros_like(input_ids)
            depth_ids[0, :len(depth_bins)] = torch.tensor(depth_bins, dtype=torch.long)
            depth_ids = depth_ids.to(device)

        # Model inference
        with torch.no_grad():
            if hasattr(model, 'sample_action'):
                # DPOPolicy
                action_idx, probs = model.sample_action(
                    input_ids[0], attention_mask[0], depth_ids[0] if depth_ids is not None else None,
                    temperature=temperature
                )
                action_idx = action_idx.item()
            else:
                # Classifier
                result = model(input_ids, attention_mask, depth_ids=depth_ids)
                logits = result['logits']

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    action_idx = torch.multinomial(probs[0], 1).item()
                else:
                    action_idx = logits.argmax(dim=-1).item()

        # Take action in environment
        action_name = BASE_ACTIONS[action_idx]

        # Convert action name to button vector
        n_buttons = game.get_available_buttons_size()
        buttons = [0] * n_buttons
        # Action mapping to button indices (simplified)
        button_map = {
            'shoot': 6,  # ATTACK
            'move_forward': 0,  # MOVE_FORWARD
            'turn_left': 2,  # TURN_LEFT
            'turn_right': 3,  # TURN_RIGHT
            'strafe_left': 4,  # MOVE_LEFT
            'strafe_right': 5,  # MOVE_RIGHT
        }
        if action_name in button_map:
            btn_idx = button_map[action_name]
            if btn_idx < n_buttons:
                buttons[btn_idx] = 1

        reward = game.make_action(buttons, 4)  # frame_skip=4
        done = game.is_episode_finished()

        # Store frame data
        frame_data = {
            'text': ascii_text,
            'action_taken': action_name,
            'action_idx': action_idx,
            'reward': float(reward),
        }

        if depth is not None:
            frame_data['depth_bins'] = depth.tolist() if isinstance(depth, np.ndarray) else depth

        # Store tokenized version for efficiency
        frame_data['input_ids'] = input_ids[0].cpu().tolist()
        frame_data['attention_mask'] = attention_mask[0].cpu().tolist()
        if depth_ids is not None:
            frame_data['depth_ids'] = depth_ids[0].cpu().tolist()

        frames.append(frame_data)
        actions.append(action_idx)
        rewards.append(frame_data['reward'])
        total_reward += frame_data['reward']

    episode = {
        'frames': frames,
        'actions': actions,
        'rewards': rewards,
        'total_reward': total_reward,
        'num_frames': len(frames),
        'kills': int(total_reward),  # In DOOM, reward = kills
        'finished': done or game.is_episode_finished(),
    }

    return episode


def main():
    parser = argparse.ArgumentParser(description="Generate DOOM episodes for DPO training")

    # Model arguments
    parser.add_argument(
        '--model', '-m',
        default='models/doom-multivec-5L',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--pool',
        default='attention',
        choices=['attention', 'mean', 'cls', 'token_vote', 'multi_proto_attn'],
        help='Pooling mode for classifier',
    )

    # Generation arguments
    parser.add_argument(
        '--num-episodes', '-n',
        type=int,
        default=50,
        help='Number of episodes to generate',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=1000,
        help='Maximum frames per episode',
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Sampling temperature (0 = greedy)',
    )

    # Environment arguments
    parser.add_argument(
        '--scenario',
        default='deathmatch',
        help='VizDoom scenario',
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=4,
        help='Frame skip for action repeat',
    )
    parser.add_argument(
        '--use-depth',
        action='store_true',
        help='Use depth buffer',
    )
    parser.add_argument(
        '--armed',
        action='store_true',
        default=True,
        help='Start with plasma rifle, armor, and full ammo (default: True)',
    )
    parser.add_argument(
        '--no-armed',
        dest='armed',
        action='store_false',
        help='Disable arming sequence',
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        default='data/episodes_generated.json',
        help='Output JSON file',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save checkpoint every N episodes',
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec - Episode Generation")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load model
    print(f"Loading model from {args.model}...")
    model_path = args.model

    # Check if it's a DPO policy or classifier
    if os.path.exists(os.path.join(model_path, 'actor_head.pt')):
        from doom_multivec.model import DPODoomPolicy
        model = DPODoomPolicy.from_pretrained(model_path)
        print("  Loaded DPO policy")
    else:
        # Load classifier
        import glob
        pt_files = glob.glob(os.path.join(model_path, '*.pt'))

        # Try to auto-detect num_actions from saved state
        num_actions = 6
        if pt_files:
            try:
                state = torch.load(pt_files[0], map_location='cpu')
                if 'classifier.weight' in state:
                    num_actions = state['classifier.weight'].shape[0]
                elif 'action_mlps.0.0.weight' in state:
                    num_actions = len([k for k in state.keys() if 'action_mlps' in k and '.0.weight' in k])
            except:
                pass

        model = DoomMultiVecClassifier(
            model_path,
            pool_mode=args.pool,
            num_actions=num_actions,
        )

        # Load weights if available
        if pt_files:
            state = torch.load(pt_files[0], map_location='cpu')
            model.load_state_dict(state, strict=False)
            print(f"  Loaded classifier with {num_actions} actions")

    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Initialize VizDoom
    print(f"\nInitializing VizDoom ({args.scenario})...")
    try:
        engine = DoomEngine(
            scenario=args.scenario,
            frame_skip=args.skip_frames,
        )
        engine.init()  # Initialize the game
        print("  VizDoom initialized successfully")
    except Exception as e:
        print(f"  Error initializing VizDoom: {e}")
        print("  Falling back to mock environment for testing...")
        engine = None

    # Create ASCII converter
    converter = AsciiConverter(width=40, height=25)

    # Generate episodes
    print(f"\nGenerating {args.num_episodes} episodes...")
    print(f"  Max frames per episode: {args.max_frames}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Armed: {args.armed}")
    print()

    episodes = []
    all_kills = []
    all_frames = []

    for ep_num in tqdm(range(args.num_episodes), desc="Episodes"):
        if engine is not None:
            episode = generate_episode(
                model, tokenizer, engine, converter, device,
                max_frames=args.max_frames,
                temperature=args.temperature,
                use_depth=args.use_depth,
                armed=args.armed,
            )
        else:
            # Mock episode for testing without VizDoom
            episode = {
                'frames': [
                    {
                        'text': 'Mock frame ' + str(i),
                        'action_taken': BASE_ACTIONS[i % 6],
                        'action_idx': i % 6,
                        'reward': 0,
                        'input_ids': [2] + [i % 128 for i in range(100)],
                        'attention_mask': [1] * 101,
                    }
                    for i in range(10)
                ],
                'actions': [i % 6 for i in range(10)],
                'rewards': [0] * 10,
                'total_reward': 0,
                'num_frames': 10,
                'kills': 0,
                'finished': True,
                'mock': True,
            }

        episodes.append(episode)
        all_kills.append(episode['kills'])
        all_frames.append(episode['num_frames'])

        # Periodic save
        if (ep_num + 1) % args.save_every == 0:
            temp_output = args.output.replace('.json', f'_temp_{ep_num+1}.json')
            with open(temp_output, 'w') as f:
                json.dump({
                    'episodes': episodes,
                    'metadata': {
                        'model': args.model,
                        'num_episodes': len(episodes),
                        'temperature': args.temperature,
                        'avg_kills': sum(all_kills) / len(all_kills) if all_kills else 0,
                    }
                }, f)
            print(f"  Checkpoint saved: {temp_output}")

    # Close environment
    if engine is not None:
        engine.close()

    # Statistics
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"{'=' * 60}")
    print(f"Episodes: {len(episodes)}")
    print(f"Total kills: {sum(all_kills)}")
    print(f"Avg kills per episode: {sum(all_kills) / len(all_kills):.2f}")
    print(f"Min kills: {min(all_kills) if all_kills else 0}")
    print(f"Max kills: {max(all_kills) if all_kills else 0}")
    print(f"Avg frames per episode: {sum(all_frames) / len(all_frames):.1f}")

    # Save final output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    output_data = {
        'episodes': episodes,
        'metadata': {
            'model': args.model,
            'pool_mode': args.pool,
            'num_episodes': len(episodes),
            'temperature': args.temperature,
            'max_frames': args.max_frames,
            'scenario': args.scenario,
            'skip_frames': args.skip_frames,
            'armed': args.armed,
            'seed': args.seed,
            'statistics': {
                'avg_kills': sum(all_kills) / len(all_kills) if all_kills else 0,
                'min_kills': min(all_kills) if all_kills else 0,
                'max_kills': max(all_kills) if all_kills else 0,
                'avg_frames': sum(all_frames) / len(all_frames) if all_frames else 0,
            }
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f)

    print(f"\nSaved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    # Cleanup temp files
    for temp_file in Path('.').glob(args.output.replace('.json', '_temp_*.json')):
        temp_file.unlink()
        print(f"Cleaned up: {temp_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
