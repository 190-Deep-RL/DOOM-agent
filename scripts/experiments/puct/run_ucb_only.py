"""
================================================================================
EXPERIMENT: SANITY CHECK (PUCT = FALSE)
================================================================================

Purpose:
This script sanity checks the PUCT algorithm. By default, the DOOM agent uses 
PUCT (Polynomial Upper Confidence Trees) to guide its tree search using the 
neural network's initial instincts. 

This script disables PUCT (--use-ucb), forcing the agent to fall back to the 
standard UCB1 algorithm. 

Hypothesis / Use Case for the Paper:
If PUCT is actually working, it should drastically speed up the MCTS by pruning 
bad branches early. If disabling it (using standard UCB1) performs identically 
or better, it proves that the neural network "priors" aren't actually helping 
the search engine in this specific DOOM environment!
"""

import argparse
import sys
import os

# Add parent directory to path so it can import experiment_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_ucb_only.csv")
    args = parser.parse_args()

    run_experiment(
        config_name="MCTS (UCB1 Only / PUCT=False)",
        is_baseline=False,
        args_list=[
            "scripts/play_doom_mcts.py",
            "--scenario", "deathmatch",
            "--episodes", str(args.episodes),
            "--armed",
            "--steps", "200",
            "--batch-size", "8",
            "--use-ucb", "--simulations", "10",  # Turns off use_puct
        ],
        episodes=args.episodes,
        output_file=args.output
    )
