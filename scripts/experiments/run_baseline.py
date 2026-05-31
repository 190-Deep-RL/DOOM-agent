"""
================================================================================
EXPERIMENT: BASELINE (PURE NEURAL NETWORK)
================================================================================

Purpose:
This script runs the "Control Group" for your paper's experiments. 
It tests the agent WITHOUT any Monte Carlo Tree Search (MCTS) or LLM evaluation.
The agent relies 100% on the raw "instincts" it learned during its initial 
behavioral cloning training. 

Hypothesis / Use Case for the Paper:
You MUST have this baseline in your paper to prove that adding MCTS actually
improves the agent! By comparing the MCTS results to this baseline, you can
calculate exactly how much "smarter" or "more human-like" the search tree 
makes the agent.

Usage:
    python scripts/experiments/run_baseline.py --episodes 10
"""

import argparse
from experiment_utils import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # You can change the default number of episodes here
    parser.add_argument("--episodes", type=int, default=10)
    # The name of the CSV file this script will generate
    parser.add_argument("--output", type=str, default="results_baseline.csv")
    args = parser.parse_args()

    run_experiment(
        config_name="Baseline (Base Model)",
        is_baseline=True,  # Tells the utility to use benchmark.py instead of MCTS
        args_list=[],      # No parameters needed for the baseline
        episodes=args.episodes,
        output_file=args.output
    )
