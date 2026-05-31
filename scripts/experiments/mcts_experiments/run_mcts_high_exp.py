import argparse
import sys
import os

# Add parent directory to path so it can import experiment_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_mcts_high_exp.csv")
    args = parser.parse_args()

    run_experiment(
        config_name="MCTS High Exploration",
        is_baseline=False,
        args_list=["--simulations", "25", "--depth", "20", "--c", "2.8"],
        episodes=args.episodes,
        output_file=args.output
    )
