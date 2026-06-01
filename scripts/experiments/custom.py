import argparse
import sys
import os

# Add parent directory to path so it can import experiment_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_rollout-temperature_0_9.csv")
    args = parser.parse_args()

    run_experiment(
        config_name="MCTS Rollout Temp=0.1",
        is_baseline=False,
        args_list=["--rollout-temperature", "0.03", "--simulations", "1", "--depth", "1"],  # Added simulations to keep it consistent with other rollout temp experiments
        episodes=args.episodes,
        output_file=args.output
    )
