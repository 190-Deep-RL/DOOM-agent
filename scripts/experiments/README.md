# DOOM Agent Experiments Suite

This folder contains a suite of automated experiment runners designed to test various parameters of the DOOM Agent. 

By separating these into independent scripts within organized folders, multiple team members can run different experiments in parallel on their own machines without causing merge conflicts or data overlap.

## How to Use

Each experiment is a standalone Python script. **You must run the scripts from the root of the repository** (not from inside the folders) so that the paths resolve correctly.

By default, each script runs **3 episodes**. To run a full experiment for the paper, use the `--episodes` flag:
```bash
python scripts/experiments/mcts_experiments/run_mcts_default.py --episodes 10
```

*(Note: You do not need to manually add parameter flags like `--use-ucb` or `--c 0.5`. The scripts inject those automatically based on their filename!)*

### Outputs
When a script finishes, it will automatically append its results to a corresponding CSV file in the root directory. You can safely run these scripts multiple times; they will append new data to the bottom of the CSV rather than overwriting your past work.

---

## 📁 Organization & The Experiments

### 1. The Baseline (`run_baseline.py`)
Located directly in this folder.
*   **What it is:** The control group. Runs the pure `DoomMultiVecClassifier` neural network without any search tree.
*   **Why it matters:** You must run this to prove that adding MCTS actually improves the agent's performance over the raw instincts.

### 2. MCTS Structural Tests (`mcts_experiments/`)
Tests the structure and size of the MCTS tree.
*   **`run_mcts_default.py`:** The standard control benchmark.
*   **`run_mcts_deep.py`:** Tests a deeper rollout (depth=40). Looks further into the future.
*   **`run_mcts_wide.py`:** Tests wider exploration (simulations=50). Explores more parallel branches.
*   **`run_mcts_high_exp.py`:** Highly random exploration constant (c=3.0).
*   **`run_mcts_low_exp.py`:** Highly greedy/deterministic exploration constant (c=0.5).

### 3. PUCT Sanity Test (`puct/`)
*   **`run_ucb_only.py`:** Disables the advanced PUCT algorithm and falls back to standard UCB1. Proves whether or not the neural network "priors" actually help guide the search in the DOOM environment.

### 4. Rollout Temperature Tests (`rollout_temp/`)
Contains files to test temperatures from `0.001` to `0.9`.
*   **What it tests:** Controls how random the agent is *while simulating the future*. A low temperature makes the simulated agent completely greedy. A high temperature makes the simulated agent erratic and random.

### 5. Prior Temperature Tests (`prior_temp/`)
Contains files to test temperatures from `0.001` to `0.9`.
*   **What it tests:** Controls how strongly the agent trusts its initial neural network "instincts" at the very root of the tree before exploring.

---

## Core Utilities
*   **`experiment_utils.py`**: Do not run this file directly. This is the core engine that handles the subprocess execution, regex parsing of the console outputs, and safe CSV writing for all the scripts above.
