"""
================================================================================
DOOM MCTS EXPERIMENT UTILITIES
================================================================================

This file contains the core logic for running automated DOOM experiments. 
Rather than duplicating the complex subprocess execution and Regex parsing logic
across every single experiment script, all the individual scripts (like 
run_baseline.py, run_mcts_deep.py) simply import the `run_experiment` function 
from this file.

What this file handles:
1. Subprocess Execution: Runs the actual DOOM scripts (benchmark.py or play_doom_mcts.py)
   in a separate background process so crashes don't kill our experiment runner.
2. Output Parsing: Reads the raw text output from the DOOM scripts and uses
   Regular Expressions (Regex) to extract the key metrics for your paper.
3. Data Logging: Safely writes the extracted metrics to a CSV file.

DO NOT RUN THIS FILE DIRECTLY. Run the individual experiment scripts instead.
"""

import argparse
import subprocess
import re
import csv
import json
import os
import sys
from datetime import datetime

# Add the parent directory to Python's path so we can run this from anywhere in the terminal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run_command(cmd_list):
    """
    Executes a shell command in a subprocess and captures the raw text output.
    
    Args:
        cmd_list (list): The command to run, split into a list of strings 
                         (e.g., ['python', 'script.py', '--arg']).
                         
    Returns:
        str: The full standard output (stdout) printed by the command.
    """
    print(f"Running: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    
    # If the DOOM agent crashes for some reason, print the error so we can debug
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd_list)}")
        print(result.stderr)
        
    return result.stdout

def parse_mcts_output(output_text):
    """
    Uses Regular Expressions to hunt through the raw text output of play_doom_mcts.py
    and extract the numerical data for the paper.
    
    Returns a list of dictionaries: one for each episode, and one final 'AVERAGE' row.
    """
    steps = []
    rewards = []
    kills = []
    healths = []
    latencies = []
    
    steps_pattern = re.compile(r"Steps:\s*(\d+)")
    reward_pattern = re.compile(r"Total reward:\s*(-?\d+)")
    kills_pattern = re.compile(r"Kills:\s*(\d+)")
    health_pattern = re.compile(r"Ending Health:\s*(\d+)")
    latency_pattern = re.compile(r"Avg decision time:\s*(\d+)ms")
    
    for line in output_text.splitlines():
        line = line.strip()
        
        m_steps = steps_pattern.search(line)
        if m_steps: steps.append(int(m_steps.group(1)))
        
        m_reward = reward_pattern.search(line)
        if m_reward: rewards.append(float(m_reward.group(1)))
        
        m_kills = kills_pattern.search(line)
        if m_kills: kills.append(int(m_kills.group(1)))
        
        m_health = health_pattern.search(line)
        if m_health: healths.append(int(m_health.group(1)))
        
        m_latency = latency_pattern.search(line)
        if m_latency: latencies.append(float(m_latency.group(1)))

    if not steps: return None
        
    episodes_data = []
    for i in range(len(steps)):
        episodes_data.append({
            "Episode": str(i + 1),
            "Steps (Survival)": steps[i],
            "Reward": rewards[i] if i < len(rewards) else 0.0,
            "Kills": kills[i] if i < len(kills) else 0.0,
            "Ending Health": healths[i] if i < len(healths) else 0.0,
            "Latency (ms)": latencies[i] if i < len(latencies) else 0.0
        })

    # Add the summary row at the bottom
    episodes_data.append({
        "Episode": "AVERAGE",
        "Steps (Survival)": sum(steps) / len(steps),
        "Reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "Kills": sum(kills) / len(kills) if kills else 0.0,
        "Ending Health": sum(healths) / len(healths) if healths else 0.0,
        "Latency (ms)": sum(latencies) / len(latencies) if latencies else 0.0
    })
    
    return episodes_data

def run_baseline(episodes):
    """
    Special runner for the Baseline (No MCTS). 
    Since the baseline uses benchmark.py instead of play_doom_mcts.py, 
    it outputs a JSON file instead of console text. We parse that JSON here.
    """
    json_path = "temp_benchmark_results.json"
    cmd = ["python", "scripts/benchmark.py", "--agent", "multivec", "--scenario", "deathmatch", "--episodes", str(episodes), "--armed", "--steps", "200", "--output", json_path]
    run_command(cmd)
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        os.remove(json_path)
        metrics = list(data.values())[0] if data else {}
        
        # We now have the raw individual episode data!
        episodes_data = []
        raw_episodes = metrics.get('raw_episodes', [])
        
        for i, ep in enumerate(raw_episodes):
            episodes_data.append({
                "Episode": str(i + 1),
                "Steps (Survival)": ep.get('steps', 0),
                "Reward": ep.get('kills', 0),
                "Kills": ep.get('kills', 0),
                "Ending Health": ep.get('health_remaining', 0.0),
                "Latency (ms)": ep.get('avg_latency', 0.0)
            })
            
        # Add the summary row at the bottom
        episodes_data.append({
            "Episode": "AVERAGE",
            "Steps (Survival)": metrics.get("avg_survival_steps", 0),
            "Reward": metrics.get("avg_kills", 0),
            "Kills": metrics.get("avg_kills", 0),
            "Ending Health": sum(ep.get('health_remaining', 0.0) for ep in raw_episodes) / len(raw_episodes) if raw_episodes else 0.0,
            "Latency (ms)": metrics.get("avg_latency_ms", 0)
        })
        
        return episodes_data
    return None

def run_experiment(config_name, is_baseline, args_list, episodes, output_file):
    print(f"\n{'='*50}\nRunning {config_name} for {episodes} episodes\n{'='*50}")
    
    if is_baseline:
        episodes_data = run_baseline(episodes)
    else:
        cmd = ["python", "scripts/play_doom_mcts.py", "--scenario", "defend_the_center", "--episodes", str(episodes), "--steps", "200", "--batch-size", "8"] + args_list
        output = run_command(cmd)
        episodes_data = parse_mcts_output(output)
        
    if episodes_data:
        # Print summary to console
        num_extracted = len(episodes_data) - 1 if not is_baseline else episodes
        print(f"Results successfully extracted for {num_extracted} episodes.")
        summary_row = episodes_data[-1]
        for k, v in summary_row.items():
            if k != "Episode":
                print(f"  Avg {k}: {v:.2f}")
            
        # Write to the CSV file
        fieldnames = ["Configuration Name", "Episode", "Steps (Survival)", "Reward", "Kills", "Ending Health", "Latency (ms)"]
        
        # If running the same experiment multiple times, overwrite it rather than appending 
        # so it stays clean with one average row at the bottom.
        with open(output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for row in episodes_data:
                row["Configuration Name"] = config_name
                writer.writerow(row)
            
        print(f"Results (All Episodes + Average) saved to {output_file}")
    else:
        print("Failed to extract metrics.")
