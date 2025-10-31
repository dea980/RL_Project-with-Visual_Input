#!/usr/bin/env python3
"""
Main script to run all experiments
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

def run_experiment(exp_name, exp_dir):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Starting {exp_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Change to experiment directory and run
        result = subprocess.run(
            [sys.executable, 'run_experiment.py'],
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"{exp_name} completed successfully in {duration:.2f} seconds")
            return True
        else:
            print(f"{exp_name} failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{exp_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"{exp_name} failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['exp1', 'exp2', 'exp3', 'vae', 'all'],
                       default=['all'],
                       help='Which experiments to run')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (not recommended)')
    
    args = parser.parse_args()
    
    # Define experiments
    experiments = {
        'exp1': ('Experiment 1: Native State RL', 'experiments/exp1_native'),
        'exp2': ('Experiment 2: CNN Before VAE', 'experiments/exp2_cnn_before'),
        'exp3': ('Experiment 3: CNN After VAE', 'experiments/exp3_cnn_after'),
        'vae': ('VAE Training: Generate Images', 'experiments/vae_training')
    }
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        exp_list = ['vae', 'exp1', 'exp2', 'exp3']  # VAE first to generate data
    else:
        exp_list = args.experiments
    
    print(f"🚀 Starting RL Experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running experiments: {', '.join(exp_list)}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # Run experiments
    results = {}
    total_start_time = time.time()
    
    for exp in exp_list:
        if exp in experiments:
            exp_name, exp_dir = experiments[exp]
            results[exp] = run_experiment(exp_name, exp_dir)
        else:
            print(f"Unknown experiment: {exp}")
            results[exp] = False
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for exp in exp_list:
        if exp in experiments:
            exp_name, _ = experiments[exp]
            status = "PASSED" if results[exp] else " FAILED"
            print(f"{exp_name}: {status}")
    
    print(f"\nTotal time: {total_duration:.2f} seconds")
    print(f"Results saved in: results/")
    
    # Check if all experiments passed
    all_passed = all(results.values())
    if all_passed:
        print("\n All experiments completed successfully!")
        return 0
    else:
        print("\n Some experiments failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
