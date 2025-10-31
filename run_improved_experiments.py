"""
Improved Experiments Runner
Combines quick training, VAE training, and progressive improvement testing.
Based on original approach with enhancements.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from experiments.quick_training.run_quick_experiment import run_quick_training_experiment, generate_test_images
from src.models.improved_vae_training import train_vae_experiment
from src.utils.generate_cartpole_data import generate_cartpole_frames


def run_all_improved_experiments():
    """
    Run all improved experiments in sequence.
    """
    print("Starting All Improved Experiments")
    print("=" * 50)
    
    # Step 1: Generate CartPole frames
    print("\nStep 1: Generating CartPole frames...")
    generate_cartpole_frames(
        n_episodes=100,
        max_steps=200,
        image_size=(64, 64),
        save_dir="data/raw/cartpole"
    )
    print("CartPole frames generated")
    
    # Step 2: Quick RL training
    print("\nStep 2: Quick RL training...")
    model, rewards = run_quick_training_experiment()
    print("RL training completed")
    
    # Step 3: Generate test images with trained model
    print("\nStep 3: Generating test images...")
    generate_test_images(model)
    print("Test images generated")
    
    # Step 4: VAE training
    print("\nStep 4: VAE training...")
    train_vae_experiment()
    print("VAE training completed")
    
    print("\nAll experiments completed successfully!")
    print("=" * 50)
    
    # Summary
    print("\nSummary:")
    print(f"RL Model: results/models/quick_training/final_model.pkl")
    print(f"Test Images: data/raw/cartpole_test/")
    print(f"VAE Models: data/generated/")
    print(f"Logs: results/logs/quick_training/")


def run_quick_rl_only():
    """
    Run only quick RL training.
    """
    print("Running Quick RL Training Only")
    print("=" * 40)
    
    model, rewards = run_quick_training_experiment()
    generate_test_images(model)
    
    print("\nQuick RL training completed!")
    print(f"Final rewards: {rewards[:5]}... (showing first 5)")


def run_vae_only():
    """
    Run only VAE training.
    """
    print("Running VAE Training Only")
    print("=" * 40)
    
    train_vae_experiment()
    
    print("\nVAE training completed!")


def run_progressive_training():
    """
    Run progressive training to test model improvement.
    """
    print("Running Progressive Training")
    print("=" * 40)
    
    # Load configuration
    with open("configs/quick_training_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Run multiple training cycles
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}/3")
        
        # Quick training
        model, rewards = run_quick_training_experiment()
        
        # Evaluate performance
        avg_reward = sum(rewards) / len(rewards)
        print(f"Cycle {cycle + 1} Average Reward: {avg_reward:.2f}")
        
        # Save cycle-specific model
        cycle_model_path = f"results/models/quick_training/cycle_{cycle + 1}_model.pkl"
        model.save(cycle_model_path)
        print(f"Cycle {cycle + 1} model saved: {cycle_model_path}")
        
        # Generate test images for this cycle
        cycle_image_dir = f"data/raw/cartpole_cycle_{cycle + 1}"
        os.makedirs(cycle_image_dir, exist_ok=True)
        
        # Update config for this cycle
        cycle_config = config.copy()
        cycle_config['test_image_dir'] = cycle_image_dir
        
        # Save cycle config
        with open(f"configs/cycle_{cycle + 1}_config.yaml", 'w') as f:
            yaml.dump(cycle_config, f)
        
        # Generate images
        generate_test_images(model)
        
        print(f"Cycle {cycle + 1} completed")
    
    print("\nProgressive training completed!")
    print("Check results in:")
    print("   - results/models/quick_training/cycle_*_model.pkl")
    print("   - data/raw/cartpole_cycle_*/")


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description="Run improved RL experiments")
    parser.add_argument(
        "experiment", 
        choices=["all", "quick_rl", "vae", "progressive"],
        help="Experiment to run"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/quick_training_config.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/generated", exist_ok=True)
    
    # Run selected experiment
    if args.experiment == "all":
        run_all_improved_experiments()
    elif args.experiment == "quick_rl":
        run_quick_rl_only()
    elif args.experiment == "vae":
        run_vae_only()
    elif args.experiment == "progressive":
        run_progressive_training()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
