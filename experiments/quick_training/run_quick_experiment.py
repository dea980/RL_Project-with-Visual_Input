"""
Quick Training Experiment - Based on original approach with improvements.
Trains PPO for short episodes and progressively increases training time.
"""

import os
import sys
import yaml
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from environments.improved_visual_wrapper import create_improved_visual_cartpole_with_monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def run_quick_training_experiment(config_path="configs/quick_training_config.yaml"):
    """
    Run quick training experiment with progressive improvement.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    log_dir = config['log_dir']
    model_dir = config['model_dir']
    image_dir = config['image_dir']
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    print("Starting Quick Training Experiment")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    print(f" Images: {image_dir}")
    
    # Create environment
    print("\nCreating environment...")
    env = create_improved_visual_cartpole_with_monitor(
        log_dir=log_dir,
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        test=False,  # Don't save images during training
        image_dir=image_dir
    )
    
    # Check environment (disabled due to spaces compatibility issue)
    # check_env(env, warn=True)
    
    # Get observation info
    obs, _ = env.reset()
    print(f" Observation shape: {obs.shape}")
    print(f" Action space: {env.action_space}")
    
    # Training phases
    training_phases = config['training_phases']
    model = None
    
    for phase, phase_config in enumerate(training_phases):
        print(f"\n Phase {phase + 1}: {phase_config['name']}")
        print(f"   Steps: {phase_config['steps']}")
        print(f"   Learning Rate: {phase_config['learning_rate']}")
        
        start_time = time.time()
        
        if model is None:
            # Create new model
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                learning_rate=float(phase_config['learning_rate']),
                n_steps=phase_config['n_steps'],
                batch_size=phase_config['batch_size'],
                n_epochs=phase_config['n_epochs'],
                gamma=phase_config['gamma'],
                gae_lambda=phase_config['gae_lambda'],
                clip_range=phase_config['clip_range'],
                ent_coef=phase_config['ent_coef'],
                vf_coef=phase_config['vf_coef'],
                max_grad_norm=phase_config['max_grad_norm'],
                tensorboard_log=log_dir
            )
        else:
            # Update learning rate for existing model
            model.learning_rate = float(phase_config['learning_rate'])
        
        # Train model
        model.learn(total_timesteps=phase_config['steps'])
        
        # Save model
        model_path = os.path.join(model_dir, f"phase_{phase + 1}_model.pkl")
        model.save(model_path)
        print(f" Model saved: {model_path}")
        
        # Evaluate model
        print(" Evaluating model...")
        eval_rewards = evaluate_model(model, env, num_episodes=10)
        avg_reward = np.mean(eval_rewards)
        max_reward = np.max(eval_rewards)
        
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Phase time: {time.time() - start_time:.2f}s")
        
        # Check if we've solved the environment
        if avg_reward >= config['success_threshold']:
            print(f"Environment solved! Average reward: {avg_reward:.2f}")
            break
    
    # Final evaluation
    print("\nFinal Evaluation")
    final_rewards = evaluate_model(model, env, num_episodes=50)
    final_avg = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    
    print(f"Final Average Reward: {final_avg:.2f} ± {final_std:.2f}")
    print(f"Best Reward: {np.max(final_rewards):.2f}")
    print(f"Worst Reward: {np.min(final_rewards):.2f}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pkl")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    return model, final_rewards


def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate model performance over multiple episodes.
    """
    rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return rewards


def generate_test_images(model, config_path="configs/quick_training_config.yaml"):
    """
    Generate test images using trained model.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create test environment
    test_env = create_improved_visual_cartpole_with_monitor(
        log_dir="test_logs/",
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        test=True,  # Save images
        image_dir=config['test_image_dir']
    )
    
    print("Generating test images...")
    
    # Generate images
    for episode in range(config['test_episodes']):
        obs, _ = test_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < config['max_steps_per_episode']:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        
        print(f"Episode {episode + 1}: Reward={total_reward}, Steps={step}")
    
    print(f"Generated images saved to: {config['test_image_dir']}")


if __name__ == "__main__":
    # Run quick training experiment
    model, rewards = run_quick_training_experiment()
    
    # Generate test images
    generate_test_images(model)
    
    print("\nQuick Training Experiment Complete!")
