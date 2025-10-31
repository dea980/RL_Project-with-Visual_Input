#!/usr/bin/env python3
"""
Experiment 2: CNN Before VAE
Uses PPO with CNN policy on direct visual inputs
"""

import os
import sys
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from environments.visual_wrapper import create_visual_cartpole

RESULTS_ROOT = 'results'
LOGS_DIR = os.path.join(RESULTS_ROOT, 'logs')
PLOTS_DIR = os.path.join(RESULTS_ROOT, 'plots')
MODELS_DIR = os.path.join(RESULTS_ROOT, 'models')
EXP_LOG_DIR = os.path.join(LOGS_DIR, 'exp2_cnn_before')
EXP_EVAL_LOG_DIR = os.path.join(LOGS_DIR, 'exp2_cnn_before_eval')
EXP_MODEL_DIR = os.path.join(MODELS_DIR, 'exp2_cnn_before')
FINAL_MODEL_PATH = os.path.join(EXP_MODEL_DIR, 'exp2_cnn_before_final')

def load_config():
    """Load experiment configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'experiment_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']['exp2_cnn_before']

def create_environment():
    """Create visual CartPole environment"""
    os.makedirs(EXP_LOG_DIR, exist_ok=True)
    env = create_visual_cartpole(
        image_size=tuple(load_config()['image_size']),
        num_frames=load_config()['num_frames'],
        test=True  # Save images for analysis
    )
    env = Monitor(env, EXP_LOG_DIR)
    return env

def train_model(env, config):
    """Train the PPO model with CNN policy"""
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=float(config['learning_rate']),
        n_steps=config['batch_size'] * 4,
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=float(config['clip_range']),
        ent_coef=float(config['ent_coef']),
        vf_coef=float(config['vf_coef']),
        max_grad_norm=float(config['max_grad_norm']),
        verbose=1,
        tensorboard_log='results/logs'
    )
    ## wrapper for gymnasium
    # env = GymCompatibilityWrapper(env) might need it .
    # Create evaluation environment
    eval_env = create_visual_cartpole(
        image_size=tuple(config['image_size']),
        num_frames=config['num_frames'],
        test=False
    )
    os.makedirs(EXP_EVAL_LOG_DIR, exist_ok=True)
    eval_env = Monitor(eval_env, EXP_EVAL_LOG_DIR)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=EXP_MODEL_DIR,
        log_path=EXP_EVAL_LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        tb_log_name="exp2_cnn_before"
    )
    
    return model

def evaluate_model(model, env, n_episodes=10):
    """Evaluate the trained model"""
    episode_rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward}")
    
    return episode_rewards

def plot_results():
    """Plot training results"""
    results = load_results(EXP_LOG_DIR)
    x, y = ts2xy(results, 'timesteps')
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Experiment 2: CNN Before VAE Training Progress')
    ax.grid(True)
    fig.savefig(os.path.join(PLOTS_DIR, 'exp2_cnn_before_training.png'))
    plt.close(fig)

def main():
    """Main experiment function"""
    print("Starting Experiment 2: CNN Before VAE")
    
    # Load configuration
    config = load_config()
    print(f"Configuration: {config}")
    
    for directory in (LOGS_DIR, EXP_LOG_DIR, EXP_EVAL_LOG_DIR, PLOTS_DIR, EXP_MODEL_DIR):
        os.makedirs(directory, exist_ok=True)

    # Create environment
    env = create_environment()
    print("Visual environment created successfully")
    
    # Train model
    print("Training model...")
    model = train_model(env, config)
    print("Training completed")
    
    # Evaluate model
    print("Evaluating model...")
    rewards = evaluate_model(model, env)
    print(f"Average reward over 10 episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Plot results
    plot_results()
    
    # Save model
    model.save(FINAL_MODEL_PATH)
    print(f"Model saved to {FINAL_MODEL_PATH}")
    
    env.close()

if __name__ == "__main__":
    main()
