#!/usr/bin/env python3
"""
Experiment 3: CNN After VAE
Uses PPO with MLP policy on VAE-encoded visual inputs
"""

import os
import sys
import yaml
import numpy as np
import torch
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
from models.linear_vae import LinearVAE
from models.conv_vae import ConvVAE

RESULTS_ROOT = 'results'
LOGS_DIR = os.path.join(RESULTS_ROOT, 'logs')
PLOTS_DIR = os.path.join(RESULTS_ROOT, 'plots')
MODELS_DIR = os.path.join(RESULTS_ROOT, 'models')
EXP_LOG_DIR = os.path.join(LOGS_DIR, 'exp3_cnn_after')
EXP_EVAL_LOG_DIR = os.path.join(LOGS_DIR, 'exp3_cnn_after_eval')
EXP_MODEL_DIR = os.path.join(MODELS_DIR, 'exp3_cnn_after')
FINAL_MODEL_PATH = os.path.join(EXP_MODEL_DIR, 'exp3_cnn_after_final')
VAE_MODEL_CANDIDATES = [
    os.path.join(EXP_MODEL_DIR, 'vae_model.pth'),
    os.path.join('data', 'generated', 'vae_linear_model.pth'),
]


class VAEEnvironmentWrapper(gym.Wrapper):
    """Wrapper that encodes visual observations with a VAE before handing them to the agent."""

    def __init__(self, env, vae_model, device='cpu'):
        super().__init__(env)
        self.vae_model = vae_model
        self.device = device
        self.vae_model.eval()

        # Infer latent dimension for observation space definition
        sample_obs, _ = self.env.reset()
        sample_latent = self._encode_observation(sample_obs)
        latent_dim = sample_latent.shape[-1]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        encoded_obs = self._encode_observation(obs)
        return encoded_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        encoded_obs = self._encode_observation(obs)
        return encoded_obs, reward, terminated, truncated, info

    def _encode_observation(self, obs):
        """Encode the visual observation using the VAE."""
        with torch.no_grad():
            obs_array = np.asarray(obs, dtype=np.float32)
            if obs_array.ndim == 2:
                obs_array = np.expand_dims(obs_array, axis=0)
            obs_tensor = torch.from_numpy(obs_array).to(self.device)

            if isinstance(self.vae_model, ConvVAE):
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(1)
                elif obs_tensor.ndim == 4 and obs_tensor.shape[1] != 1:
                    obs_tensor = obs_tensor.mean(dim=1, keepdim=True)
            else:
                obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)

            mean, _ = self.vae_model.encode(obs_tensor)
            encoded = mean.cpu().numpy().astype(np.float32)
            return encoded[0] if encoded.shape[0] == 1 else encoded


def load_config():
    """Load experiment configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'experiment_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']['exp3_cnn_after']


def load_vae_model(config):
    """Load a pre-trained VAE model or fall back to a fresh instance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_model = LinearVAE(
        in_dim=(64, 64),
        latent_dim=config['vae_latent_dim'],
        device=device,
    )

    loaded = False
    for candidate in VAE_MODEL_CANDIDATES:
        if not os.path.exists(candidate):
            continue
        try:
            state_dict = torch.load(candidate, map_location=device)
            vae_model.load_state_dict(state_dict)
        except RuntimeError as exc:
            print(f"Warning: could not load VAE weights from {candidate}: {exc}")
            continue
        print(f"Loaded VAE model from {candidate}")
        loaded = True
        break

    if not loaded:
        print("Warning: VAE model not found. Using untrained weights.")

    return vae_model, device


def create_environment(config, vae_model=None, device=None, log_dir=None, test=False):
    """Create the CartPole environment with visual observations encoded by the VAE."""
    base_env = create_visual_cartpole(
        image_size=(64, 64),
        num_frames=1,
        test=test,
    )

    if vae_model is None or device is None:
        vae_model, device = load_vae_model(config)

    env = VAEEnvironmentWrapper(base_env, vae_model, device)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)

    return env


def train_model(env, config, vae_model, device):
    """Train the PPO model on VAE-encoded inputs."""
    model = PPO(
        "MlpPolicy",
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
        tensorboard_log=LOGS_DIR,
    )

    eval_env = create_environment(config, vae_model, device, log_dir=EXP_EVAL_LOG_DIR)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=EXP_MODEL_DIR,
        log_path=EXP_EVAL_LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        tb_log_name="exp3_cnn_after",
    )

    return model


def evaluate_model(model, env, n_episodes=10):
    """Evaluate the trained model across several episodes."""
    episode_rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
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
    """Plot training results saved by the monitor."""
    results = load_results(EXP_LOG_DIR)
    x, y = ts2xy(results, 'timesteps')

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Experiment 3: CNN After VAE Training Progress')
    plt.grid(True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'exp3_cnn_after_training.png'))
    plt.show()


def main():
    """Main experiment function."""
    print("Starting Experiment 3: CNN After VAE")

    config = load_config()
    print(f"Configuration: {config}")

    for directory in (LOGS_DIR, EXP_LOG_DIR, EXP_EVAL_LOG_DIR, PLOTS_DIR, EXP_MODEL_DIR):
        os.makedirs(directory, exist_ok=True)

    vae_model, device = load_vae_model(config)
    env = create_environment(config, vae_model, device, log_dir=EXP_LOG_DIR)
    print("VAE-encoded environment created successfully")

    print("Training model...")
    model = train_model(env, config, vae_model, device)
    print("Training completed")

    print("Evaluating model...")
    rewards = evaluate_model(model, env)
    print(f"Average reward over 10 episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    plot_results()

    model.save(FINAL_MODEL_PATH)
    print(f"Model saved to {FINAL_MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()
