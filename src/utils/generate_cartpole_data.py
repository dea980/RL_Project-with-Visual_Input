#!/usr/bin/env python3
"""
Generate CartPole visual data for VAE training
"""

import os
import sys
import gymnasium as gym
import numpy as np
import cv2
from PIL import Image
import argparse
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.visual_wrapper import create_visual_cartpole

def generate_cartpole_frames(n_episodes=100, max_steps=500, image_size=(64, 64), save_dir="data/raw/cartpole"):
    """
    Generate CartPole frames for VAE training
    
    Args:
        n_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        image_size: Size of generated images
        save_dir: Directory to save frames
    """
    print(f"🎮 Generating CartPole frames...")
    print(f"Episodes: {n_episodes}, Max steps: {max_steps}, Image size: {image_size}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visual environment
    env = create_visual_cartpole(
        image_size=image_size,
        num_frames=1,  # Single frame for VAE
        test=True,     # Save images
        image_dir=save_dir
    )
    
    all_frames = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            # Random action (you can replace with a trained policy)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            # Store frame for VAE training
            if len(obs.shape) == 3:  # Single frame
                all_frames.append(obs[0])  # Take first (and only) frame
            else:  # Multiple frames
                all_frames.append(obs[0])  # Take first frame
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}")
    
    # Convert to numpy array
    frames_array = np.array(all_frames)
    
    # Save frames as .npy file for easy loading
    frames_path = os.path.join(save_dir, "cartpole_frames.npy")
    np.save(frames_path, frames_array)
    
    # Save metadata
    metadata = {
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'image_size': image_size,
        'total_frames': len(all_frames),
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }
    
    metadata_path = os.path.join(save_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Generated {len(all_frames)} frames")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Frames saved to: {frames_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    env.close()
    return frames_array

def load_cartpole_frames(data_path="data/raw/cartpole/cartpole_frames.npy"):
    """Load previously generated CartPole frames"""
    if os.path.exists(data_path):
        frames = np.load(data_path)
        print(f"Loaded {len(frames)} frames from {data_path}")
        return frames
    else:
        print(f"No frames found at {data_path}")
        print("Run generate_cartpole_frames() first")
        return None

def create_sample_grid(frames, save_path, n_samples=16):
    """Create a grid of sample frames"""
    # Select random samples
    indices = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
    sample_frames = frames[indices]
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for i, frame in enumerate(sample_frames):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].imshow(frame, cmap='gray')
        axes[row, col].set_title(f'Frame {indices[i]}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(sample_frames), grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample grid saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate CartPole visual data')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to generate')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64],
                       help='Image size (height width)')
    parser.add_argument('--save_dir', default='data/raw/cartpole',
                       help='Directory to save frames')
    parser.add_argument('--load_existing', action='store_true',
                       help='Load existing frames instead of generating new ones')
    
    args = parser.parse_args()
    
    if args.load_existing:
        frames = load_cartpole_frames(os.path.join(args.save_dir, "cartpole_frames.npy"))
        if frames is not None:
            create_sample_grid(frames, os.path.join(args.save_dir, "sample_grid.png"))
    else:
        frames = generate_cartpole_frames(
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            image_size=tuple(args.image_size),
            save_dir=args.save_dir
        )
        create_sample_grid(frames, os.path.join(args.save_dir, "sample_grid.png"))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
