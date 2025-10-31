"""
Gymnasium to Gym compatibility wrapper
Converts gymnasium.spaces to gym.spaces for Stable-Baselines3 compatibility
"""

import gymnasium as gym
import gym as old_gym
from gymnasium import spaces as gymnasium_spaces
from gym import spaces as gym_spaces
import numpy as np


class GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper that converts gymnasium.spaces to gym.spaces for Stable-Baselines3 compatibility
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Keep gymnasium spaces (Stable-Baselines3 2.7.0+ supports gymnasium)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


def create_gym_compatible_env(env_name, **kwargs):
    """
    Create a gymnasium environment and wrap it for Stable-Baselines3 compatibility
    """
    env = gym.make(env_name, **kwargs)
    env = GymCompatibilityWrapper(env)
    return env


def create_gym_compatible_visual_env(env_name, image_size=(64, 64), num_frames=5, **kwargs):
    """
    Create a visual environment compatible with Stable-Baselines3
    """
    from .improved_visual_wrapper import create_improved_visual_cartpole
    
    # Create visual environment
    env = create_improved_visual_cartpole(image_size, num_frames, **kwargs)
    
    # Wrap for gym compatibility
    env = GymCompatibilityWrapper(env)
    
    return env
