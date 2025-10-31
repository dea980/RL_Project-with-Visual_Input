"""
Visual wrapper for CartPole environment
Converts state observations to visual RGB observations
"""

import gymnasium as gym
import numpy as np
import cv2
from gymnasium import Wrapper, spaces
from collections import deque
import os
from PIL import Image


class RGBArrayAsObservationWrapper(Wrapper):
    """
    Wrapper that converts CartPole state to RGB visual observations
    """
    
    def __init__(self, env, image_size=(64, 64), num_frames=5, test=False, image_dir="data/raw/cartpole"):
        super(RGBArrayAsObservationWrapper, self).__init__(env)
        
        self.image_size = image_size
        self.num_frames = num_frames
        self.test = test
        self.i = 0
        
        # Create image directory if in test mode
        if self.test:
            os.makedirs(image_dir, exist_ok=True)
            self.image_dir = image_dir
        
        # Initialize environment to get observation space
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            _, _ = reset_result
        dummy_obs = self._render_and_resize()
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.num_frames, *dummy_obs.shape), 
            dtype=dummy_obs.dtype
        )
        
        # Frame buffer for temporal information
        self.frame_buffer = deque(maxlen=self.num_frames)
        for _ in range(self.num_frames):
            self.frame_buffer.append(dummy_obs)
    
    def _render_and_resize(self):
        """Render environment and resize image"""
        obs = self.env.render()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.image_size, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)
    
    def reset(self, **kwargs):
        """Reset environment and initialize frame buffer"""
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            _, info = reset_result
        else:
            info = {}
        obs = self._render_and_resize()
        
        # Fill frame buffer with initial observation
        self.frame_buffer.clear()
        for _ in range(self.num_frames):
            self.frame_buffer.append(obs)
        
        return np.stack(list(self.frame_buffer)), info
    
    def step(self, action):
        """Step environment and update frame buffer"""
        step_result = self.env.step(action)
        if len(step_result) == 5:
            _, reward, terminated, truncated, info = step_result
        else:
            _, reward, done, info = step_result
            terminated = bool(done)
            truncated = False
        obs = self._render_and_resize()
        
        # Save image if in test mode
        if self.test:
            im_path = os.path.join(self.image_dir, f"frame_{self.i}.png")
            self.i += 1
            Image.fromarray(obs).save(im_path)
        
        # Update frame buffer
        self.frame_buffer.append(obs)
        
        return np.stack(list(self.frame_buffer)), reward, terminated, truncated, info


def create_visual_cartpole(image_size=(64, 64), num_frames=5, test=False, image_dir="data/raw/cartpole"):
    """
    Create CartPole environment with visual observations
    
    Args:
        image_size: Tuple of (height, width) for resized images
        num_frames: Number of frames to stack for temporal information
        test: Whether to save images to disk
        image_dir: Directory to save images when test=True
    
    Returns:
        Wrapped environment with visual observations
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RGBArrayAsObservationWrapper(env, image_size, num_frames, test, image_dir)
    return env
