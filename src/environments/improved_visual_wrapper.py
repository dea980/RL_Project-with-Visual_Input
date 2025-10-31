"""
Improved Visual Wrapper for CartPole with 5-frame stacking and temporal information.
Based on the original implementation but with enhancements.
"""

import gymnasium as gym
import numpy as np
import cv2
import os
from PIL import Image
import collections
from gymnasium import Wrapper
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


class ImprovedRGBArrayAsObservationWrapper(Wrapper):
    """
    Improved wrapper that converts CartPole state observations into visual RGB arrays
    with 5-frame stacking for temporal information and optional image saving.
    """
    
    def __init__(self, env, image_size=64, num_frames=5, test=False, image_dir="images/"):
        super(ImprovedRGBArrayAsObservationWrapper, self).__init__(env)
        
        self.image_size = image_size
        self.num_frames = num_frames
        self.test = test
        self.image_dir = image_dir
        self.i = 0
        
        # Create image directory if in test mode
        if self.test:
            os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize environment and get observation space
        obs, _ = self.env.reset()
        dummy_obs = self._render_and_resize()
        
        # Define observation space with stacked frames
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.num_frames, *dummy_obs.shape), 
            dtype=dummy_obs.dtype
        )
        
        # Deque for frame stacking
        self.dq = collections.deque(maxlen=self.num_frames)
        
        # Initialize with dummy frames
        self.reset()
    
    def _render_and_resize(self):
        """Render environment and resize to specified dimensions."""
        obs = self.env.render()
        
        # Convert to grayscale and resize
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        return obs.astype(np.uint8)
    
    def reset(self, **kwargs):
        """Reset environment and return stacked observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Get initial frame
        frame = self._render_and_resize()
        
        # Fill deque with initial frame
        for _ in range(self.num_frames):
            self.dq.append(frame)
        
        # Return stacked observation
        obs_stacked = np.stack(list(self.dq))
        return obs_stacked, info
    
    def step(self, action):
        """Step environment and return stacked observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Get new frame
        frame = self._render_and_resize()
        
        # Save image if in test mode
        if self.test:
            im_path = os.path.join(self.image_dir, f"frame_{self.i}.png")
            self.i += 1
            Image.fromarray(frame).save(im_path)
        
        # Add frame to deque (automatically maintains size)
        self.dq.append(frame)
        
        # Return stacked observation
        obs_stacked = np.stack(list(self.dq))
        
        return obs_stacked, reward, terminated, truncated, info


def create_improved_visual_cartpole(image_size=64, num_frames=5, test=False, image_dir="images/"):
    """
    Create CartPole environment with improved visual wrapper.
    
    Args:
        image_size: Size of the rendered images (default: 64x64)
        num_frames: Number of frames to stack (default: 5)
        test: Whether to save images during testing (default: False)
        image_dir: Directory to save images if test=True (default: "images/")
    
    Returns:
        Wrapped CartPole environment
    """
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env = ImprovedRGBArrayAsObservationWrapper(
        env, 
        image_size=image_size, 
        num_frames=num_frames, 
        test=test, 
        image_dir=image_dir
    )
    return env


def create_improved_visual_cartpole_with_monitor(log_dir, image_size=64, num_frames=5, test=False, image_dir="images/"):
    """
    Create CartPole environment with improved visual wrapper and monitoring.
    
    Args:
        log_dir: Directory for monitoring logs
        image_size: Size of the rendered images (default: 64x64)
        num_frames: Number of frames to stack (default: 5)
        test: Whether to save images during testing (default: False)
        image_dir: Directory to save images if test=True (default: "images/")
    
    Returns:
        Monitored CartPole environment
    """
    env = create_improved_visual_cartpole(image_size, num_frames, test, image_dir)
    env = Monitor(env, log_dir)
    return env


# Example usage and testing
if __name__ == "__main__":
    # Test the wrapper
    print("Testing Improved Visual Wrapper...")
    
    # Create environment
    env = create_improved_visual_cartpole(test=True, image_dir="test_images/")
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    
    # Test step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward}, Done={done}")
        
        if done:
            obs, info = env.reset()
            print("Environment reset")
            break
    
    print("Test completed!")
