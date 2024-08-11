import gymnasium as gym
from scipy.io import savemat
import numpy as np
from PIL import Image

if __name__ == "__main__":
    """
    Loop to generate sample data.
    """    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    observation, info = env.reset(seed=42)
    frames = []
    for idx in range(1000):
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(np.array(Image.fromarray(env.render()).resize((64, 64), resample=Image.BILINEAR))/255)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    savemat("dataset/cartpole.mat", {"frames": np.array(frames)})
    