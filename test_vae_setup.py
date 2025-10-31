#!/usr/bin/env python3
"""
Test script to verify VAE setup and data generation
"""

import os
import sys
import subprocess

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("PyTorch")
    except ImportError as e:
        print(f"PyTorch: {e}")
        return False
    
    try:
        import gym
        print("Gym")
    except ImportError as e:
        print(f"Gym: {e}")
        return False
    
    try:
        import stable_baselines3
        print("Stable-Baselines3")
    except ImportError as e:
        print(f"Stable-Baselines3: {e}")
        return False
    
    try:
        import cv2
        print("OpenCV")
    except ImportError as e:
        print(f"OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print("PIL")
    except ImportError as e:
        print(f"PIL: {e}")
        return False
    
    return True

def test_data_generation():
    """Test CartPole data generation"""
    print("\n Testing CartPole data generation...")
    
    try:
        result = subprocess.run([
            sys.executable, 'src/utils/generate_cartpole_data.py',
            '--episodes', '5',  # Small test
            '--max_steps', '50',
            '--image_size', '64', '64'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(" Data generation successful")
            return True
        else:
            print(f" Data generation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(" Data generation timed out")
        return False
    except Exception as e:
        print(f" Data generation error: {e}")
        return False

def test_vae_training():
    """Test VAE training (quick test)"""
    print("\nTesting VAE training...")
    
    try:
        result = subprocess.run([
            sys.executable, 'src/models/train_vae.py',
            '--model', 'linear',
            '--epochs', '2',  # Very short test
            '--latent_dim', '8'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(" VAE training successful")
            return True
        else:
            print(f" VAE training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("VAE training timed out")
        return False
    except Exception as e:
        print(f"VAE training error: {e}")
        return False

def check_outputs():
    """Check if outputs were generated"""
    print("\nChecking outputs...")
    
    # Check data directory
    data_dir = 'data/raw/cartpole'
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"Data directory exists with {len(files)} files")
        if 'cartpole_frames.npy' in files:
            print("CartPole frames file found")
        else:
            print("CartPole frames file not found")
    else:
        print("Data directory not found")
    
    # Check generated directory
    generated_dir = 'data/generated'
    if os.path.exists(generated_dir):
        vae_dirs = [d for d in os.listdir(generated_dir) if d.startswith('vae_')]
        print(f"Generated directory exists with {len(vae_dirs)} VAE runs")
        
        for vae_dir in vae_dirs:
            vae_path = os.path.join(generated_dir, vae_dir)
            generated_imgs = os.path.join(vae_path, 'generated')
            if os.path.exists(generated_imgs):
                png_files = [f for f in os.listdir(generated_imgs) if f.endswith('.png')]
                print(f"    {vae_dir}: {len(png_files)} PNG images")
    else:
        print("Generated directory not found")

def main():
    print("VAE Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\nImport test failed. Install missing packages:")
        print("pip install -r requirements.txt")
        return 1
    
    # Test data generation
    if not test_data_generation():
        print("\n Data generation test failed")
        return 1
    
    # Test VAE training
    if not test_vae_training():
        print("\n VAE training test failed")
        return 1
    
    # Check outputs
    check_outputs()
    
    print("\nAll tests passed!")
    print("\nYou can now run the full experiments:")
    print("  python run_all_experiments.py")
    print("  python experiments/vae_training/run_vae_experiment.py")
    print("  python visualize_vae_results.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
