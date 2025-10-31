#!/usr/bin/env python3
"""
VAE Training Experiment
Trains VAE models and saves generated images as PNG files
"""

import os
import sys
import yaml
import argparse
import subprocess
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def load_config():
    """Load VAE configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'vae_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_training_data():
    """Generate CartPole training data"""
    print("🎮 Generating CartPole training data...")
    
    data_script = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'utils', 'generate_cartpole_data.py')
    
    try:
        result = subprocess.run([
            sys.executable, data_script,
            '--episodes', '100',
            '--max_steps', '300',
            '--image_size', '64', '64',
            '--save_dir', 'data/raw/cartpole'
        ], check=True, capture_output=True, text=True)
        
        print("Training data generated successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate training data: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def train_linear_vae(config):
    """Train Linear VAE model"""
    print("\nTraining Linear VAE...")
    
    vae_script = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models', 'train_vae.py')
    data_path = 'data/raw/cartpole/cartpole_frames.npy'
    
    try:
        result = subprocess.run([
            sys.executable, vae_script,
            '--model', 'linear',
            '--epochs', str(config['vae']['linear']['epochs']),
            '--latent_dim', str(config['vae']['linear']['latent_dim']),
            '--beta', str(config['vae']['linear']['beta'])
        ], check=True, capture_output=True, text=True)
        
        print("Linear VAE training completed")
        print("Generated images saved as PNG files")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Linear VAE training failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def train_conv_vae(config):
    """Train Convolutional VAE model"""
    print("\n🔧 Training Convolutional VAE...")
    
    vae_script = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models', 'train_vae.py')
    data_path = 'data/raw/cartpole/cartpole_frames.npy'
    
    try:
        result = subprocess.run([
            sys.executable, vae_script,
            '--model', 'conv',
            '--epochs', str(config['vae']['conv']['epochs']),
            '--latent_dim', str(config['vae']['conv']['latent_dim']),
            '--beta', str(config['vae']['conv']['beta'])
        ], check=True, capture_output=True, text=True)
        
        print("Convolutional VAE training completed")
        print("Generated images saved as PNG files")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Convolutional VAE training failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def create_vae_summary():
    """Create a summary of VAE training results"""
    print("\nCreating VAE training summary...")
    
    # Find all VAE output directories
    vae_dirs = []
    generated_dir = 'data/generated'
    
    if os.path.exists(generated_dir):
        for item in os.listdir(generated_dir):
            if item.startswith('vae_'):
                vae_dirs.append(os.path.join(generated_dir, item))
    
    if not vae_dirs:
        print("No VAE output directories found")
        return
    
    print(f"Found {len(vae_dirs)} VAE training runs:")
    for vae_dir in vae_dirs:
        print(f"  - {vae_dir}")
        
        # Count generated images
        generated_imgs_dir = os.path.join(vae_dir, 'generated')
        if os.path.exists(generated_imgs_dir):
            png_files = [f for f in os.listdir(generated_imgs_dir) if f.endswith('.png')]
            print(f"    Generated {len(png_files)} PNG images")
        
        # Check for training info
        info_file = os.path.join(vae_dir, 'training_info.txt')
        if os.path.exists(info_file):
            print(f"    Training info available")
    
    print("\n🎉 VAE training summary complete!")
    print("Check data/generated/ for all VAE outputs and PNG images")

def main():
    parser = argparse.ArgumentParser(description='Run VAE training experiment')
    parser.add_argument('--models', nargs='+', choices=['linear', 'conv', 'all'],
                       default=['all'], help='Which VAE models to train')
    parser.add_argument('--skip_data_gen', action='store_true',
                       help='Skip data generation (use existing data)')
    
    args = parser.parse_args()
    
    print("VAE Training Experiment")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: {config['vae']}")
    
    # Create necessary directories
    os.makedirs('data/raw/cartpole', exist_ok=True)
    os.makedirs('data/generated', exist_ok=True)
    
    # Generate training data if needed
    if not args.skip_data_gen:
        if not generate_training_data():
            print("Failed to generate training data. Exiting.")
            return 1
    else:
        print("Skipping data generation (using existing data)")
    
    # Determine which models to train
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['linear', 'conv']
    
    # Train VAE models
    results = {}
    
    for model_type in models_to_train:
        if model_type == 'linear':
            results['linear'] = train_linear_vae(config)
        elif model_type == 'conv':
            results['conv'] = train_conv_vae(config)
    
    # Create summary
    create_vae_summary()
    
    # Print final results
    print(f"\n{'='*50}")
    print("VAE TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for model_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{model_type.upper()} VAE: {status}")
    
    all_success = all(results.values())
    if all_success:
        print("\nAll VAE models trained successfully!")
        print("Generated PNG images are saved in data/generated/")
        return 0
    else:
        print("\n Warning : Some VAE models failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
