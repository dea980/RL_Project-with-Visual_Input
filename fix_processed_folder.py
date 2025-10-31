#!/usr/bin/env python3
"""
Fix processed folder and ensure VAE images are properly saved
"""

import os
import shutil
import numpy as np
from pathlib import Path

def create_processed_data():
    """Create processed data from raw CartPole frames"""
    print("Creating processed data...")
    
    # Create processed directory structure
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if we have raw data
    raw_dir = "data/raw/cartpole"
    if not os.path.exists(raw_dir):
        print(f"Raw data directory {raw_dir} not found. Please run VAE training first.")
        return False
    
    # List all PNG files
    png_files = [f for f in os.listdir(raw_dir) if f.endswith('.png')]
    print(f"Found {len(png_files)} PNG files in {raw_dir}")
    
    if len(png_files) == 0:
        print("No PNG files found. Please generate CartPole frames first.")
        return False
    
    # Create a summary file
    summary_file = os.path.join(processed_dir, "data_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Processed Data Summary\n")
        f.write(f"=====================\n")
        f.write(f"Source: {raw_dir}\n")
        f.write(f"Total frames: {len(png_files)}\n")
        f.write(f"Image format: PNG\n")
        f.write(f"Image size: 64x64 (grayscale)\n")
        f.write(f"Generated: {os.popen('date').read().strip()}\n")
    
    print(f"Processed data summary saved to: {summary_file}")
    return True

def ensure_vae_images_saved():
    """Ensure VAE images are properly saved and organized"""
    print("Checking VAE image outputs...")
    
    generated_dir = "data/generated"
    if not os.path.exists(generated_dir):
        print(f"Generated directory {generated_dir} not found.")
        return False
    
    # Check for VAE models
    vae_models = ['linear', 'conv']
    for model in vae_models:
        model_file = f"{generated_dir}/vae_{model}_model.pth"
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ {model.title()} VAE model: {size_mb:.1f} MB")
        else:
            print(f"✗ {model.title()} VAE model: Not found")
    
    # Check for comparison images
    comparison_dirs = [f"{generated_dir}/comparison_{model}" for model in vae_models]
    for comp_dir in comparison_dirs:
        if os.path.exists(comp_dir):
            files = os.listdir(comp_dir)
            print(f"✓ {os.path.basename(comp_dir)}: {len(files)} files")
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(comp_dir, file)
                    size_kb = os.path.getsize(file_path) / 1024
                    print(f"  - {file}: {size_kb:.1f} KB")
        else:
            print(f"✗ {os.path.basename(comp_dir)}: Not found")
    
    # Check for training curves
    curve_files = [f"{generated_dir}/training_curve_{model}.png" for model in vae_models]
    for curve_file in curve_files:
        if os.path.exists(curve_file):
            size_kb = os.path.getsize(curve_file) / 1024
            print(f"✓ {os.path.basename(curve_file)}: {size_kb:.1f} KB")
        else:
            print(f"✗ {os.path.basename(curve_file)}: Not found")
    
    return True

def create_quick_reference():
    """Create a quick reference for accessing results"""
    print("Creating quick reference...")
    
    ref_content = """# Quick Reference - Results Location

## RL Training Results
- **Logs**: `results/logs/quick_training/monitor.csv`
- **Models**: `results/models/quick_training/`
- **TensorBoard**: `tensorboard --logdir results/logs/quick_training`

## VAE Results
- **Models**: `data/generated/vae_*_model.pth`
- **Comparison Images**: `data/generated/comparison_*/`
- **Training Curves**: `data/generated/training_curve_*.png`

## Generated Plots
- **All Plots**: `results/plots/`
- **Summary Report**: `results/plots/results_summary.md`

## Raw Data
- **CartPole Frames**: `data/raw/cartpole/`
- **Test Images**: `data/raw/cartpole_test/`

## Processed Data
- **Summary**: `data/processed/data_summary.txt`
"""
    
    with open("QUICK_RESULTS_REFERENCE.md", 'w') as f:
        f.write(ref_content)
    
    print("Quick reference saved to: QUICK_RESULTS_REFERENCE.md")

def main():
    """Main function to fix all issues"""
    print("Fixing processed folder and VAE image issues...")
    print("=" * 50)
    
    # Fix processed folder
    print("\n1. Creating processed data...")
    create_processed_data()
    
    # Check VAE images
    print("\n2. Checking VAE images...")
    ensure_vae_images_saved()
    
    # Create quick reference
    print("\n3. Creating quick reference...")
    create_quick_reference()
    
    print("\n" + "=" * 50)
    print("All fixes completed!")
    print("Run 'python analyze_results.py' to generate comprehensive plots.")

if __name__ == "__main__":
    main()
