#!/usr/bin/env python3
"""
Visualize VAE training results and generated images
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from datetime import datetime

def find_vae_outputs():
    """Find all VAE output directories"""
    generated_dir = 'data/generated'
    vae_dirs = []
    
    if os.path.exists(generated_dir):
        for item in os.listdir(generated_dir):
            if item.startswith('vae_'):
                vae_dirs.append(os.path.join(generated_dir, item))
    
    return sorted(vae_dirs)

def load_training_info(vae_dir):
    """Load training information from a VAE directory"""
    info_file = os.path.join(vae_dir, 'training_info.txt')
    info = {}
    
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    info[key.strip()] = value.strip()
    
    return info

def create_image_gallery(vae_dir, max_images=20):
    """Create a gallery of generated images"""
    generated_dir = os.path.join(vae_dir, 'generated')
    
    if not os.path.exists(generated_dir):
        print(f" No generated images found in {generated_dir}")
        return None
    
    # Find all PNG files
    png_files = glob.glob(os.path.join(generated_dir, '*.png'))
    png_files = sorted(png_files)[:max_images]
    
    if not png_files:
        print(f" No PNG files found in {generated_dir}")
        return None
    
    # Load images
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        images.append(np.array(img))
    
    # Create gallery
    n_images = len(images)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Image {i+1}')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_curves(vae_dirs):
    """Plot training curves for all VAE models"""
    n_dirs = len(vae_dirs)
    if n_dirs == 0:
        return None
    
    # Create subplots based on number of directories
    if n_dirs == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, min(n_dirs, 4), figsize=(15, 6))
        if n_dirs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_dirs > 1 else [axes]
    
    for i, vae_dir in enumerate(vae_dirs):
        if i >= len(axes):
            break
            
        info = load_training_info(vae_dir)
        model_type = info.get('model_type', 'unknown')
        
        # Look for training curve
        curve_file = os.path.join(vae_dir, 'training_curve.png')
        if os.path.exists(curve_file):
            img = Image.open(curve_file)
            axes[i].imshow(img)
            axes[i].set_title(f'{model_type.upper()} VAE Training Curve')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No training curve\nfound for {model_type}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_type.upper()} VAE')
    
    plt.tight_layout()
    return fig

def create_comparison_grid(vae_dirs):
    """Create comparison grid of different VAE models"""
    if len(vae_dirs) < 2:
        print("Need at least 2 VAE models for comparison")
        return None
    
    # Load sample images from each model
    sample_images = {}
    
    for vae_dir in vae_dirs:
        info = load_training_info(vae_dir)
        model_type = info.get('model_type', 'unknown')
        
        generated_dir = os.path.join(vae_dir, 'generated')
        if os.path.exists(generated_dir):
            png_files = glob.glob(os.path.join(generated_dir, '*.png'))
            if png_files:
                # Load first few images
                sample_images[model_type] = []
                for png_file in sorted(png_files)[:4]:  # Take first 4 images
                    img = Image.open(png_file)
                    sample_images[model_type].append(np.array(img))
    
    if not sample_images:
        print(" No sample images found for comparison")
        return None
    
    # Create comparison grid
    n_models = len(sample_images)
    n_samples = min(4, min(len(imgs) for imgs in sample_images.values()))
    
    fig, axes = plt.subplots(n_models, n_samples, figsize=(15, 4*n_models))
    
    if n_models == 1:
        axes = [axes] if n_samples == 1 else axes
    else:
        axes = axes.reshape(n_models, n_samples)
    
    for i, (model_type, images) in enumerate(sample_images.items()):
        for j in range(n_samples):
            if j < len(images):
                axes[i, j].imshow(images[j], cmap='gray')
                axes[i, j].set_title(f'{model_type.upper()} - Sample {j+1}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    return fig

def print_vae_summary(vae_dirs):
    """Print summary of VAE training results"""
    print("\n" + "="*80)
    print("VAE TRAINING RESULTS SUMMARY")
    print("="*80)
    
    for vae_dir in vae_dirs:
        info = load_training_info(vae_dir)
        model_type = info.get('model_type', 'unknown')
        
        print(f"\n📁 {model_type.upper()} VAE Results:")
        print(f"   Directory: {vae_dir}")
        
        # Training info
        for key, value in info.items():
            if key != 'model_type':
                print(f"   {key}: {value}")
        
        # Count generated images
        generated_dir = os.path.join(vae_dir, 'generated')
        if os.path.exists(generated_dir):
            png_files = glob.glob(os.path.join(generated_dir, '*.png'))
            print(f"   Generated PNG images: {len(png_files)}")
        
        # Check for other outputs
        outputs = []
        if os.path.exists(os.path.join(vae_dir, 'comparison_grid.png')):
            outputs.append('comparison_grid.png')
        if os.path.exists(os.path.join(vae_dir, 'training_curve.png')):
            outputs.append('training_curve.png')
        if os.path.exists(os.path.join(vae_dir, f'vae_{model_type}_model.pth')):
            outputs.append(f'vae_{model_type}_model.pth')
        
        if outputs:
            print(f"   Additional outputs: {', '.join(outputs)}")

def main():
    print("VAE Results Visualization")
    print("=" * 50)
    
    # Find VAE outputs
    vae_dirs = find_vae_outputs()
    
    if not vae_dirs:
        print("No VAE output directories found!")
        print("Run the VAE training experiment first:")
        print("  python experiments/vae_training/run_vae_experiment.py")
        return 1
    
    print(f"Found {len(vae_dirs)} VAE training runs:")
    for vae_dir in vae_dirs:
        print(f"  - {vae_dir}")
    
    # Create output directory for visualizations
    viz_dir = 'results/plots/vae_visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Print summary
    print_vae_summary(vae_dirs)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # Training curves
    print(" Creating training curves...")
    fig_curves = plot_training_curves(vae_dirs)
    if fig_curves:
        fig_curves.savefig(os.path.join(viz_dir, 'vae_training_curves.png'), 
                          dpi=300, bbox_inches='tight')
        plt.close(fig_curves)
    
    # Comparison grid
    print("Creating comparison grid...")
    fig_comparison = create_comparison_grid(vae_dirs)
    if fig_comparison:
        fig_comparison.savefig(os.path.join(viz_dir, 'vae_comparison_grid.png'), 
                              dpi=300, bbox_inches='tight')
        plt.close(fig_comparison)
    
    # Individual galleries
    print("Creating image galleries...")
    for vae_dir in vae_dirs:
        info = load_training_info(vae_dir)
        model_type = info.get('model_type', 'unknown')
        
        fig_gallery = create_image_gallery(vae_dir)
        if fig_gallery:
            gallery_path = os.path.join(viz_dir, f'{model_type}_vae_gallery.png')
            fig_gallery.savefig(gallery_path, dpi=300, bbox_inches='tight')
            plt.close(fig_gallery)
            print(f"    Saved {model_type} gallery to {gallery_path}")
    
    print(f"\nVisualization complete!")
    print(f"All visualizations saved to: {viz_dir}")
    print(f"Generated PNG images are in: data/generated/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
