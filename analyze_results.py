#!/usr/bin/env python3
"""
Comprehensive Results Analysis and Visualization
Analyzes RL training logs, VAE results, and generates plots
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import yaml
import glob
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_monitor_data(log_dir):
    """Load monitor.csv data from training logs"""
    monitor_files = glob.glob(os.path.join(log_dir, "**/monitor.csv"), recursive=True)
    
    all_data = []
    for file in monitor_files:
        try:
            df = pd.read_csv(file, skiprows=1)  # Skip metadata line
            df['file'] = os.path.basename(os.path.dirname(file))
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if not all_data:
        print("No monitor data found")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def plot_training_curves(data, save_dir="results/plots"):
    """Plot training curves from monitor data"""
    os.makedirs(save_dir, exist_ok=True)
    
    if data is None or data.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards over Time
    axes[0, 0].plot(data['r'], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Length over Time
    axes[0, 1].plot(data['l'], alpha=0.7, linewidth=1, color='orange')
    axes[0, 1].set_title('Episode Length')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Average Rewards
    window = min(100, len(data) // 10)
    if window > 1:
        rolling_rewards = data['r'].rolling(window=window).mean()
        axes[1, 0].plot(rolling_rewards, linewidth=2, color='green')
        axes[1, 0].set_title(f'Rolling Average Rewards (window={window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(data['r'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rl_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_dir}/rl_training_curves.png")

def plot_vae_results(save_dir="results/plots"):
    """Plot VAE training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for VAE comparison images
    vae_dirs = ['data/generated/comparison_linear', 'data/generated/comparison_conv']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('VAE Results Comparison', fontsize=16, fontweight='bold')
    
    for i, vae_dir in enumerate(vae_dirs):
        if os.path.exists(vae_dir):
            comparison_file = os.path.join(vae_dir, f'comparison_{os.path.basename(vae_dir)}.png')
            if os.path.exists(comparison_file):
                # Load and display the comparison image
                img = plt.imread(comparison_file)
                axes[i].imshow(img)
                axes[i].set_title(f'{os.path.basename(vae_dir).replace("_", " ").title()} VAE')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No comparison image found\nin {vae_dir}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{os.path.basename(vae_dir).replace("_", " ").title()} VAE')
        else:
            axes[i].text(0.5, 0.5, f'Directory not found:\n{vae_dir}', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{os.path.basename(vae_dir).replace("_", " ").title()} VAE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"VAE results saved to: {save_dir}/vae_comparison.png")

def plot_training_curves_vae(save_dir="results/plots"):
    """Plot VAE training curves if available"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for training curve images
    curve_files = glob.glob("data/generated/training_curve_*.png")
    
    if curve_files:
        fig, axes = plt.subplots(1, len(curve_files), figsize=(6*len(curve_files), 5))
        if len(curve_files) == 1:
            axes = [axes]
        
        fig.suptitle('VAE Training Curves', fontsize=16, fontweight='bold')
        
        for i, curve_file in enumerate(curve_files):
            img = plt.imread(curve_file)
            axes[i].imshow(img)
            model_type = os.path.basename(curve_file).replace('training_curve_', '').replace('.png', '')
            axes[i].set_title(f'{model_type.title()} VAE Training')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vae_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"VAE training curves saved to: {save_dir}/vae_training_curves.png")

def generate_summary_report(data, save_dir="results/plots"):
    """Generate a summary report of all results"""
    os.makedirs(save_dir, exist_ok=True)
    
    report = []
    report.append("# RL Project Results Summary")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # RL Results
    if data is not None and not data.empty:
        report.append("## RL Training Results")
        report.append(f"- Total Episodes: {len(data)}")
        report.append(f"- Average Reward: {data['r'].mean():.2f} ± {data['r'].std():.2f}")
        report.append(f"- Max Reward: {data['r'].max():.2f}")
        report.append(f"- Min Reward: {data['r'].min():.2f}")
        report.append(f"- Average Episode Length: {data['l'].mean():.2f} ± {data['l'].std():.2f}")
        report.append("")
    
    # VAE Results
    report.append("## VAE Training Results")
    vae_models = ['linear', 'conv']
    for model in vae_models:
        model_file = f"data/generated/vae_{model}_model.pth"
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            report.append(f"- {model.title()} VAE: ✓ Trained (Model size: {size_mb:.1f} MB)")
        else:
            report.append(f"- {model.title()} VAE: ✗ Not found")
    
    report.append("")
    
    # Generated Images
    report.append("## Generated Images")
    comparison_dirs = ['data/generated/comparison_linear', 'data/generated/comparison_conv']
    for comp_dir in comparison_dirs:
        if os.path.exists(comp_dir):
            files = os.listdir(comp_dir)
            report.append(f"- {os.path.basename(comp_dir)}: {len(files)} files")
        else:
            report.append(f"- {os.path.basename(comp_dir)}: Not found")
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(save_dir, 'results_summary.md'), 'w') as f:
        f.write(report_text)
    
    print(f"Summary report saved to: {save_dir}/results_summary.md")
    print("\n" + "="*50)
    print(report_text)
    print("="*50)

def main():
    """Main analysis function"""
    print("Analyzing RL Project Results...")
    
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Load RL training data
    print("Loading RL training data...")
    rl_data = load_monitor_data("results/logs")
    
    # Generate plots
    print("Generating RL training curves...")
    plot_training_curves(rl_data)
    
    print("Generating VAE results...")
    plot_vae_results()
    
    print("Generating VAE training curves...")
    plot_training_curves_vae()
    
    print("Generating summary report...")
    generate_summary_report(rl_data)
    
    print("\nAnalysis complete! Check results/plots/ for all generated plots.")

if __name__ == "__main__":
    main()