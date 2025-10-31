#!/usr/bin/env python3
"""
Enhanced VAE training script with proper image saving
"""

import os
import sys
import yaml
import torch as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conv_vae import ConvVAE
from linear_vae import LinearVAE
from dataset import CustomDataset

def load_config():
    """Load VAE configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'vae_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def loss_function(true_batch, recon_batch, mean, logvar, beta):
    """
    Minimize the reconstruction loss + KL divergence
    """
    recon_loss = F.mse_loss(recon_batch, true_batch)
    kl_div = -0.5 * T.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div

def train_vae(model, optimizer, epoch, dataloader, beta, device, model_type='linear'):
    """
    Performs one iteration of training using examples from dataloader.
    """
    model.train()
    train_loss = []
    for _, x in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Handle different input shapes for linear vs conv VAE
        if model_type == 'linear':
            x_ = x.reshape(x.shape[0], -1)
        else:  # conv
            x_ = x  # Keep as (batch, channels, height, width)
            
        recon_batch, mean, variance = model.forward(x_)
        
        # For loss calculation, flatten conv output
        if model_type == 'conv':
            x_flat = x_.reshape(x_.shape[0], -1)
            recon_flat = recon_batch.reshape(recon_batch.shape[0], -1)
        else:
            x_flat = x_
            recon_flat = recon_batch
            
        loss = loss_function(x_flat, recon_flat, mean, variance, beta)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    avg_loss = np.mean(train_loss)
    if epoch % 1 == 0:
        print(f'====> Epoch: {epoch} Training loss: {avg_loss:.4f}')
    
    return model, avg_loss

def generate_images(model, z_random, device):
    """Generate images from random latent vectors"""
    model.eval()
    with T.no_grad():
        res = model.decode(z_random.squeeze())
        return res.reshape(z_random.shape[0], 64, 64).detach().cpu().numpy()

def save_images_as_png(imgs, base_path, prefix="generated"):
    """Save images as PNG files"""
    os.makedirs(base_path, exist_ok=True)
    
    for idx, img in enumerate(imgs):
        # Normalize to 0-255 range
        img_normalized = (img * 255).astype(np.uint8)
        
        # Save as PNG
        img_path = os.path.join(base_path, f"{prefix}_{idx:03d}.png")
        Image.fromarray(img_normalized, mode='L').save(img_path)
        print(f"Saved: {img_path}")

def save_training_images(imgs, base_path, epoch):
    """Save training progress images"""
    epoch_path = os.path.join(base_path, f"epoch_{epoch:03d}")
    os.makedirs(epoch_path, exist_ok=True)
    
    for idx, img in enumerate(imgs):
        img_normalized = (img * 255).astype(np.uint8)
        img_path = os.path.join(epoch_path, f"train_{idx:03d}.png")
        Image.fromarray(img_normalized, mode='L').save(img_path)

def plot_training_curve(losses, save_path):
    """Plot and save training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('VAE Training Progress')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_grid(original_imgs, reconstructed_imgs, save_path, n_samples=8):
    """Create comparison grid of original vs reconstructed images"""
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 5))
    
    for i in range(n_samples):
        # Original images
        axes[0, i].imshow(original_imgs[i], cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed_imgs[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_vae_model(model_type='linear', epochs=30, latent_dim=32, beta=1.0, data_path=None):
    """Main VAE training function"""
    print(f"Starting VAE training ({model_type})")
    print(f"Epochs: {epochs}, Latent dim: {latent_dim}, Beta: {beta}")
    
    # Load configuration
    config = load_config()
    
    # Setup device
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/generated/vae_{model_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        frames = np.load(data_path)
        print(f"Loaded {len(frames)} frames")
    else:
        print("Warning: No data found, generating CartPole frames...")
        # Import here to avoid circular imports
        from utils.generate_cartpole_data import generate_cartpole_frames
        frames = generate_cartpole_frames(n_episodes=50, max_steps=200)
        frames = frames.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Create model
    img_dim = (64, 64)
    if model_type == 'linear':
        model = LinearVAE(img_dim, latent_dim, device)
    elif model_type == 'conv':
        model = ConvVAE(img_dim, latent_dim, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dataset and dataloader
    dataset = CustomDataset(frames, device)
    dataloader = T.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    # For convolutional VAE, we need to reshape data to (batch, channels, height, width)
    if model_type == 'conv':
        # Reshape frames from (N, H, W) to (N, 1, H, W) for grayscale
        frames = frames.reshape(frames.shape[0], 1, frames.shape[1], frames.shape[2])
        dataset = CustomDataset(frames, device)
        dataloader = T.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Training loop
    losses = []
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        model, loss = train_vae(model, optimizer, epoch, dataloader, beta, device, model_type)
        losses.append(loss)
        
        # Save training progress images every 5 epochs
        if epoch % 5 == 0:
            with T.no_grad():
                sample_batch = next(iter(dataloader))
                if model_type == 'linear':
                    sample_batch = sample_batch.reshape(sample_batch.shape[0], -1)
                    recon_batch, _, _ = model.forward(sample_batch)
                    recon_imgs = recon_batch.reshape(sample_batch.shape[0], 64, 64).cpu().numpy()
                else:  # conv
                    recon_batch, _, _ = model.forward(sample_batch)
                    recon_imgs = recon_batch.squeeze(1).cpu().numpy()  # Remove channel dimension
                save_training_images(recon_imgs, output_dir, epoch)
    
    # Save final model
    model_path = os.path.join(output_dir, f"vae_{model_type}_model.pth")
    T.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Generate final images
    print("Generating final images...")
    normal_distribution = T.distributions.normal.Normal(T.Tensor([0.0]), T.Tensor([1.0]))
    z_random = normal_distribution.sample((20, latent_dim)).to(device)
    generated_imgs = generate_images(model, z_random, device)
    
    # Save generated images as PNG
    save_images_as_png(generated_imgs, os.path.join(output_dir, "generated"), "vae_generated")
    
    # Create comparison with original data
    sample_original = frames[:8]
    sample_reconstructed = generated_imgs[:8]
    
    # Ensure original data is 2D for display
    if len(sample_original.shape) == 4:  # (N, C, H, W)
        sample_original = sample_original.squeeze(1)  # Remove channel dimension
    
    create_comparison_grid(
        sample_original, 
        sample_reconstructed, 
        os.path.join(output_dir, "comparison_grid.png")
    )
    
    # Plot training curve
    plot_training_curve(losses, os.path.join(output_dir, "training_curve.png"))
    
    # Save training info
    info = {
        'model_type': model_type,
        'epochs': epochs,
        'latent_dim': latent_dim,
        'beta': beta,
        'final_loss': losses[-1],
        'device': str(device),
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, "training_info.txt"), 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"VAE training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated {len(generated_imgs)} images")
    
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--model', choices=['linear', 'conv'], default='linear',
                       help='VAE model type')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for KL divergence')
    
    args = parser.parse_args()
    
    # Train VAE
    model, output_dir = train_vae_model(
        model_type=args.model,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        beta=args.beta
    )
    
    print(f"\n VAE training complete!")
    print(f"Check the generated images in: {output_dir}")

if __name__ == "__main__":
    main()
