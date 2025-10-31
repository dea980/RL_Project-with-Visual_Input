"""
Improved VAE Training - Based on original approach with pythae option.
Supports both custom VAE and pythae implementations.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.linear_vae import LinearVAE
from models.conv_vae import ConvVAE


class CustomDataset(Dataset):
    """Custom dataset for VAE training."""
    
    def __init__(self, data, device="cpu"):
        self.data = torch.tensor(data).float().to(device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def loss_function(true_batch, recon_batch, mean, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence.
    """
    recon_loss = F.mse_loss(recon_batch, true_batch)
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def train_custom_vae(model, optimizer, epoch, dataloader, beta=1.0, device="cpu"):
    """
    Train custom VAE model for one epoch.
    """
    model.train()
    train_loss = []
    
    for batch_idx, x in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Reshape data based on model type
        if isinstance(model, ConvVAE):
            # For ConvVAE: (batch, channels, height, width)
            x_ = x.unsqueeze(1) if x.dim() == 3 else x
        else:
            # For LinearVAE: flatten to (batch, features)
            x_ = x.reshape(x.shape[0], -1)
        
        # Forward pass
        recon_batch, mean, variance = model.forward(x_)
        
        # Calculate loss
        if isinstance(model, ConvVAE):
            # For ConvVAE: flatten for loss calculation
            recon_batch_flat = recon_batch.reshape(recon_batch.shape[0], -1)
            x_flat = x_.reshape(x_.shape[0], -1)
            loss = loss_function(x_flat, recon_batch_flat, mean, variance, beta)
        else:
            loss = loss_function(x_, recon_batch, mean, variance, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
    
    avg_loss = np.mean(train_loss)
    if epoch % 1 == 0:
        print(f'====> Epoch: {epoch} Training loss: {avg_loss:.4f}')
    
    return avg_loss


def train_pythae_vae(frames, config):
    """
    Train VAE using pythae library.
    """
    try:
        from pythae.models import VAE, VAEConfig
        from pythae.trainers import BaseTrainerConfig
        from pythae.pipelines.training import TrainingPipeline
        from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
    except ImportError:
        print("pythae not installed. Install with: pip install pythae")
        return None
    
    print("raining VAE with pythae...")
    
    # Prepare data
    frames_array = np.array(frames)
    if frames_array.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        frames_array = frames_array[:, np.newaxis, :, :]
    
    # Create dataset
    dataset = CustomDataset(frames_array)
    batch_size = config['vae'][model_type]['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Configure trainer
    learning_rate = float(config['vae'][model_type]['learning_rate'])
    trainer_config = BaseTrainerConfig(
        output_dir=config['output_dir'],
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=config['vae'][model_type]['epochs'],
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)}
    )
    
    # Configure model
    model_config = VAEConfig(
        input_dim=(1, 64, 64),  # Fixed image size
        latent_dim=config['vae'][model_type]['latent_dim']
    )
    
    # Create model
    model = VAE(
        model_config=model_config,
        encoder=Encoder_VAE_MLP(model_config),
        decoder=Decoder_AE_MLP(model_config)
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        training_config=trainer_config,
        model=model
    )
    
    # Train model
    pipeline(train_loader=dataloader, eval_loader=dataloader)
    
    return model


def train_custom_vae_model(frames, config, model_type="linear"):
    """
    Train custom VAE model.
    """
    print(f"Training {model_type.upper()} VAE...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    frames_array = np.array(frames)
    
    # Create dataset
    dataset = CustomDataset(frames_array, device)
    batch_size = config['vae'][model_type]['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    if model_type == "linear":
        model = LinearVAE(
            in_dim=(64, 64),  # Fixed image size
            latent_dim=config['vae'][model_type]['latent_dim'],
            device=device
        )
    elif model_type == "conv":
        model = ConvVAE(
            in_dim=(64, 64),  # Fixed image size
            latent_dim=config['vae'][model_type]['latent_dim'],
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer
    learning_rate = float(config['vae'][model_type]['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(1, config['vae'][model_type]['epochs'] + 1):
        loss = train_custom_vae(model, optimizer, epoch, dataloader, config['vae'][model_type]['beta'], device)
        losses.append(loss)
    
    return model, losses


def generate_vae_images(model, num_images=10, latent_dim=500, device="cpu"):
    """
    Generate images using trained VAE.
    """
    model.eval()
    
    # Generate random latent vectors
    z_random = torch.randn(num_images, latent_dim).to(device)
    
    # Generate images
    with torch.no_grad():
        if isinstance(model, ConvVAE):
            recon_images = model.decode(z_random)
            recon_images = recon_images.squeeze(1).cpu().numpy()  # Remove channel dimension
        else:
            recon_images = model.decode(z_random)
            recon_images = recon_images.reshape(num_images, 64, 64).cpu().numpy()
    
    return recon_images


def save_training_images(original_images, recon_images, output_dir, model_type="linear"):
    """
    Save original and reconstructed images for comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison grid
    fig, axes = plt.subplots(2, min(10, len(original_images)), figsize=(15, 6))
    if len(original_images) == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(10, len(original_images))):
        # Original image
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        if model_type == "conv" and recon_images[i].ndim == 3:
            recon_img = recon_images[i].squeeze(0)
        else:
            recon_img = recon_images[i]
        
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{model_type}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison images saved to: {output_dir}")


def train_vae_experiment(config_path="configs/vae_config.yaml"):
    """
    Run VAE training experiment with both custom and pythae options.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load frames
    frames_dir = config['frames_dir']
    frames = []
    
    print(f"Loading frames from: {frames_dir}")
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith('.png'):
            img_path = os.path.join(frames_dir, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img).astype(float) / 255.0  # Normalize
            
            # Ensure consistent shape (64x64)
            if img_array.shape != (64, 64):
                img_array = np.array(Image.fromarray(img_array).resize((64, 64)))
            
            frames.append(img_array)
    
    print(f"Loaded {len(frames)} frames")
    
    # Convert to numpy array and verify shape
    frames_array = np.array(frames)
    print(f"Frames array shape: {frames_array.shape}")
    
    if frames_array.ndim != 3 or frames_array.shape[1:] != (64, 64):
        print(f"Warning: Expected shape (N, 64, 64), got {frames_array.shape}")
        # Reshape if needed
        if frames_array.ndim == 2:
            frames_array = frames_array.reshape(1, 64, 64)
        elif frames_array.ndim == 4:
            frames_array = frames_array.squeeze()
    
    frames = frames_array
    
    # Train custom VAE models
    for model_type in config['model_types']:
        print(f"\nTraining {model_type.upper()} VAE...")
        
        # Train model
        model, losses = train_custom_vae_model(frames, config, model_type)
        
        # Save model
        model_path = os.path.join(config['output_dir'], f'vae_{model_type}_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
        
        # Generate and save images
        num_generated = 20  # Generate more images
        recon_images = generate_vae_images(
            model, 
            num_generated,
            config['vae'][model_type]['latent_dim']
        )
        
        # Save individual generated images
        generated_dir = os.path.join(config['output_dir'], f'generated_{model_type}')
        os.makedirs(generated_dir, exist_ok=True)
        
        for i, img in enumerate(recon_images):
            # Convert to 0-255 range and save as PNG
            img_uint8 = (img * 255).astype(np.uint8)
            img_path = os.path.join(generated_dir, f'generated_{model_type}_{i:03d}.png')
            Image.fromarray(img_uint8).save(img_path)
        
        print(f"Generated {num_generated} images saved to: {generated_dir}")
        
        # Save comparison images
        comparison_dir = os.path.join(config['output_dir'], f'comparison_{model_type}')
        save_training_images(frames[:10], recon_images[:10], comparison_dir, model_type)
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'{model_type.upper()} VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(config['output_dir'], f'training_curve_{model_type}.png'))
        plt.close()
    
    # Train pythae VAE if enabled
    if config.get('use_pythae', False):
        print(f"\nTraining pythae VAE...")
        pythae_model = train_pythae_vae(frames, config)
        if pythae_model:
            print("pythae VAE training completed")


if __name__ == "__main__":
    # Run VAE training experiment
    train_vae_experiment()
