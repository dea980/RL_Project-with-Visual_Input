# RL Project with Visual Input - Improved Implementation

## 🚀 Overview

This is an improved implementation of the Reinforcement Learning project with Visual Input, based on the original approach but with significant enhancements. The project combines:

- **Reinforcement Learning** with PPO (Proximal Policy Optimization)
- **Variational Autoencoders (VAE)** for visual data processing
- **5-frame temporal stacking** for better temporal information
- **Progressive training** with incremental improvement
- **Multiple VAE implementations** (custom + pythae option)

## 📁 Project Structure

```
RL_Project-with-Visual_Input/
├── src/
│   ├── environments/
│   │   ├── improved_visual_wrapper.py    # Enhanced visual wrapper with 5-frame stacking
│   │   └── visual_wrapper.py             # Original visual wrapper
│   ├── models/
│   │   ├── improved_vae_training.py      # Enhanced VAE training with pythae option
│   │   ├── linear_vae.py                 # Linear VAE implementation
│   │   ├── conv_vae.py                   # Convolutional VAE implementation
│   │   └── train_vae.py                  # Original VAE training
│   └── utils/
│       └── generate_cartpole_data.py     # CartPole data generation
├── experiments/
│   ├── quick_training/
│   │   └── run_quick_experiment.py       # Quick training with progressive phases
│   ├── exp1_native/                      # Original native state experiment
│   ├── exp2_cnn_before/                  # Original CNN before VAE experiment
│   └── exp3_cnn_after/                   # Original CNN after VAE experiment
├── configs/
│   ├── quick_training_config.yaml        # Configuration for quick training
│   ├── vae_config.yaml                   # Updated VAE configuration
│   └── experiment_config.yaml            # Original experiment configuration
├── data/
│   ├── raw/                              # Raw CartPole frames
│   └── generated/                        # Generated VAE data
├── results/
│   ├── logs/                             # Training logs
│   └── models/                           # Saved models (.pkl format)
├── run_improved_experiments.py           # Main experiment runner
└── README_IMPROVED.md                    # This documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium (replacement for Gym)
- OpenCV
- Matplotlib
- PIL (Pillow)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd RL_Project-with-Visual_Input
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install optional dependencies:**
```bash
# For pythae VAE option
pip install pythae

# For classic control environments
pip install "gymnasium[classic-control]"
```

## Quick Start

### Run All Improved Experiments

```bash
python run_improved_experiments.py all
```

This will run:
1. Generate CartPole frames
2. Quick RL training with progressive phases
3. Generate test images with trained model
4. VAE training (both linear and conv)

### Run Individual Components

```bash
# Quick RL training only
python run_improved_experiments.py quick_rl

# VAE training only
python run_improved_experiments.py vae

# Progressive training (3 cycles)
python run_improved_experiments.py progressive
```

## Key Features

### 1. Enhanced Visual Wrapper

**File:** `src/environments/improved_visual_wrapper.py`

- **5-frame temporal stacking** for better temporal information
- **Grayscale conversion** with OpenCV
- **64x64 resolution** images
- **Optional image saving** during testing
- **Gymnasium compatibility**

```python
from src.environments.improved_visual_wrapper import create_improved_visual_cartpole

# Create environment with 5-frame stacking
env = create_improved_visual_cartpole(
    image_size=64,
    num_frames=5,
    test=True,  # Save images
    image_dir="test_images/"
)
```

### 2. Quick Training System

**File:** `experiments/quick_training/run_quick_experiment.py`

- **Progressive training phases:**
  - Phase 1: 1,000 steps (Initial Training)
  - Phase 2: 5,000 steps (Extended Training)
  - Phase 3: 10,000 steps (Fine-tuning)
- **Model saving in .pkl format**
- **Performance evaluation** after each phase
- **Automatic early stopping** if environment is solved

```python
from experiments.quick_training.run_quick_experiment import run_quick_training_experiment

# Run quick training
model, rewards = run_quick_training_experiment()
```

### 3. Enhanced VAE Training

**File:** `src/models/improved_vae_training.py`

- **Multiple VAE implementations:**
  - Custom Linear VAE
  - Custom Convolutional VAE
  - pythae library option
- **500 latent dimensions** (matching original approach)
- **5 epochs training** (matching original approach)
- **Beta=1.0** for KL divergence
- **MSE loss** for reconstruction

```python
from src.models.improved_vae_training import train_vae_experiment

# Train VAE models
train_vae_experiment("configs/vae_config.yaml")
```

### 4. Progressive Training

**File:** `run_improved_experiments.py`

- **Multiple training cycles** to test improvement
- **Cycle-specific model saving**
- **Performance comparison** across cycles
- **Comprehensive logging**

## Configuration

### Quick Training Configuration

**File:** `configs/quick_training_config.yaml`

```yaml
# Environment settings
image_size: 64
num_frames: 5  # 5-frame stacking

# Training phases
training_phases:
  - name: "Initial Training"
    steps: 1000
    learning_rate: 3e-4
    # ... more parameters
  - name: "Extended Training"
    steps: 5000
    learning_rate: 1e-4
    # ... more parameters
  - name: "Fine-tuning"
    steps: 10000
    learning_rate: 5e-5
    # ... more parameters

# Success threshold
success_threshold: 450
```

### VAE Configuration

**File:** `configs/vae_config.yaml`

```yaml
# VAE settings
vae:
  linear:
    latent_dim: 500
    epochs: 5
    batch_size: 64
  conv:
    latent_dim: 500
    epochs: 5
    batch_size: 64

# Model types to train
model_types: ["linear", "conv"]

# pythae option
use_pythae: false
```

## Results and Outputs

### Model Files

- **RL Models:** `results/models/quick_training/`
  - `phase_1_model.pkl` - Phase 1 model
  - `phase_2_model.pkl` - Phase 2 model
  - `phase_3_model.pkl` - Phase 3 model
  - `final_model.pkl` - Final trained model

### Generated Data

- **Test Images:** `data/raw/cartpole_test/`
- **VAE Generated Data:** `data/generated/`
- **Training Logs:** `results/logs/quick_training/`

### VAE Models

- **Linear VAE:** `data/generated/vae_linear_model.pth`
- **Conv VAE:** `data/generated/vae_conv_model.pth`
- **Comparison Images:** `data/generated/comparison_*/`

## Advanced Usage

### Custom Training Phases

You can modify the training phases in `configs/quick_training_config.yaml`:

```yaml
training_phases:
  - name: "Custom Phase"
    steps: 2000
    learning_rate: 2e-4
    n_steps: 128
    batch_size: 64
    n_epochs: 4
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
```

### Using pythae VAE

To use pythae instead of custom VAE:

1. Install pythae:
```bash
pip install pythae
```

2. Update configuration:
```yaml
use_pythae: true
```

3. Run VAE training:
```bash
python run_improved_experiments.py vae
```

### Custom VAE Parameters

Modify VAE settings in `configs/vae_config.yaml`:

```yaml
vae:
  linear:
    latent_dim: 1000  # Increase latent dimensions
    epochs: 10        # More epochs
    batch_size: 128   # Larger batch size
```

## Testing and Validation

### Test Individual Components

```python
# Test visual wrapper
from src.environments.improved_visual_wrapper import create_improved_visual_cartpole

env = create_improved_visual_cartpole(test=True)
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (5, 64, 64)

# Test VAE training
from src.models.improved_vae_training import train_vae_experiment
train_vae_experiment()
```

### Performance Monitoring

The system provides detailed performance metrics:

- **Episode rewards** for each training phase
- **Training loss curves** for VAE models
- **Model evaluation** after each phase
- **Progressive improvement** tracking

## Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Make sure you're in the project root directory
   cd RL_Project-with-Visual_Input
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Gymnasium vs Gym:**
   ```python
   # Use gymnasium instead of gym
   import gymnasium as gym
   ```

3. **CUDA/GPU Issues:**
   ```python
   # Force CPU usage
   device = torch.device("cpu")
   ```

4. **Memory Issues:**
   - Reduce batch size in configuration
   - Use smaller image sizes
   - Reduce number of training steps

### Debug Mode

Enable debug logging by modifying the configuration:

```yaml
training:
  verbose: 2  # Increase verbosity
  log_freq: 1  # Log every step
```

## Performance Expectations

### RL Training

- **Phase 1 (1K steps):** Basic learning, rewards ~50-100
- **Phase 2 (5K steps):** Improved performance, rewards ~200-300
- **Phase 3 (10K steps):** Near-optimal, rewards ~400-500

### VAE Training

- **Linear VAE:** Fast training, good for simple patterns
- **Conv VAE:** Slower training, better for complex visual patterns
- **pythae:** Professional implementation, more stable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original implementation based on the Jupyter notebook approach
- Stable-Baselines3 for RL algorithms
- PyTorch for VAE implementations
- pythae for additional VAE options
- Gymnasium for environment management

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the configuration files
3. Check the logs in `results/logs/`
4. Open an issue on GitHub

---

**Note:** This is the improved implementation. The original implementation is preserved in the main README.md file.
