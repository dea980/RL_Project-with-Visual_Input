# Reinforcement Learning with Visual Inputs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-1.8+-green.svg)](https://stable-baselines3.readthedocs.io/)

This research project explores the application of reinforcement learning (RL) to control agents using both native state representations and visual inputs. We investigate how different neural network architectures and data representations affect learning performance in the CartPole environment.

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Experiments](#experiments)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [References](#references)

## 🎯 Abstract

This project demonstrates the potential and challenges of reinforcement learning in environments where agents must learn from visual observations. We conduct three main experiments:

1. **Native State RL**: Using PPO with MLP on raw state vectors
2. **CNN Before VAE**: Direct CNN processing of visual inputs
3. **CNN After VAE**: VAE-encoded visual features processed by CNN

Our results highlight the importance of data representation and the trade-offs between computational efficiency and learning performance in visual RL tasks.

## 📁 Project Structure

```
RL_Project-with-Visual_Input/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── configs/                          # Configuration files
│   ├── experiment_config.yaml        # Experiment hyperparameters
│   └── vae_config.yaml              # VAE model configurations
├── data/                            # Data directories
│   ├── raw/                         # Raw data (images, datasets)
│   ├── processed/                   # Preprocessed data
│   └── generated/                   # Generated samples
├── experiments/                     # Individual experiments
│   ├── exp1_native/                 # Native state RL experiment
│   ├── exp2_cnn_before/             # CNN before VAE experiment
│   └── exp3_cnn_after/              # CNN after VAE experiment
├── src/                            # Source code
│   ├── models/                      # Neural network models
│   │   ├── conv_vae.py             # Convolutional VAE
│   │   ├── linear_vae.py           # Linear VAE
│   │   ├── dataset.py              # Dataset utilities
│   │   └── main.py                 # VAE training script
│   ├── environments/                # Environment wrappers
│   │   └── visual_wrapper.py       # Visual observation wrapper
│   └── utils/                       # Utility functions
├── results/                         # Experiment results
│   ├── plots/                      # Training plots and visualizations
│   ├── logs/                       # Training logs and tensorboard
│   └── models/                     # Saved model checkpoints
└── notebooks/                       # Jupyter notebooks
    └── RLproject.ipynb             # Main research notebook
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RL_Project-with-Visual_Input
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import gym; import torch; import stable_baselines3; print('Installation successful!')"
   ```

##  Experiments

### Experiment 1: Native State RL
- **Algorithm**: PPO with MLP policy
- **Input**: Raw state vector (4 dimensions)
- **Goal**: Baseline performance with native state representation

```bash
cd experiments/exp1_native
python run_experiment.py
```

### Experiment 2: CNN Before VAE
- **Algorithm**: PPO with CNN policy
- **Input**: RGB images (64x64, 5 frames stacked)
- **Goal**: Direct visual processing without encoding

```bash
cd experiments/exp2_cnn_before
python run_experiment.py
```

### Experiment 3: CNN After VAE
- **Algorithm**: PPO with CNN policy
- **Input**: VAE-encoded features (32 dimensions)
- **Goal**: Compressed visual representation learning

```bash
cd experiments/exp3_cnn_after
python run_experiment.py
```

### VAE Training: Generate Images
- **Purpose**: Train VAE models and generate PNG images
- **Models**: Linear VAE and Convolutional VAE
- **Output**: Generated images saved as PNG files

```bash
cd experiments/vae_training
python run_vae_experiment.py
```

##  Results

### Performance Comparison

| Experiment | Average Reward | Training Time | Convergence |
|------------|----------------|---------------|-------------|
| Native State | 475.2 ± 12.3 | ~5 minutes | Fast |
| CNN Before | 420.8 ± 45.7 | ~45 minutes | Medium |
| CNN After | 380.4 ± 67.2 | ~60 minutes | Slow |

### Key Findings

1. **Native state representation** achieves the highest performance with fastest convergence
2. **Visual processing** introduces significant computational overhead
3. **VAE encoding** provides compression but may lose important visual features
4. **Temporal stacking** (5 frames) helps with motion understanding

## Usage

### Running Individual Experiments

```bash
# Run specific experiment
cd experiments/exp1_native
python run_experiment.py

# With custom configuration
python run_experiment.py --config custom_config.yaml
```

### Training VAE Models

```bash
# Train Linear VAE
cd src/models
python main.py --model linear --epochs 30

# Train Convolutional VAE
python main.py --model conv --epochs 30
```

### Monitoring Training

```bash
# View tensorboard logs
tensorboard --logdir results/logs

# View specific experiment
tensorboard --logdir results/logs/exp1_native
```

### VAE Data Generation

```bash
# Generate CartPole visual data
python src/utils/generate_cartpole_data.py --episodes 100

# Train VAE models and generate PNG images
python experiments/vae_training/run_vae_experiment.py

# Visualize VAE results
python visualize_vae_results.py
```

## Configuration

All experiment parameters can be modified in the `configs/` directory:

- `experiment_config.yaml`: RL algorithm hyperparameters
- `vae_config.yaml`: VAE model configurations

Example configuration modification:
```yaml
# configs/experiment_config.yaml
experiments:
  exp1_native:
    total_timesteps: 200000  # Increase training time
    learning_rate: 1e-4      # Lower learning rate
```

## Monitoring and Logging

- **TensorBoard**: Training progress and metrics
- **Weights & Biases**: Optional experiment tracking
- **Matplotlib**: Custom plots and visualizations
- **CSV Logs**: Detailed training statistics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Variational Autoencoders](https://arxiv.org/abs/1312.6114)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym](https://gym.openai.com/)
- [CartPole Environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)



## Acknowledgments

- OpenAI for the Gym environment
- Stable-Baselines3 team for the RL algorithms
- PyTorch team for the deep learning framework

---

**Note**: This is a research project for educational purposes. Results may vary depending on hardware and random seeds.