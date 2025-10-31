# Quick Reference Guide - Improved RL Implementation

## 🚀 Quick Commands

```bash
# Run all experiments
python run_improved_experiments.py all

# Quick RL training only
python run_improved_experiments.py quick_rl

# VAE training only
python run_improved_experiments.py vae

# Progressive training (3 cycles)
python run_improved_experiments.py progressive
```

## Key Files

| File | Purpose |
|------|---------|
| `src/environments/improved_visual_wrapper.py` | 5-frame stacking visual wrapper |
| `experiments/quick_training/run_quick_experiment.py` | Progressive RL training |
| `src/models/improved_vae_training.py` | Enhanced VAE training |
| `run_improved_experiments.py` | Main experiment runner |
| `configs/quick_training_config.yaml` | RL training configuration |
| `configs/vae_config.yaml` | VAE training configuration |

## Configuration Quick Reference

### RL Training Phases
```yaml
training_phases:
  - name: "Initial Training"
    steps: 1000
    learning_rate: 3e-4
  - name: "Extended Training"
    steps: 5000
    learning_rate: 1e-4
  - name: "Fine-tuning"
    steps: 10000
    learning_rate: 5e-5
```

### VAE Settings
```yaml
vae:
  linear:
    latent_dim: 500
    epochs: 5
    batch_size: 64
  conv:
    latent_dim: 500
    epochs: 5
    batch_size: 64
```

## Expected Outputs

### Model Files
- `results/models/quick_training/phase_1_model.pkl`
- `results/models/quick_training/phase_2_model.pkl`
- `results/models/quick_training/phase_3_model.pkl`
- `results/models/quick_training/final_model.pkl`

### Generated Data
- `data/raw/cartpole_test/` - Test images
- `data/generated/` - VAE generated data
- `results/logs/quick_training/` - Training logs

## Common Modifications

### Change Training Steps
Edit `configs/quick_training_config.yaml`:
```yaml
training_phases:
  - name: "Custom Phase"
    steps: 2000  # Change this
```

### Enable pythae VAE
Edit `configs/vae_config.yaml`:
```yaml
use_pythae: true
```

### Change Image Size
Edit both config files:
```yaml
image_size: 128  # Change from 64
```

## Quick Fixes

### Import Error
```bash
cd RL_Project-with-Visual_Input
pip install -r requirements.txt
```

### Memory Issue
Reduce batch size in config:
```yaml
batch_size: 32  # Reduce from 64
```

### CUDA Error
Force CPU usage in code:
```python
device = torch.device("cpu")
```

## Performance Targets

| Phase | Steps | Expected Reward |
|-------|-------|----------------|
| Phase 1 | 1,000 | 50-100 |
| Phase 2 | 5,000 | 200-300 |
| Phase 3 | 10,000 | 400-500 |

## Key Features

- 5-frame temporal stacking
- Progressive training phases
- Model saving in .pkl format
- Multiple VAE implementations
- Performance tracking
- Automatic early stopping
- Comprehensive logging
