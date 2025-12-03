# RL Training Guide for GO-MELT

This guide explains how to train a reinforcement learning agent to control laser power in GO-MELT simulations.

## Overview

The RL training system consists of:
1. **Environment Wrapper** (`go_melt/go_melt_env.py`): Gym-compatible environment interface
2. **Training Script** (`train_rl.py`): Script to train RL agents using stable-baselines3
3. **RL Controller** (`go_melt/rl_controller.py`): Controller that can use trained models

## Installation

Install the required dependencies:

```bash
pip install stable-baselines3[extra] gymnasium
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train an RL Agent

Train a PPO agent (default):

```bash
python train_rl.py examples/rl.json --output-dir ./rl_training --total-timesteps 100000
```

Train with different algorithms:

```bash
# PPO (default)
python train_rl.py examples/rl.json --algorithm PPO --total-timesteps 100000

# SAC (better for continuous control)
python train_rl.py examples/rl.json --algorithm SAC --total-timesteps 100000

# TD3 (another continuous control algorithm)
python train_rl.py examples/rl.json --algorithm TD3 --total-timesteps 100000
```

### 2. Use the Trained Model

After training, update your configuration file to use the trained model:

```json
{
    "power_controller": {
        "type": "rl",
        "target_temperature": 3000.0,
        "update_interval": 10,
        "controller_params": {
            "power_min": 0.0,
            "power_max": 500.0,
            "rl_model_path": "./rl_training/final_model.zip"
        }
    }
}
```

Then run the simulation as usual:

```bash
python go_melt/go_melt.py 0 examples/rl.json
```

## Training Options

### Basic Options

- `--config-file`: Path to GO-MELT configuration JSON file (required)
- `--output-dir`: Directory to save training results (default: `./rl_training`)
- `--algorithm`: RL algorithm: PPO, SAC, or TD3 (default: PPO)
- `--total-timesteps`: Total number of training timesteps (default: 100000)
- `--n-envs`: Number of parallel environments (default: 1)
- `--device-id`: GPU device ID, use -1 for CPU (default: 0)

### Environment Options

- `--target-temp`: Target temperature in Kelvin (default: 3000.0)
- `--power-min`: Minimum power in Watts (default: 0.0)
- `--power-max`: Maximum power in Watts (default: 500.0)
- `--base-power`: Base power in Watts (default: 285.0)
- `--update-interval`: Steps between power updates (default: 10)
- `--max-steps`: Maximum steps per episode (default: None, run full simulation)

### Example: Full Training Command

```bash
python train_rl.py examples/rl.json \
    --output-dir ./rl_training \
    --algorithm SAC \
    --total-timesteps 500000 \
    --n-envs 4 \
    --device-id 0 \
    --target-temp 3000.0 \
    --power-min 0.0 \
    --power-max 500.0 \
    --base-power 285.0 \
    --update-interval 10
```

## Training Output

The training script creates the following directory structure:

```
rl_training/
├── final_model.zip          # Final trained model
├── best_model/              # Best model (based on evaluation)
│   └── best_model.zip
├── checkpoints/             # Periodic checkpoints
│   ├── rl_model_10000_steps.zip
│   ├── rl_model_20000_steps.zip
│   └── ...
├── tensorboard/             # TensorBoard logs
├── monitor/                 # Training statistics
├── evaluations/             # Evaluation results
├── training_config.json     # Training configuration
└── training_info.json       # Training metadata
```

## Monitoring Training

### TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir ./rl_training/tensorboard
```

Then open http://localhost:6006 in your browser.

### Training Statistics

Training statistics are saved in `monitor/` directory and can be analyzed with:

```python
from stable_baselines3.common.monitor import load_results
import pandas as pd

df = load_results("./rl_training/monitor")
print(df.describe())
```

## Environment Details

### Observation Space

- **Shape**: `(1,)`
- **Type**: `Box[0.0, 1.0]`
- **Description**: Normalized maximum temperature from Level 3 (0-5000K mapped to 0-1)

### Action Space

- **Shape**: `(1,)`
- **Type**: `Box[-1.0, 1.0]`
- **Description**: Normalized power adjustment (-1 to +1, maps to ±50% of power range)

### Reward Function

The reward is the negative squared error from target temperature:

```
reward = -scale * (target_temp - current_temp)²
```

This encourages the agent to maintain temperature close to the target.

## Tips for Training

1. **Start with shorter episodes**: Use `--max-steps` to limit episode length during initial training
2. **Use SAC or TD3 for continuous control**: These algorithms often work better than PPO for continuous action spaces
3. **Increase training time**: More timesteps generally lead to better performance
4. **Tune hyperparameters**: Adjust learning rate, batch size, etc. in `train_rl.py` if needed
5. **Monitor with TensorBoard**: Watch training curves to detect issues early

## Current Limitations

The current environment uses a simplified temperature dynamics model for faster training. For production use, you may want to:

1. Integrate with the full GO-MELT simulation for more realistic training
2. Add more observations (e.g., temperature history, power history)
3. Use more sophisticated reward functions
4. Implement curriculum learning

## Troubleshooting

### Import Errors

If you get import errors for stable-baselines3:

```bash
pip install stable-baselines3[extra] gymnasium
```

### GPU Memory Issues

If you run out of GPU memory:

1. Reduce `--n-envs` (number of parallel environments)
2. Use CPU: `--device-id -1`
3. Reduce batch size in the training script

### Training Not Converging

If the agent doesn't learn:

1. Check reward scale: adjust `reward_scale` in environment
2. Increase training time: use more `--total-timesteps`
3. Try different algorithms: SAC or TD3 instead of PPO
4. Check observation normalization: ensure temperatures are in expected range

## Next Steps

After training:

1. Evaluate the trained model on test simulations
2. Compare performance with PID controller
3. Fine-tune hyperparameters if needed
4. Deploy the trained model in production simulations

For questions or issues, please refer to the main README or open an issue on GitHub.

