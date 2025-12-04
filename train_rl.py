"""
RL Training Script for GO-MELT Power Control

This script trains a reinforcement learning agent to control laser power
in GO-MELT simulations to maintain target temperature.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add go_melt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'go_melt'))

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")

from go_melt_env import GoMeltEnv
from observation_utils import print_observation_breakdown
import gymnasium as gym


class ObservationPrintingWrapper(gym.Wrapper):
    """
    Wrapper that prints observations at each step during training.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_num = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_num += 1
        
        # Get unwrapped environment for observation config
        unwrapped = self.unwrapped if hasattr(self, 'unwrapped') else self.env
        obs_config = unwrapped.observation_config if hasattr(unwrapped, 'observation_config') else {}
        
        # Handle action format (could be scalar or array)
        action_str = f"{action[0]:.4f}" if hasattr(action, '__len__') and len(action) > 0 else f"{action:.4f}"
        
        print(f"\n{'='*60}")
        print(f"Step {self.step_num}")
        print(f"{'='*60}")
        print(f"Action: {action_str} (normalized)")
        print(f"Reward: {reward:.4f}")
        
        # Handle info dict (could be dict or array of dicts for vectorized)
        if isinstance(info, dict):
            temp_val = info.get('temperature', 'N/A')
            power_val = info.get('power', 'N/A')
            temp_str = f"{temp_val:.2f}K" if isinstance(temp_val, (int, float)) else str(temp_val)
            power_str = f"{power_val:.2f}W" if isinstance(power_val, (int, float)) else str(power_val)
            print(f"Temperature: {temp_str}")
            print(f"Power: {power_str}")
        else:
            print(f"Temperature: {info}")
            print(f"Power: {info}")
        
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"\nObservation:")
        
        toolpath_range = obs_config.get('toolpath_normalization_range', 10.0) if obs_config else 10.0
        print_observation_breakdown(
            observation=obs,
            observation_config=obs_config,
            history_length=unwrapped.observation_history_length if hasattr(unwrapped, 'observation_history_length') else 10,
            power_min=unwrapped.power_min if hasattr(unwrapped, 'power_min') else 0.0,
            power_max=unwrapped.power_max if hasattr(unwrapped, 'power_max') else 500.0,
            temp_min=unwrapped.temp_min if hasattr(unwrapped, 'temp_min') else 0.0,
            temp_max=unwrapped.temp_max if hasattr(unwrapped, 'temp_max') else 5000.0,
            toolpath_normalization_range=toolpath_range,
            prefix="  "
        )
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_num = 0
        
        # Get unwrapped environment for observation config
        unwrapped = self.unwrapped if hasattr(self, 'unwrapped') else self.env
        obs_config = unwrapped.observation_config if hasattr(unwrapped, 'observation_config') else {}
        
        print(f"\n{'='*60}")
        print(f"Environment Reset")
        print(f"{'='*60}")
        print(f"\nInitial Observation:")
        
        toolpath_range = obs_config.get('toolpath_normalization_range', 10.0) if obs_config else 10.0
        print_observation_breakdown(
            observation=obs,
            observation_config=obs_config,
            history_length=unwrapped.observation_history_length if hasattr(unwrapped, 'observation_history_length') else 10,
            power_min=unwrapped.power_min if hasattr(unwrapped, 'power_min') else 0.0,
            power_max=unwrapped.power_max if hasattr(unwrapped, 'power_max') else 500.0,
            temp_min=unwrapped.temp_min if hasattr(unwrapped, 'temp_min') else 0.0,
            temp_max=unwrapped.temp_max if hasattr(unwrapped, 'temp_max') else 5000.0,
            toolpath_normalization_range=toolpath_range,
            prefix="  "
        )
        
        return obs, info


def create_training_config(
    base_config_file: str,
    output_dir: str,
    episode_length: int = 1000,
    **kwargs
) -> Dict:
    """
    Create a training configuration from base config.
    
    Parameters:
    -----------
    base_config_file : str
        Path to base configuration file
    output_dir : str
        Output directory for training results
    episode_length : int
        Maximum steps per episode
    **kwargs
        Additional configuration overrides
        
    Returns:
    --------
    dict
        Training configuration dictionary
    """
    with open(base_config_file, 'r') as f:
        config = json.load(f)
    
    # Modify config for training
    # Use shorter simulation for faster training
    if "nonmesh" in config:
        # Reduce simulation length for training
        if "record_step" in config["nonmesh"]:
            config["nonmesh"]["record_step"] = min(config["nonmesh"].get("record_step", 25), 25)
    
    # Set output directory
    if "nonmesh" in config:
        config["nonmesh"]["save_path"] = output_dir
    
    # Override with kwargs
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def train_rl_agent(
    config_file: str,
    output_dir: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    n_envs: int = 1,
    device_id: int = 0,
    **env_kwargs
):
    """
    Train an RL agent for power control.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration JSON file
    output_dir : str
        Directory to save training results and model
    algorithm : str
        RL algorithm to use: "PPO", "SAC", or "TD3"
    total_timesteps : int
        Total number of training timesteps
    n_envs : int
        Number of parallel environments
    device_id : int
        GPU device ID
    **env_kwargs
        Additional arguments for environment
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training. Install with: pip install stable-baselines3[extra]")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create training config
    train_config = create_training_config(config_file, str(output_path / "sim_results"))
    
    # Save training config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(train_config, f, indent=2)
    
    print(f"Training {algorithm} agent for GO-MELT power control")
    print(f"Config file: {config_file}")
    print(f"Output directory: {output_dir}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of environments: {n_envs}")
    
    # Create environment
    def make_env():
        env = GoMeltEnv(
            config_file=config_file,
            device_id=device_id,
            **env_kwargs
        )
        env = Monitor(env, str(output_path / "monitor"))
        env = ObservationPrintingWrapper(env)
        return env
    
    # Validate environment before training
    print("\nValidating environment...")
    try:
        test_env = make_env()
        # Get unwrapped environment (in case it's wrapped by Monitor)
        unwrapped_env = test_env.unwrapped if hasattr(test_env, 'unwrapped') else test_env
        test_obs, test_info = test_env.reset()
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {test_env.observation_space}")
        print(f"  Action space: {test_env.action_space}")
        print(f"  Initial observation shape: {test_obs.shape}")
        
        # Print observation breakdown using shared utility
        obs_config = unwrapped_env.observation_config if hasattr(unwrapped_env, 'observation_config') else {}
        
        toolpath_range = obs_config.get('toolpath_normalization_range', 10.0) if obs_config else 10.0
        print_observation_breakdown(
            observation=test_obs,
            observation_config=obs_config,
            history_length=unwrapped_env.observation_history_length,
            power_min=unwrapped_env.power_min,
            power_max=unwrapped_env.power_max,
            temp_min=unwrapped_env.temp_min,
            temp_max=unwrapped_env.temp_max,
            toolpath_normalization_range=toolpath_range,
            prefix="  "
        )
        
        # Test a step
        test_action = test_env.action_space.sample()
        test_obs, test_reward, test_term, test_trunc, test_info = test_env.step(test_action)
        print(f"\n  Test step successful:")
        print(f"    Action: {test_action[0]:.4f} (normalized)")
        print(f"    Reward: {test_reward:.4f}")
        print(f"    Temperature: {test_info['temperature']:.2f}K")
        print(f"    Power: {test_info['power']:.2f}W")
        print(f"    Terminated: {test_term}, Truncated: {test_trunc}")
        test_env.close()
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if n_envs > 1:
        env = make_vec_env(make_env, n_envs=n_envs)
    else:
        env = make_env()
    
    # Create evaluation environment
    eval_env = make_env()
    
    # Select algorithm
    algorithm = algorithm.upper()
    # For MLP policies, CPU is typically faster than GPU
    # User can override with device_id, but we'll use CPU by default for MLP
    use_device = "cpu" if algorithm == "PPO" else (f"cuda:{device_id}" if device_id >= 0 else "cpu")
    
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(output_path / "tensorboard"),
            device=use_device,
            learning_rate=1e-4,  # Reduced from 3e-4 for stability
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,  # Add gradient clipping to prevent explosion
            vf_coef=0.5,  # Value function coefficient
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(output_path / "tensorboard"),
            device=use_device,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(output_path / "tensorboard"),
            device=use_device,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            policy_delay=2,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: PPO, SAC, TD3")
    
    # Setup callbacks
    # Create checkpoints directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix="rl_model"
    )
    
    # Create evaluations directory
    eval_dir = output_path / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_path / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_dir),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Limit evaluation episodes
        warn=False  # Suppress warnings
    )
    
    # Train the model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {use_device}")
    print(f"Number of environments: {n_envs}")
    print("="*60 + "\n")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")
        # Save interrupted model
        interrupted_model_path = output_path / "interrupted_model"
        model.save(str(interrupted_model_path))
        print(f"Interrupted model saved to: {interrupted_model_path}")
        raise
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        print("Saving current model before exit...")
        try:
            error_model_path = output_path / "error_model"
            model.save(str(error_model_path))
            print(f"Model saved to: {error_model_path}")
        except:
            print("Could not save model.")
        raise
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")
    
    # Save training info
    training_info = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "config_file": config_file,
        "env_kwargs": env_kwargs
    }
    
    with open(output_path / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print(f"To use the trained model, update rl_controller.py to load: {final_model_path}.zip")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for GO-MELT power control")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to GO-MELT configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rl_training",
        help="Output directory for training results (default: ./rl_training)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "TD3"],
        default="PPO",
        help="RL algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=3000.0,
        help="Target temperature in Kelvin (default: 3000.0)"
    )
    parser.add_argument(
        "--power-min",
        type=float,
        default=0.0,
        help="Minimum power in Watts (default: 0.0)"
    )
    parser.add_argument(
        "--power-max",
        type=float,
        default=500.0,
        help="Maximum power in Watts (default: 500.0)"
    )
    parser.add_argument(
        "--base-power",
        type=float,
        default=285.0,
        help="Base power in Watts (default: 285.0)"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10,
        help="Steps between power updates (default: 10)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: None, run full simulation)"
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.1,
        help="Reward scaling factor (default: 0.1, lower = smaller rewards)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file not found: {args.config_file}")
        sys.exit(1)
    
    # Validate config file is valid JSON
    try:
        with open(args.config_file, 'r') as f:
            config_test = json.load(f)
        print(f"✓ Configuration file loaded successfully")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read configuration file: {e}")
        sys.exit(1)
    
    # Check if stable-baselines3 is available
    if not HAS_SB3:
        print("Error: stable-baselines3 is required for training.")
        print("Install with: pip install stable-baselines3[extra]")
        sys.exit(1)
    
    # Set GPU device
    if args.device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        print(f"Using GPU device: {args.device_id}")
    else:
        print("Using CPU")
    
    # Train the agent
    try:
        model = train_rl_agent(
            config_file=args.config_file,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            device_id=args.device_id,
            target_temperature=args.target_temp,
            power_min=args.power_min,
            power_max=args.power_max,
            base_power=args.base_power,
            update_interval=args.update_interval,
            max_steps=args.max_steps,
            reward_scale=args.reward_scale
        )
        print("\n" + "="*60)
        print("✓ Training completed successfully!")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

