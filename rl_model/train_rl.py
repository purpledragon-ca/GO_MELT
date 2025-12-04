"""
RL Training Script for GO-MELT Power Control

This script trains a PPO (Proximal Policy Optimization) agent to control laser power
in GO-MELT simulations to maintain target temperature using the GoMeltEnv environment.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")

from rl_model.go_melt_env import GoMeltEnv
from rl_model.observation_utils import print_observation_breakdown
import gymnasium as gym


class ObservationPrintingWrapper(gym.Wrapper):
    """
    Wrapper that prints observations at each step during training.
    Only prints when debug mode is enabled.
    """
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.step_num = 0
        self.debug = debug
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_num += 1
        
        if not self.debug:
            return obs, reward, terminated, truncated, info
        
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
        
        if not self.debug:
            return obs, info
        
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
    episode_length: int = 5000,
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
    total_timesteps: int = 100000,
    n_envs: int = 1,
    device_id: int = 0,
    debug: bool = False,
    **env_kwargs
):
    """
    Train a PPO RL agent for power control using GoMeltEnv.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration JSON file
    output_dir : str
        Directory to save training results and model
    total_timesteps : int
        Total number of training timesteps
    n_envs : int
        Number of parallel environments
    device_id : int
        GPU device ID
    debug : bool
        Enable detailed debug printing
    **env_kwargs
        Additional arguments for environment
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training. Install with: pip install stable-baselines3[extra]")
    
    print("Stage: Initializing training...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load PPO configuration from config file
    ppo_cfg = config.get("ppo_cfg", {})
    
    # Default PPO parameters (used if not specified in config)
    default_ppo_cfg = {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.95,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5
    }
    
    # Merge config with defaults (config takes precedence)
    ppo_params = {**default_ppo_cfg, **ppo_cfg}
    
    # Load training configuration from config file
    training_cfg = config.get("training_cfg", {})
    
    # Default training parameters (used if not specified in config)
    default_training_cfg = {
        "checkpoint_freq": 10000,
        "eval_freq": 5000,
        "n_eval_episodes": 1
    }
    
    # Merge training config with defaults (config takes precedence)
    training_params = {**default_training_cfg, **training_cfg}
    
    # Create training config
    train_config = create_training_config(config_file, str(output_path / "sim_results"))
    
    # Save training config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(train_config, f, indent=2)
    
    if debug:
        print(f"Training PPO agent for GO-MELT power control")
        print(f"Config file: {config_file}")
        print(f"Output directory: {output_dir}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Number of environments: {n_envs}")
        print(f"PPO parameters from config:")
        for key, value in ppo_params.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Algorithm: PPO, Timesteps: {total_timesteps:,}, Envs: {n_envs}")
    
    # Create environment
    def make_env():
        env = GoMeltEnv(
            config_file=config_file,
            device_id=device_id,
            verbose=debug,  # Only print step details in debug mode
            **env_kwargs
        )
        env = Monitor(env, str(output_path / "monitor"))
        env = ObservationPrintingWrapper(env, debug=debug)
        return env
    
    # Validate environment before training
    print("Stage: Validating environment...")
    try:
        test_env = make_env()
        # Get unwrapped environment (in case it's wrapped by Monitor)
        unwrapped_env = test_env.unwrapped if hasattr(test_env, 'unwrapped') else test_env
        test_obs, test_info = test_env.reset()
        
        if debug:
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
        else:
            print(f"  ✓ Environment validated (obs shape: {test_obs.shape})")
        test_env.close()
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise
    
    print("Stage: Creating environments...")
    if n_envs > 1:
        env = make_vec_env(make_env, n_envs=n_envs)
        if debug:
            print(f"  Created {n_envs} parallel environments")
    else:
        env = make_env()
        if debug:
            print(f"  Created single environment")
    
    # Create evaluation environment
    eval_env = make_env()
    
    # For MLP policies, CPU is typically faster than GPU
    # User can override with device_id, but we'll use CPU by default for MLP
    use_device = "cpu" if device_id < 0 else (f"cuda:{device_id}" if device_id >= 0 else "cpu")
    
    print("Stage: Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1 if debug else 0,
        tensorboard_log=str(output_path / "tensorboard"),
        device=use_device,
        learning_rate=ppo_params["learning_rate"],
        n_steps=ppo_params["n_steps"],
        batch_size=ppo_params["batch_size"],
        n_epochs=ppo_params["n_epochs"],
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params["gae_lambda"],
        clip_range=ppo_params["clip_range"],
        ent_coef=ppo_params["ent_coef"],
        max_grad_norm=ppo_params["max_grad_norm"],
        vf_coef=ppo_params["vf_coef"],
    )
    
    if debug:
        print(f"  Model initialized on {use_device}")
    
    # Setup callbacks
    print("Stage: Setting up callbacks...")
    # Create checkpoints directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=training_params["checkpoint_freq"],
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
        eval_freq=training_params["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=training_params["n_eval_episodes"],
        warn=False  # Suppress warnings
    )
    
    if debug:
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  Evaluations: {eval_dir}")
    
    # Train the model
    if debug:
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        print(f"Algorithm: PPO")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Device: {use_device}")
        print(f"Number of environments: {n_envs}")
        print("="*60 + "\n")
    else:
        print("Stage: Starting training...")
        print(f"  Training PPO model for {total_timesteps:,} timesteps on {use_device}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=not debug  # Show progress bar only in non-debug mode
        )
    except KeyboardInterrupt:
        print("\nStage: Training interrupted by user. Saving model...")
        # Save interrupted model
        interrupted_model_path = output_path / "interrupted_model"
        model.save(str(interrupted_model_path))
        print(f"  Model saved to: {interrupted_model_path}")
        raise
    except Exception as e:
        print(f"\nStage: Training error occurred. Saving model...")
        if debug:
            print(f"  Error: {e}")
        try:
            error_model_path = output_path / "error_model"
            model.save(str(error_model_path))
            print(f"  Model saved to: {error_model_path}")
        except:
            print("  Could not save model.")
        raise
    
    training_time = time.time() - start_time
    print("Stage: Training completed")
    if debug:
        print(f"  Training time: {training_time:.2f} seconds")
    else:
        print(f"  Completed in {training_time:.2f} seconds")
    
    # Save final model
    print("Stage: Saving final model...")
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    if debug:
        print(f"  Final model saved to: {final_model_path}")
    else:
        print(f"  Saved to: {final_model_path}")
    
    # Save training info
    training_info = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "config_file": config_file,
        "env_kwargs": env_kwargs
    }
    
    with open(output_path / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("Stage: Training complete!")
    if debug:
        print(f"  Model saved to: {output_dir}")
        print(f"  To use the trained model, update rl_controller.py to load: {final_model_path}.zip")
    else:
        print(f"  Model: {final_model_path}.zip")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for GO-MELT power control")
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/rl.json",
        help="Path to GO-MELT configuration JSON file (default: config/rl.json)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug printing (default: False)"
    )
    
    args = parser.parse_args()
    
    # Use config_file from args
    config_file = args.config_file
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Load and validate config file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        if args.debug:
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
    
    # Load training configuration from config file
    training_cfg = config.get("training_cfg", {})
    # Fallback to power_controller if not in training_cfg (for backward compatibility)
    power_controller = config.get("power_controller", {})
    controller_params = power_controller.get("controller_params", {})
    
    # Extract all training parameters from config
    output_dir = training_cfg.get("output_dir", "./rl_training")
    total_timesteps = training_cfg.get("total_timesteps", 100000)
    n_envs = training_cfg.get("n_envs", 1)
    device_id = training_cfg.get("device_id", 0)
    target_temperature = training_cfg.get("target_temperature", power_controller.get("target_temperature", 3000.0))
    power_min = training_cfg.get("power_min", controller_params.get("power_min", 0.0))
    power_max = training_cfg.get("power_max", controller_params.get("power_max", 500.0))
    base_power = training_cfg.get("base_power", 285.0)
    update_interval = training_cfg.get("update_interval", power_controller.get("update_interval", 10))
    max_steps = training_cfg.get("max_steps", None)
    reward_scale = training_cfg.get("reward_scale", 0.1)
    
    # Set GPU device
    if device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if args.debug:
            print(f"Using GPU device: {device_id}")
    else:
        if args.debug:
            print("Using CPU")
    
    # Train the agent
    try:
        model = train_rl_agent(
            config_file=config_file,
            output_dir=output_dir,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            device_id=device_id,
            debug=args.debug,
            target_temperature=target_temperature,
            power_min=power_min,
            power_max=power_max,
            base_power=base_power,
            update_interval=update_interval,
            max_steps=max_steps,
            reward_scale=reward_scale
        )
        if args.debug:
            print("\n" + "="*60)
            print("✓ Training completed successfully!")
            print("="*60)
        else:
            print("✓ Training completed successfully!")
    except KeyboardInterrupt:
        if args.debug:
            print("\n\nTraining interrupted by user.")
        else:
            print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

