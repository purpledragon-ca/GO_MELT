# Training Stability Fixes

## Problem
The training was experiencing exploding loss values:
- `ep_rew_mean`: -4.15e+09 (extremely negative)
- `loss`: 2.36e+14 (extremely high)
- `value_loss`: 4.01e+14 (extremely high)

## Root Causes

1. **Reward Scale Too Large**: The reward was computed as `-scale * (error²)` where error could be 300K, leading to rewards of -90,000 per step. Over 4000 steps, this accumulates to billions.

2. **Unstable Temperature Dynamics**: The temperature model could produce extreme values without proper clamping.

3. **Large Action Adjustments**: Actions could cause power to swing by ±50% of the power range, leading to unstable control.

4. **No Gradient Clipping**: Large gradients could cause the policy to diverge.

## Fixes Applied

### 1. Normalized Reward Function
```python
# Before: reward = -scale * (error²)
# After: reward = -scale * (normalized_error²)
normalized_error = error / target_temperature
reward = -scale * (normalized_error²)
reward = clip(reward, -100.0, 0.0)  # Prevent extreme values
```

This keeps rewards in a reasonable range regardless of temperature scale.

### 2. Improved Temperature Dynamics
- Added temperature clamping: `[500K, 5000K]`
- Reduced time constant: `0.05` (from 0.1) for more stable response
- Reduced noise: `5K` (from 10K)
- Better power-to-temperature relationship using power law

### 3. Reduced Action Scale
- Changed from ±50% to ±20% of power range
- Prevents large power swings that destabilize training

### 4. Training Hyperparameters
- Reduced learning rate: `1e-4` (from 3e-4)
- Added gradient clipping: `max_grad_norm=0.5`
- Added value function coefficient: `vf_coef=0.5`

### 5. Default Reward Scale
- Changed default from `1.0` to `0.1` for more conservative rewards

## Expected Results

After these fixes, you should see:
- Rewards in range: `[-100, 0]` per step
- Episode rewards: `[-40000, 0]` for 4000-step episodes
- Stable loss values: `[0.1, 100]` range
- Positive learning: `explained_variance > 0`

## If Training Still Fails

1. **Reduce reward scale further**: Use `--reward-scale 0.01` in environment
2. **Shorter episodes**: Use `--max-steps 1000` to test faster
3. **Lower learning rate**: Modify in `train_rl.py` to `5e-5`
4. **Check observation range**: Ensure observations are properly normalized [0, 1]

## Testing

To verify the fixes work:

```bash
# Test with shorter episodes first
python train_rl.py examples/rl.json \
    --output-dir ./rl_training_test \
    --total-timesteps 10000 \
    --max-steps 500

# Monitor with TensorBoard
tensorboard --logdir ./rl_training_test/tensorboard
```

Look for:
- Rewards stabilizing (not exploding)
- Loss decreasing over time
- Explained variance increasing
- Episode length increasing (agent learning to stay alive)

