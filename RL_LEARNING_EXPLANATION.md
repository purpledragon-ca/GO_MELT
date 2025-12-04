# How RL Training Works: Error Calculation and Learning Process

## 1. Error Calculation (Reward Function)

The error is calculated in `go_melt_env.py` in the `_compute_reward()` method:

```python
def _compute_reward(self, temperature: float) -> float:
    # Step 1: Calculate raw error
    error = self.target_temperature - temperature
    # Example: error = 3000.0 - 299.0 = 2701.0 K
    
    # Step 2: Normalize the error (divide by target to get relative error)
    normalized_error = error / self.target_temperature
    # Example: normalized_error = 2701.0 / 3000.0 = 0.9003 (90% error)
    
    # Step 3: Square the normalized error and make it negative
    # Squaring penalizes large errors more (quadratic penalty)
    # Negative because we want to maximize reward (closer to target = less negative)
    reward = -self.reward_scale * (normalized_error ** 2)
    # Example: reward = -0.1 * (0.9003^2) = -0.1 * 0.8105 = -0.081
    
    # Step 4: Clip to prevent extreme values
    reward = np.clip(reward, -100.0, 0.0)
    return reward
```

### Reward Characteristics:
- **Best case** (temp = target): reward = 0.0
- **Worst case** (temp far from target): reward approaches -100.0
- **Shape**: Quadratic penalty - small errors are penalized lightly, large errors heavily

### Example Calculations:
| Temperature | Error | Normalized Error | Reward |
|------------|-------|------------------|--------|
| 3000.0 K (target) | 0.0 | 0.0 | 0.0 |
| 2700.0 K | 300.0 | 0.1 | -0.001 |
| 1500.0 K | 1500.0 | 0.5 | -0.025 |
| 299.0 K | 2701.0 | 0.9003 | -0.081 |

## 2. How the Model Learns (PPO Algorithm Flow)

### Training Loop Overview:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Environment Step (go_melt_env.py)                        │
│    - Agent outputs action (power adjustment)                 │
│    - Simulator runs with that power                          │
│    - Environment returns: (observation, reward, done, info) │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Collect Experience (stable-baselines3)                    │
│    - Store: (state, action, reward, next_state)              │
│    - Collect n_steps (default: 2048) before learning         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Compute Returns (GAE - Generalized Advantage Estimation) │
│    - Calculate discounted future rewards                    │
│    - Estimate advantage: A = Q(s,a) - V(s)                   │
│    - GAE combines multiple n-step returns                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Update Policy (PPO Loss)                                  │
│    - Policy Loss: Maximize probability of good actions      │
│    - Value Loss: Predict expected returns accurately        │
│    - Entropy Loss: Encourage exploration                    │
│    - Total Loss = policy_loss + vf_coef*value_loss -         │
│                   ent_coef*entropy_loss                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Backpropagation                                           │
│    - Compute gradients of loss w.r.t. network parameters     │
│    - Update network weights using optimizer (Adam)           │
│    - Clip gradients (max_grad_norm = 0.5)                   │
└─────────────────────────────────────────────────────────────┘
                        ↓
                    Repeat
```

### Detailed Learning Process:

#### Step 1: Action Selection
```python
# In train_rl.py, model.learn() calls:
action = model.predict(observation)  # Neural network outputs action
# Action is in range [-1, 1], converted to power adjustment
```

#### Step 2: Environment Interaction
```python
# In go_melt_env.py step():
observation, reward, terminated, truncated, info = env.step(action)
# Reward is computed using _compute_reward() based on temperature error
```

#### Step 3: Experience Collection
- PPO collects `n_steps=2048` steps before updating
- Stores: states, actions, rewards, values, log_probs
- Computes advantages using GAE (Generalized Advantage Estimation)

#### Step 4: Policy Update (PPO Loss Components)

**a) Policy Loss (Clipped Surrogate Objective):**
```python
# PPO tries to increase probability of actions that led to good rewards
# But limits how much the policy can change (clipping)
ratio = new_prob(action) / old_prob(action)
clipped_ratio = clip(ratio, 1-clip_range, 1+clip_range)
policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**b) Value Loss:**
```python
# Value function predicts expected future rewards
# Loss = MSE between predicted value and actual returns
value_loss = (predicted_value - actual_return)^2
```

**c) Entropy Loss:**
```python
# Encourages exploration (prevent policy from becoming too deterministic)
entropy_loss = -sum(p * log(p))  # Higher entropy = more exploration
```

**Total Loss:**
```python
total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
```

#### Step 5: Gradient Update
- Backpropagate through neural network
- Update weights using Adam optimizer with learning_rate
- Repeat for `n_epochs=10` times on the same batch

### Key PPO Parameters (from config/rl.json):

- **learning_rate (1e-4)**: How fast the network learns
- **n_steps (2048)**: Steps collected before updating
- **batch_size (64)**: Samples per gradient update
- **n_epochs (10)**: How many times to update on same data
- **gamma (0.99)**: Discount factor for future rewards
- **gae_lambda (0.95)**: GAE parameter for advantage estimation
- **clip_range (0.95)**: Limits policy update size
- **vf_coef (0.5)**: Weight of value loss
- **ent_coef (0.01)**: Weight of entropy (exploration)

## 3. Why Temperature is Low (299K vs 3000K target)

The agent is getting very negative rewards (around -0.081), which means:
1. The policy is learning that current actions lead to bad outcomes
2. But it may not be learning the right actions to increase temperature
3. Possible issues:
   - Reward scale too small (0.1) - rewards are very negative
   - Learning rate might be too low
   - Not enough exploration
   - Simulator might not be responding correctly to power changes

## 4. How to Monitor Learning

Check TensorBoard logs:
```bash
tensorboard --logdir=./rl_training/tensorboard
```

Key metrics:
- `train/policy_loss`: Should decrease (policy improving)
- `train/value_loss`: Should decrease (better value predictions)
- `rollout/ep_rew_mean`: Should increase (better rewards)
- `train/approx_kl`: Should be small (policy not changing too much)

