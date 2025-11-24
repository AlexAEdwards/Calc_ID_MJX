# Minimal Example: What You Need to Change

## TL;DR - Two Simple Steps

### Step 1: Replace the Model (in `physics_informed_transformer.py`)

**Find this:**
```python
class YourModelPlaceholder(nn.Module):
    input_size: int = 100
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x
```

**Replace with your transformer:**
```python
class YourTransformer(nn.Module):
    input_size: int
    # Add your hyperparameters here
    d_model: int = 256
    num_heads: int = 8
    # etc.
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Your transformer code here
        # Must return shape: [batch, 12]
        return predictions
```

### Step 2: Use Your Model (in `train_physics_transformer.py`)

**Find this (around line 229):**
```python
model = YourModelPlaceholder(
    input_size=input_size,
    output_size=12
)
```

**Change to:**
```python
model = YourTransformer(
    input_size=input_size,
    d_model=256,
    num_heads=8,
    output_size=12
)
```

**That's it!**

---

## What Gets Handled Automatically

You don't need to touch any of this:

- ✓ Data loading from `.npy` files
- ✓ Batch creation
- ✓ Physics-informed loss (tau = J^T @ F)
- ✓ Gradient computation
- ✓ Optimizer updates
- ✓ Training loop
- ✓ Checkpointing
- ✓ Evaluation

## Input/Output Contract

**Your model receives:**
- `x`: JAX array of shape `[batch_size, input_size]`
  - Default: First 100 features from kinematics (qpos, qvel, qacc)
  - You can customize this in `prepare_data()`

**Your model must return:**
- Predictions: JAX array of shape `[batch_size, 12]`
  - Indices 0-5: Left foot (GRF_x, GRF_y, GRF_z, COP_x, COP_y, COP_z)
  - Indices 6-11: Right foot (GRF_x, GRF_y, GRF_z, COP_x, COP_y, COP_z)

## The Physics Magic

Your predictions get transformed into joint torques:

```
F_predicted = your_model(kinematics)  # Shape: [batch, 12]
tau_predicted = J^T @ F_predicted      # Shape: [batch, nv]
loss = MSE(tau_predicted, tau_true)
```

Gradients automatically flow back through:
1. The Jacobian multiplication (J^T @ F)
2. Your model's predictions (F)
3. All your transformer layers
4. To update your model parameters

This ensures your predictions are **physically plausible** - they must produce torques that match the actual motion!

## Optional Customizations

### Change Input Preparation
In `prepare_data()` (line ~109 of `train_physics_transformer.py`):

```python
def prepare_data(data: Dict[str, np.ndarray]) -> Tuple:
    """CUSTOMIZE THIS based on your needs."""
    
    # Example: Use different timestep window
    start_idx = 50
    end_idx = 150
    
    # Example: Different feature selection
    qpos_subset = data['qpos'][start_idx:end_idx, :10]  # Only first 10 joints
    
    # Example: Normalization
    features = (features - features.mean()) / features.std()
    
    # Must return: (inputs, jacobians, grf_cop, torques)
    return inputs, jacobians, grf_cop, torques
```

### Add More Loss Terms
In `physics_informed_loss()` (line ~63 of `physics_informed_transformer.py`):

```python
# Add your own losses
contact_loss = jnp.mean((predictions[:, 2] - 0)**2)  # Force vertical GRF
symmetry_loss = jnp.mean((predictions[:, :6] - predictions[:, 6:])**2)

total_loss = torque_loss + 0.1 * contact_loss + 0.05 * symmetry_loss
```

### Change Hyperparameters
Run with different arguments:

```bash
python train_physics_transformer.py \
    --epochs 200 \
    --batch_size 32 \
    --lr 5e-5 \
    --lambda_torque 2.0 \
    --lambda_grf 0.05
```

---

## Complete Minimal Example

Here's a complete transformer you could drop in:

```python
class SimpleTransformer(nn.Module):
    """Minimal transformer for GRF/COP prediction."""
    input_size: int
    d_model: int = 128
    num_heads: int = 4
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        
        # Project to d_model
        x = nn.Dense(self.d_model)(x)
        x = x.reshape(batch_size, -1, self.d_model)  # [batch, seq, d_model]
        
        # Single self-attention layer
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads
        )(x, x)
        x = x + attn_out  # Residual
        
        # Feed-forward
        ff_out = nn.Dense(self.d_model * 4)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        x = x + ff_out  # Residual
        
        # Output projection
        x = x.mean(axis=1)  # Pool over sequence
        x = nn.Dense(self.output_size)(x)
        
        return x
```

Then use it:
```python
model = SimpleTransformer(
    input_size=input_size,
    d_model=128,
    num_heads=4,
    output_size=12
)
```

**Done!** The rest is handled automatically.
