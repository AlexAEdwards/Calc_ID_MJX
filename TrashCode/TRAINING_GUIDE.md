# Training Guide: Physics-Informed Model

This guide explains how to replace the placeholder model with your own transformer and train it with physics-informed loss.

## Overview

The training framework consists of two main files:

1. **`physics_informed_transformer.py`** - Contains:
   - `YourModelPlaceholder` - Simple 2-layer MLP you'll replace
   - Physics-informed loss functions
   - Training step functions

2. **`train_physics_transformer.py`** - Contains:
   - Data loading from pre-calculated matrices
   - Batch creation
   - Training loop

## Quick Start

### 1. Replace the Placeholder Model

In `physics_informed_transformer.py`, find `YourModelPlaceholder` (around line 20):

```python
class YourModelPlaceholder(nn.Module):
    """
    REPLACE THIS with your transformer architecture.
    
    Requirements:
    - Input: [batch_size, input_size] 
    - Output: [batch_size, 12] (6 values per foot: GRF_x, GRF_y, GRF_z, COP_x, COP_y, COP_z)
    """
    input_size: int = 100
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Replace this simple MLP with your transformer
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x
```

**Replace it with your transformer:**

```python
class YourTransformer(nn.Module):
    """Your transformer architecture."""
    input_size: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Your transformer implementation here
        # ...
        return output  # Shape: [batch, 12]
```

### 2. Update the Training Script

In `train_physics_transformer.py`, find the model initialization (around line 229):

```python
# Initialize YOUR MODEL HERE
model = YourModelPlaceholder(
    input_size=input_size,
    output_size=12
)
```

Change it to use your model:

```python
# Initialize YOUR MODEL HERE
model = YourTransformer(
    input_size=input_size,
    d_model=256,
    num_heads=8,
    num_layers=6,
    output_size=12
)
```

### 3. Customize Data Preparation (Optional)

In `train_physics_transformer.py`, the `prepare_data()` function (around line 109) prepares your input features:

```python
def prepare_data(data: Dict[str, np.ndarray]) -> Tuple:
    """
    CUSTOMIZE THIS function based on how you want to structure inputs.
    
    Currently: Takes first 100 timesteps of qpos, qvel, qacc and flattens
    You might want: 
    - Different timestep window
    - Different feature selection
    - Normalization
    - Sequence format for transformers
    """
    # ... customize here
```

### 4. Run Training

```bash
python train_physics_transformer.py \
    --data_dir Results \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --lambda_torque 1.0 \
    --lambda_grf 0.1
```

## Physics-Informed Loss

The key innovation is the physics-informed loss that ensures predictions are physically plausible:

```
tau_predicted = J^T @ F_predicted
loss = ||tau_predicted - tau_ground_truth||^2
```

Where:
- `J` is the contact Jacobian (pre-calculated)
- `F_predicted` comes from your model's GRF/COP predictions
- `tau_ground_truth` is from inverse dynamics

**This loss is fully differentiable**, so gradients flow through:
1. Your model parameters
2. GRF/COP predictions
3. Jacobian multiplication
4. Torque comparison

## Data Format

Your pre-calculated data includes:

- **`qpos.npy`** - Joint positions [num_timesteps, nq]
- **`qvel.npy`** - Joint velocities [num_timesteps, nv]
- **`qacc.npy`** - Joint accelerations [num_timesteps, nv]
- **`contact_jacobian.npy`** - Contact Jacobians [num_timesteps, nv, 12]
- **`grf_cop.npy`** - Ground truth GRF/COP [num_timesteps, 12]
- **`tau.npy`** - Ground truth torques [num_timesteps, nv]

The framework automatically loads these and creates batches.

## Important Notes

1. **Body IDs**: Update the body IDs in `main()` to match your model:
   ```python
   body_ids = {
       'calcn_l': 10,  # Your left foot body ID
       'calcn_r': 15,  # Your right foot body ID
       'pelvis': 1     # Your pelvis body ID
   }
   ```

2. **Output Format**: Your model MUST output 12 values:
   - Indices 0-5: Left foot (GRF_x, GRF_y, GRF_z, COP_x, COP_y, COP_z)
   - Indices 6-11: Right foot (same format)

3. **Train/Val Split**: Currently uses the same data. For production:
   - Load multiple trials/subjects
   - Split into separate train/validation sets

4. **Jacobian Shape**: The Jacobian has shape [batch, nv, 12], where:
   - `nv` = number of degrees of freedom (varies by model)
   - 12 = 6 coordinates per foot × 2 feet

## Key Functions You Can Ignore

These functions handle the physics - **you don't need to modify them**:

- `jacobian_to_joint_torques()` - Computes tau = J^T @ F
- `physics_informed_loss()` - Compares predicted vs ground truth torques
- `train_step()` - JIT-compiled training step with gradients
- `eval_step()` - Evaluation without gradients

Just focus on replacing `YourModelPlaceholder` with your transformer!

## Example: Attention-Based Transformer

Here's a skeleton for a transformer replacement:

```python
class ContactTransformer(nn.Module):
    input_size: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    output_size: int = 12
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Reshape to sequence if needed
        # x shape: [batch, seq_len, features] or [batch, features]
        
        # Positional encoding
        x = PositionalEncoding(d_model=self.d_model)(x)
        
        # Transformer encoder layers
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, train=train)
        
        # Output projection
        x = x.mean(axis=1) if len(x.shape) == 3 else x  # Pool sequence
        x = nn.Dense(self.output_size)(x)
        
        return x
```

Then implement `PositionalEncoding` and `TransformerEncoderBlock` as needed.

## Questions?

The structure is intentionally minimal to make it easy to insert your own architecture. The only requirements are:

1. Your model inherits from `nn.Module`
2. It has a `__call__` method that takes input and returns 12 outputs
3. You can use `@nn.compact` or the `setup()` method style

Everything else (loss computation, training loop, gradient updates) is handled automatically!
