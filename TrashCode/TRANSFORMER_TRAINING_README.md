# Physics-Informed Transformer for Inverse Dynamics

This project implements a **physics-informed neural network** that learns to predict ground reaction forces (GRF) and center of pressure (COP) from kinematic data, with gradients flowing through a differentiable Jacobian-based inverse dynamics calculation.

## 🎯 Overview

The training pipeline:
1. **Input**: Position, velocity, acceleration (`qpos`, `qvel`, `qacc`)
2. **Model**: Transformer predicts GRF and COP for both feet
3. **Physics Layer**: Jacobian multiplication converts cartesian forces → joint torques
4. **Loss**: Comparison with ground truth torques (physics-informed!)
5. **Backpropagation**: Gradients flow through Jacobian and transformer

```
Kinematics → Transformer → GRF/COP → Jacobian (J^T·F) → Torques → Loss
    ↑                                                                  ↓
    └──────────────────── Gradients flow back ──────────────────────┘
```

## 📂 Files

### Core Implementation
- **`physics_informed_transformer.py`**: Main transformer architecture and training functions
  - `GRFCOPTransformer`: Transformer model with positional encoding
  - `jacobian_to_joint_torques()`: Differentiable J^T·F operation
  - `physics_informed_loss()`: Torque-space loss computation
  - `train_step()`: JIT-compiled training step with gradients
  - `eval_step()`: Evaluation without gradients

### Training & Validation
- **`train_physics_transformer.py`**: Complete training script
  - Loads data from `Results/` directory
  - Creates sequences with overlap
  - Trains with physics-informed loss
  - Saves checkpoints and metrics

- **`validate_jax_inverse_dynamics.py`**: Validation script
  - Tests JAX inverse dynamics implementation
  - Compares with reference `tau.csv`
  - Plots comparisons and calculates RMSE

### Supporting Files
- **`inverse_dynamics_jax_compatible.py`**: JAX-compatible inverse dynamics
- **`JAX_COMPATIBILITY_GUIDE.md`**: Migration guide for JAX

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install jax[cuda] flax optax mujoco numpy pandas matplotlib
```

### 2. Validate Inverse Dynamics (Optional but Recommended)

First, ensure your JAX inverse dynamics is accurate:

```bash
python validate_jax_inverse_dynamics.py
```

Check the generated plots in `Results/jax_validation_comparison_*.png`. RMSE should be < 1.0 N·m.

### 3. Train the Transformer

```bash
python train_physics_transformer.py \
    --epochs 100 \
    --batch_size 16 \
    --seq_len 200 \
    --lr 1e-4 \
    --lambda_torque 1.0 \
    --lambda_grf 0.1
```

**Key arguments**:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--seq_len`: Sequence length for transformer
- `--overlap`: Overlap between sequences (default: 100)
- `--lr`: Learning rate
- `--lambda_torque`: Weight for torque loss (physics constraint)
- `--lambda_grf`: Weight for GRF regularization

### 4. Monitor Training

The script outputs:
```
Epoch 1/100 (12.3s)
  Train - Loss: 0.8542, Torque RMSE: 0.3214 N·m, Grad Norm: 12.45
  Val   - Loss: 0.7891, Torque RMSE: 0.2987 N·m
  ✓ Saved best model (val_loss: 0.7891)
```

Checkpoints are saved to `checkpoints/`:
- `best_model.npy`: Best model based on validation loss
- `checkpoint_epoch_N.npy`: Periodic checkpoints
- `final_model.npy`: Final model after all epochs
- `training_history.json`: Loss curves and metrics

## 🧠 Model Architecture

### Transformer Details

```python
GRFCOPTransformer(
    d_model=256,        # Model dimension
    num_heads=8,        # Multi-head attention heads
    num_layers=6,       # Transformer encoder layers
    d_ff=1024,          # Feed-forward dimension
    output_dim=12,      # GRF (6) + COP (6)
    dropout_rate=0.1    # Dropout for regularization
)
```

**Input shape**: `[batch, seq_len, 3*nv]` where `nv` = number of DOFs
- Concatenated: `[qpos, qvel, qacc]`

**Output shape**: `[batch, seq_len, 12]`
- `[:, :, 0:3]`: GRF left foot (Fx, Fy, Fz)
- `[:, :, 3:6]`: GRF right foot (Fx, Fy, Fz)
- `[:, :, 6:9]`: COP left foot (x, y, z)
- `[:, :, 9:12]`: COP right foot (x, y, z)

### Physics-Informed Loss

```python
tau_predicted = J^T @ [GRF_left, GRF_right, COP_left, COP_right]
loss = lambda_torque * MSE(tau_predicted, tau_target) + lambda_grf * regularization
```

**Why this works**:
- Gradients flow through Jacobian multiplication: `∂loss/∂GRF = ∂loss/∂tau · J^T`
- Model learns to predict forces that produce correct torques
- Physics constraint is enforced through differentiable simulation

## 📊 Data Format

### Required Data Files (in `Results/`)

1. **Kinematics**:
   - `qpos_matrix.npy`: `[num_timesteps, nv]` - Joint positions
   - `qvel_matrix.npy`: `[num_timesteps, nv]` - Joint velocities
   - `qacc_matrix.npy`: `[num_timesteps, nv]` - Joint accelerations

2. **Jacobian**:
   - `jacobian_data.npz`: Contains Jacobian matrices
     - `J_calcn_l`: `[num_timesteps, nv, 6]` - Left foot Jacobian
     - `J_calcn_r`: `[num_timesteps, nv, 6]` - Right foot Jacobian
   - Combined to `[num_timesteps, nv, 12]` for full mapping

3. **Ground Truth**:
   - `grf_matrix.npy`: `[num_timesteps, 6]` - GRF for both feet
   - `cop_matrix.npy`: `[num_timesteps, 6]` - COP for both feet
   - `jax_joint_forces.npy` or `tau.csv`: `[num_timesteps, nv]` - Target torques

### Data Preparation

The training script automatically:
1. Loads all data files
2. Creates overlapping sequences (e.g., 200 timesteps with 100 overlap)
3. Splits into train/validation sets (default 80/20)
4. Creates batches for training

## 🔧 Advanced Usage

### Custom Jacobian Format

If your Jacobian is stored differently, modify `load_training_data()` in `train_physics_transformer.py`:

```python
# Example: Single Jacobian for both feet
if 'jacobian_full' in jacobian_data:
    jacobian_matrix = jacobian_data['jacobian_full']  # [T, nv, 12]

# Example: Separate processing
J_left = process_jacobian(jacobian_data['J_left'])
J_right = process_jacobian(jacobian_data['J_right'])
jacobian_matrix = np.concatenate([J_left, J_right], axis=-1)
```

### Custom Body IDs

Update body IDs in `train_physics_transformer.py`:

```python
# Find body IDs from your model
import mujoco
mj_model = mujoco.MjModel.from_xml_path("path/to/model.xml")

body_ids = {
    'calcn_l': mj_model.body('calcn_l').id,
    'calcn_r': mj_model.body('calcn_r').id,
    'pelvis': mj_model.body('pelvis').id
}
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Model size**: Larger model = more capacity but slower
   ```bash
   --d_model 512 --num_heads 16 --num_layers 8 --d_ff 2048
   ```

2. **Loss weights**: Balance physics accuracy vs GRF realism
   ```bash
   --lambda_torque 1.0  # Higher = prioritize torque accuracy
   --lambda_grf 0.1     # Higher = penalize unrealistic forces
   ```

3. **Sequence length**: Longer = more context but more memory
   ```bash
   --seq_len 300 --overlap 150  # Longer sequences
   ```

4. **Learning rate schedule**: Already uses warmup + cosine decay
   - Modify in `create_train_state()` if needed

## 🎓 Understanding the Gradient Flow

### Forward Pass
```python
# 1. Transformer predicts GRF/COP
grf_cop = transformer(kinematics)  # [batch, seq, 12]

# 2. Convert to joint torques via Jacobian
tau = J^T @ grf_cop  # [batch, seq, nv]

# 3. Compute loss in torque space
loss = MSE(tau, tau_target)
```

### Backward Pass (Automatic!)
```python
# JAX automatically computes:
∂loss/∂tau = 2(tau - tau_target)
∂tau/∂grf_cop = J^T
∂loss/∂grf_cop = ∂loss/∂tau · J^T  # Chain rule

# Then gradients flow to transformer parameters:
∂loss/∂θ = ∂loss/∂grf_cop · ∂grf_cop/∂θ
```

This is handled automatically by:
```python
grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
(loss, metrics), grads = grad_fn(state.params)
```

## 📈 Monitoring & Debugging

### Key Metrics

1. **Torque RMSE**: Physics accuracy
   - Target: < 1.0 N·m for good model
   - If high: Increase `--lambda_torque` or check Jacobian

2. **GRF Regularization**: Force realism
   - Penalizes negative vertical forces
   - If high: Forces are unrealistic, increase `--lambda_grf`

3. **Gradient Norm**: Training stability
   - Typical: 1.0 - 100.0
   - If too high (>1000): Reduce learning rate
   - If too low (<0.01): Learning might be stuck

### Common Issues

**Issue**: Loss not decreasing
- Check data normalization (consider standardizing inputs)
- Reduce learning rate: `--lr 5e-5`
- Check Jacobian shape matches expected format

**Issue**: Gradient explosion
- Reduce learning rate
- Add gradient clipping in optimizer (modify `create_train_state()`)
- Check for NaN values in data

**Issue**: Torque RMSE high
- Validate inverse dynamics first with `validate_jax_inverse_dynamics.py`
- Check Jacobian is correct
- Increase model capacity: `--d_model 512 --num_layers 8`

## 🔬 Physics Validation

Before training, validate your physics:

```bash
# 1. Run validation
python validate_jax_inverse_dynamics.py

# 2. Check plots
ls Results/jax_validation_comparison_*.png

# 3. Look for low RMSE (<1.0 N·m)
# Mean RMSE: 0.234 N·m  ← Good!
# Mean RMSE: 5.678 N·m  ← Problem with physics
```

## 🚀 Production Deployment

### Save Trained Model

```python
import numpy as np

# Load trained parameters
params = np.load('checkpoints/best_model.npy', allow_pickle=True).item()

# Use for inference
predictions = model.apply({'params': params}, kinematics, train=False)
```

### Batch Inference

```python
from physics_informed_transformer import GRFCOPTransformer
import jax.numpy as jnp

# Load model
model = GRFCOPTransformer()
params = load_params('checkpoints/best_model.npy')

# Batch inference (JIT-compiled)
@jax.jit
def predict_batch(kinematics):
    return model.apply({'params': params}, kinematics, train=False)

# Use
kinematics = jnp.array(...)  # [batch, seq_len, 3*nv]
grf_cop_predictions = predict_batch(kinematics)
```

## 📚 References

- **JAX**: https://github.com/google/jax
- **Flax**: https://github.com/google/flax
- **MuJoCo**: https://github.com/google-deepmind/mujoco
- **Physics-Informed Neural Networks**: Raissi et al. (2019)

## 🤝 Contributing

Key areas for improvement:
1. **Contact detection**: Currently assumes both feet always in contact
2. **Multi-contact**: Extend to hands, other body parts
3. **Adaptive Jacobian**: Learn to predict Jacobian changes
4. **Uncertainty**: Add probabilistic predictions (dropout, ensembles)

## 📝 Citation

If you use this code, please cite:
```
[Your paper/project]
Uses differentiable inverse dynamics for physics-informed learning
```

## ⚡ Performance Tips

1. **Use GPU**: Training is 10-100x faster on GPU
   ```bash
   # Check GPU is detected
   python -c "import jax; print(jax.devices())"
   ```

2. **Increase batch size**: Larger batches = better GPU utilization
   ```bash
   --batch_size 32  # If you have enough memory
   ```

3. **JIT compilation**: First epoch is slow (compilation), then fast

4. **Mixed precision**: For even faster training (advanced)
   ```python
   # In create_train_state, add:
   optimizer = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.scale_by_adam(),
       optax.scale(-learning_rate)
   )
   ```

## 🎯 Next Steps

After successful training:
1. **Analyze predictions**: Plot predicted vs actual GRF/COP
2. **Test generalization**: Try on new subjects/movements
3. **Integrate with control**: Use predictions for robot control
4. **Real-time deployment**: Optimize inference for real-time use

---

**Questions?** Check `JAX_COMPATIBILITY_GUIDE.md` for JAX-specific details or open an issue!
