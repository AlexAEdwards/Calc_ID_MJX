"""
Physics-Informed Model Training Framework

Simple placeholder for your custom transformer.
Includes:
- Placeholder model (replace with your transformer)
- Differentiable Jacobian multiplication (cartesian → joint space)
- Physics-informed loss: torque space comparison
- Training functions with automatic gradients
"""

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Dict, Any
from functools import partial


class YourModelPlaceholder(nn.Module):
    """
    PLACEHOLDER - Replace this with your own transformer!
    
    Input: [batch, input_size] - Your input format
    Output: [batch, 12] - GRF and COP predictions
        [0:3]   = GRF left foot (Fx, Fy, Fz)
        [3:6]   = GRF right foot (Fx, Fy, Fz)
        [6:9]   = COP left foot (x, y, z)
        [9:12]  = COP right foot (x, y, z)
    """
    input_size: int = 100
    output_size: int = 12
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Replace this entire function with your transformer!
        
        Args:
            x: Input features [batch, input_size]
            train: Training mode flag
        Returns:
            predictions: [batch, 12] GRF/COP values
        """
        # Simple placeholder - replace with your model
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x


def grf_cop_to_external_forces(
    grf_left: jnp.ndarray,
    grf_right: jnp.ndarray,
    cop_left: jnp.ndarray,
    cop_right: jnp.ndarray,
    body_ids: Dict[str, int]
) -> jnp.ndarray:
    """
    Convert GRF and COP predictions to external forces format.
    
    Args:
        grf_left: Ground reaction force left foot [3] (x, y, z)
        grf_right: Ground reaction force right foot [3]
        cop_left: Center of pressure left foot [3]
        cop_right: Center of pressure right foot [3]
        body_ids: Dict with keys 'calcn_l', 'calcn_r', 'pelvis'
    
    Returns:
        external_forces: [nbody, 6] array for MJX
            - forces: [nbody, 3] - force in world frame
            - torques: [nbody, 3] - torque in world frame
    """
    # Assuming you have a fixed number of bodies (e.g., from model)
    # For now, we'll assume nbody is known or passed in
    # In your actual code, this should match mjx_model.nbody
    
    # Create empty external forces array [nbody, 6]
    # You'll need to know nbody - let's assume it's 47 for Falisse model
    nbody = 47  # Adjust based on your model
    external_forces = jnp.zeros((nbody, 6))
    
    # Apply forces at calcn_l (left foot)
    calcn_l_id = body_ids['calcn_l']
    external_forces = external_forces.at[calcn_l_id, :3].set(grf_left)
    
    # Apply forces at calcn_r (right foot)
    calcn_r_id = body_ids['calcn_r']
    external_forces = external_forces.at[calcn_r_id, :3].set(grf_right)
    
    # Calculate torques from COP offset
    # torque = r × F where r is COP offset from body center
    # For simplicity, if COP is already in the right frame, you might skip this
    # or compute: torque_left = jnp.cross(cop_left, grf_left)
    # torque_right = jnp.cross(cop_right, grf_right)
    
    return external_forces


def jacobian_to_joint_torques(
    grf_cop_predictions: jnp.ndarray,
    jacobian: jnp.ndarray,
    body_ids: Dict[str, int]
) -> jnp.ndarray:
    """
    Convert GRF/COP predictions to joint torques using Jacobian.
    
    This is the DIFFERENTIABLE operation that maps cartesian forces to joint torques:
        tau = J^T * F
    
    Args:
        grf_cop_predictions: [12] array with [GRF_left(3), GRF_right(3), COP_left(3), COP_right(3)]
        jacobian: [nv, 6] or [nv, 12] Jacobian matrix relating cartesian to joint space
        body_ids: Body IDs for foot bodies
    
    Returns:
        joint_torques: [nv] array of joint torques
    """
    # Extract GRF and COP
    grf_left = grf_cop_predictions[:3]
    grf_right = grf_cop_predictions[3:6]
    cop_left = grf_cop_predictions[6:9]
    cop_right = grf_cop_predictions[9:12]
    
    # Option 1: If Jacobian is [nv, 12] - direct multiplication
    if jacobian.shape[1] == 12:
        # tau = J^T @ [GRF_left, GRF_right, COP_left, COP_right]
        joint_torques = jacobian.T @ grf_cop_predictions
    
    # Option 2: If Jacobian is [nv, 6] per contact - sum contributions
    elif jacobian.shape[1] == 6:
        # Combine GRF and COP into 6D wrench per foot
        wrench_left = jnp.concatenate([grf_left, cop_left])  # [6]: force + moment
        wrench_right = jnp.concatenate([grf_right, cop_right])
        
        # Sum both contributions
        joint_torques = jacobian.T @ wrench_left + jacobian.T @ wrench_right
    
    else:
        raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
    
    return joint_torques


def physics_informed_loss(
    predictions: jnp.ndarray,
    jacobian: jnp.ndarray,
    target_torques: jnp.ndarray,
    body_ids: Dict[str, int],
    lambda_torque: float = 1.0,
    lambda_grf: float = 0.1
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute physics-informed loss in torque space.
    
    Loss = lambda_torque * MSE(predicted_torques, target_torques) 
           + lambda_grf * regularization
    
    Args:
        predictions: [batch, 12] GRF/COP predictions (single timestep per sample)
        jacobian: [batch, nv, 12] or [nv, 12] Jacobian for each sample
        target_torques: [batch, nv] ground truth joint torques (single timestep)
        body_ids: Body IDs for feet
        lambda_torque: Weight for torque loss
        lambda_grf: Weight for GRF regularization (optional)
    
    Returns:
        total_loss: Scalar loss value
        metrics: Dict with loss components for logging
    """
    batch_size = predictions.shape[0]
    nv = target_torques.shape[-1]
    
    # Vectorize over batch
    def compute_torques_single(pred, jac):
        """Compute torques for single sample."""
        return jacobian_to_joint_torques(pred, jac, body_ids)
    
    # Handle different Jacobian shapes
    if jacobian.ndim == 3:  # [batch, nv, 12]
        # Vectorize over batch
        predicted_torques = jax.vmap(compute_torques_single)(predictions, jacobian)
    elif jacobian.ndim == 2:  # [nv, 12] - same Jacobian for all batch items
        # Broadcast Jacobian
        predicted_torques = jax.vmap(lambda pred: compute_torques_single(pred, jacobian))(predictions)
    else:
        raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
    
    # predicted_torques: [batch, nv]
    
    # Torque loss (MSE in joint space)
    torque_loss = jnp.mean((predicted_torques - target_torques) ** 2)
    
    # Optional: GRF regularization (e.g., encourage physically reasonable forces)
    grf_left = predictions[:, :3]
    grf_right = predictions[:, 3:6]
    
    # Penalty for unrealistic vertical forces (should be positive when in contact)
    grf_reg = jnp.mean(jnp.maximum(0, -grf_left[:, 2])) + jnp.mean(jnp.maximum(0, -grf_right[:, 2]))
    
    # Total loss
    total_loss = lambda_torque * torque_loss + lambda_grf * grf_reg
    
    # Metrics for logging
    metrics = {
        'total_loss': total_loss,
        'torque_loss': torque_loss,
        'grf_regularization': grf_reg,
        'torque_rmse': jnp.sqrt(torque_loss),
    }
    
    return total_loss, metrics


@partial(jax.jit, static_argnums=(4,))
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    jacobian: jnp.ndarray,
    body_ids: Dict[str, int],
    lambda_torque: float = 1.0,
    lambda_grf: float = 0.1
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step with gradient computation.
    
    Args:
        state: Training state (params, optimizer, etc.)
        batch: Dict with keys:
            - 'kinematics': [batch, seq_len, input_dim] (qpos, qvel, qacc)
            - 'target_torques': [batch, seq_len, nv]
        jacobian: Pre-calculated Jacobian
        body_ids: Body IDs for feet
        lambda_torque: Torque loss weight
        lambda_grf: GRF regularization weight
    
    Returns:
        new_state: Updated training state
        metrics: Loss and gradient metrics
    """
    def loss_fn(params):
        """Loss function for gradient computation."""
        # Forward pass
        predictions = state.apply_fn(
            {'params': params},
            batch['kinematics'],
            train=True,
            rngs={'dropout': jax.random.PRNGKey(0)}  # Use proper RNG in production
        )
        
        # Compute physics-informed loss
        loss, loss_metrics = physics_informed_loss(
            predictions,
            jacobian,
            batch['target_torques'],
            body_ids,
            lambda_torque,
            lambda_grf
        )
        
        return loss, (loss_metrics, predictions)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, predictions)), grads = grad_fn(state.params)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    # Add gradient norm to metrics
    grad_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)]))
    metrics['grad_norm'] = grad_norm
    
    return new_state, metrics


@partial(jax.jit, static_argnums=(3,))
def eval_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    jacobian: jnp.ndarray,
    body_ids: Dict[str, int],
    lambda_torque: float = 1.0,
    lambda_grf: float = 0.1
) -> Dict[str, jnp.ndarray]:
    """
    Evaluation step (no gradients).
    
    Args:
        state: Training state
        batch: Evaluation batch
        jacobian: Pre-calculated Jacobian
        body_ids: Body IDs
        lambda_torque: Torque loss weight
        lambda_grf: GRF regularization weight
    
    Returns:
        metrics: Evaluation metrics
    """
    # Forward pass (no dropout in eval)
    predictions = state.apply_fn(
        {'params': state.params},
        batch['kinematics'],
        train=False
    )
    
    # Compute loss
    loss, metrics = physics_informed_loss(
        predictions,
        jacobian,
        batch['target_torques'],
        body_ids,
        lambda_torque,
        lambda_grf
    )
    
    return metrics


def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4
) -> train_state.TrainState:
    """
    Create initial training state.
    
    Args:
        rng: Random key
        model: Flax model
        input_shape: (batch_size, seq_len, input_dim)
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
    
    Returns:
        Training state with initialized parameters
    """
    # Initialize model
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']
    
    # Create optimizer with learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=50000,
        end_value=1e-6
    )
    
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=weight_decay
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )


def train_epoch(
    state: train_state.TrainState,
    train_loader: Any,  # DataLoader or iterator
    jacobian_data: jnp.ndarray,
    body_ids: Dict[str, int],
    lambda_torque: float = 1.0,
    lambda_grf: float = 0.1
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        state: Training state
        train_loader: Training data loader
        jacobian_data: Pre-calculated Jacobian for all training samples
        body_ids: Body IDs
        lambda_torque: Torque loss weight
        lambda_grf: GRF regularization weight
    
    Returns:
        new_state: Updated training state
        epoch_metrics: Average metrics over epoch
    """
    epoch_metrics = {
        'total_loss': 0.0,
        'torque_loss': 0.0,
        'grf_regularization': 0.0,
        'grad_norm': 0.0
    }
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Get Jacobian for this batch
        # Assuming jacobian_data is indexed by batch or sample
        batch_jacobian = jacobian_data[batch_idx]  # Adjust indexing as needed
        
        # Training step
        state, metrics = train_step(
            state,
            batch,
            batch_jacobian,
            body_ids,
            lambda_torque,
            lambda_grf
        )
        
        # Accumulate metrics
        for key in epoch_metrics.keys():
            epoch_metrics[key] += float(metrics[key])
        num_batches += 1
    
    # Average metrics
    for key in epoch_metrics.keys():
        epoch_metrics[key] /= num_batches
    
    return state, epoch_metrics


# Example usage and training loop
if __name__ == "__main__":
    print("Physics-Informed Model Training Setup")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 16
    input_size = 100  # Your input size
    output_size = 12  # GRF and COP (2 feet × (3 force + 3 position))
    nv = 37  # Number of DOFs in model
    
    # Initialize model (REPLACE WITH YOUR TRANSFORMER!)
    model = YourModelPlaceholder(
        input_size=input_size,
        output_size=output_size
    )
    
    # Create training state
    rng = random.PRNGKey(0)
    state = create_train_state(
        rng,
        model,
        input_shape=(batch_size, input_size),
        learning_rate=1e-4
    )
    
    print(f"✓ Model initialized")
    print(f"  Input shape: ({batch_size}, {input_size})")
    print(f"  Output shape: ({batch_size}, {output_size})")
    print(f"  Parameters: {sum([p.size for p in jax.tree_util.tree_leaves(state.params)])}")
    
    # Dummy data for testing
    dummy_kinematics = jnp.ones((batch_size, input_size))
    dummy_jacobian = jnp.ones((nv, 12))
    dummy_torques = jnp.zeros((batch_size, nv))
    body_ids = {'calcn_l': 10, 'calcn_r': 15, 'pelvis': 1}
    
    dummy_batch = {
        'kinematics': dummy_kinematics,
        'target_torques': dummy_torques
    }
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    predictions = model.apply(
        {'params': state.params},
        dummy_kinematics,
        train=False
    )
    print(f"  Predictions shape: {predictions.shape}")
    
    # Test training step
    print("\n✓ Testing training step...")
    new_state, metrics = train_step(state, dummy_batch, dummy_jacobian, body_ids)
    print(f"  Loss: {metrics['total_loss']:.4f}")
    print(f"  Gradient norm: {metrics['grad_norm']:.4f}")
    
    print("\n" + "=" * 70)
    print("Setup complete! Ready for training.")
    print("\nNext steps:")
    print("1. Replace YourModelPlaceholder with your transformer")
    print("2. Load your actual data (qpos, qvel, qacc, Jacobian, tau)")
    print("3. Run training: python train_physics_transformer.py")
    print("4. Monitor torque_rmse to track physics accuracy")
