"""
Simplified Training Script for Physics-Informed Model

This script:
1. Loads your pre-calculated data (qpos, qvel, qacc, Jacobian, tau)
2. Prepares data for training
3. Trains your model with physics-informed loss
4. Saves checkpoints

Usage:
    python train_physics_transformer.py --epochs 100
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pathlib import Path
import argparse
from typing import Dict, Tuple, List, Optional
import time
import json

from TrashCode.physics_informed_transformer import (
    YourModelPlaceholder,
    create_train_state,
    train_step,
    eval_step
)


def load_training_data(data_dir: str = "Results") -> Dict[str, np.ndarray]:
    """
    Load all saved training data.
    
    Returns:
        data: Dict containing:
            - qpos_matrix: [num_timesteps, nv]
            - qvel_matrix: [num_timesteps, nv]
            - qacc_matrix: [num_timesteps, nv]
            - jacobian_data: [num_timesteps, nv, 12] or similar
            - target_torques: [num_timesteps, nv] (from tau.csv)
            - grf_matrix: [num_timesteps, 6] (optional, for comparison)
            - cop_matrix: [num_timesteps, 6] (optional, for comparison)
    """
    print("Loading training data...")
    data_path = Path(data_dir)
    
    # Load kinematics
    qpos_matrix = np.load(data_path / "qpos_matrix.npy")
    qvel_matrix = np.load(data_path / "qvel_matrix.npy")
    qacc_matrix = np.load(data_path / "qacc_matrix.npy")
    
    print(f"  Kinematics shapes:")
    print(f"    qpos: {qpos_matrix.shape}")
    print(f"    qvel: {qvel_matrix.shape}")
    print(f"    qacc: {qacc_matrix.shape}")
    
    # Load Jacobian data
    jacobian_data = np.load(data_path / "jacobian_data.npz")
    print(f"  Jacobian data keys: {list(jacobian_data.keys())}")
    
    # Extract Jacobians for left and right foot
    # Adjust based on your actual data structure
    if 'J_calcn_l' in jacobian_data and 'J_calcn_r' in jacobian_data:
        J_left = jacobian_data['J_calcn_l']  # [num_timesteps, nv, 6]
        J_right = jacobian_data['J_calcn_r']  # [num_timesteps, nv, 6]
        
        # Concatenate to form [num_timesteps, nv, 12]
        jacobian_matrix = np.concatenate([J_left, J_right], axis=-1)
    else:
        # If stored differently, adjust accordingly
        print("  Warning: Expected 'J_calcn_l' and 'J_calcn_r' in jacobian_data")
        print(f"  Available keys: {list(jacobian_data.keys())}")
        # Use first available Jacobian as placeholder
        jacobian_matrix = jacobian_data[list(jacobian_data.keys())[0]]
    
    print(f"    Jacobian: {jacobian_matrix.shape}")
    
    # Load GRF and COP (these will be our prediction targets, but also used for validation)
    grf_matrix = np.load(data_path / "grf_matrix.npy")
    cop_matrix = np.load(data_path / "cop_matrix.npy")
    
    print(f"  Ground truth:")
    print(f"    GRF: {grf_matrix.shape}")
    print(f"    COP: {cop_matrix.shape}")
    
    # Load target torques (from tau.csv or computed inverse dynamics)
    # First, try to load from saved inverse dynamics
    if (data_path / "jax_joint_forces.npy").exists():
        target_torques = np.load(data_path / "jax_joint_forces.npy")
        print(f"    Torques (from JAX ID): {target_torques.shape}")
    else:
        # Load from tau.csv
        print("  Loading tau from CSV...")
        import pandas as pd
        tau_df = pd.read_csv("PatientData/Falisse_2017_subject_01/tau.csv")
        # Extract numeric columns (skip 'time' column)
        tau_columns = [col for col in tau_df.columns if col != 'time']
        target_torques = tau_df[tau_columns].values
        print(f"    Torques (from CSV): {target_torques.shape}")
    
    return {
        'qpos_matrix': qpos_matrix,
        'qvel_matrix': qvel_matrix,
        'qacc_matrix': qacc_matrix,
        'jacobian_matrix': jacobian_matrix,
        'target_torques': target_torques,
        'grf_matrix': grf_matrix,
        'cop_matrix': cop_matrix
    }


def prepare_data(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare your data for training.
    
    CUSTOMIZE THIS FUNCTION to match your data format!
    
    Args:
        data: Dict with 'qpos_matrix', 'qvel_matrix', 'qacc_matrix',
              'jacobian_matrix', 'target_torques', etc.
    
    Returns:
        input_features: [num_samples, input_size] - Your model inputs
        jacobian: [num_samples, nv, 12] or [nv, 12] - Jacobian matrices
        grf_cop_target: [num_samples, 12] - Target GRF/COP (optional)
        torque_target: [num_samples, nv] - Target torques
    """
    print("\nPreparing data...")
    
    qpos = data['qpos_matrix']
    qvel = data['qvel_matrix']
    qacc = data['qacc_matrix']
    jacobian = data['jacobian_matrix']
    torques = data['target_torques']
    
    num_timesteps = qpos.shape[0]
    nv = qpos.shape[1]
    
    print(f"  Loaded {num_timesteps} timesteps, {nv} DOFs")
    
    # CUSTOMIZE: Prepare your input features
    # Example: concatenate kinematics
    input_features = np.concatenate([qpos, qvel, qacc], axis=-1)  # [T, 3*nv]
    
    # CUSTOMIZE: If you need different input format, change here
    # E.g., flatten first 100 timesteps:
    # input_features = input_features[:100].flatten()
    
    # GRF/COP target (if you have it)
    if 'grf_matrix' in data and 'cop_matrix' in data:
        grf_cop_target = np.concatenate([data['grf_matrix'], data['cop_matrix']], axis=-1)
    else:
        grf_cop_target = np.zeros((num_timesteps, 12))
    
    print(f"  Input features shape: {input_features.shape}")
    print(f"  Jacobian shape: {jacobian.shape}")
    print(f"  Torques shape: {torques.shape}")
    
    return input_features, jacobian, grf_cop_target, torques


def create_batches(
    input_features: np.ndarray,
    jacobians: np.ndarray,
    grf_cop: np.ndarray,
    torques: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: Optional[jax.random.PRNGKey] = None
) -> List[Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]:
    """
    Create batches for training.
    
    Args:
        input_features: [num_samples, input_size] 
        jacobians: [num_samples, nv, 12] or [nv, 12]
        grf_cop: [num_samples, 12]
        torques: [num_samples, nv]
        batch_size: Batch size
        shuffle: Whether to shuffle
        rng: Random key for shuffling
    
    Returns:
        List of (batch_dict, batch_jacobian) tuples
    """
    num_samples = input_features.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle and rng is not None:
        indices = jax.random.permutation(rng, indices)
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        if len(batch_indices) < batch_size:
            # Pad last batch if needed
            pad_size = batch_size - len(batch_indices)
            batch_indices = np.concatenate([batch_indices, np.zeros(pad_size, dtype=int)])
        
        batch = {
            'kinematics': jnp.array(input_features[batch_indices]),
            'target_torques': jnp.array(torques[batch_indices]),
            'grf_cop_target': jnp.array(grf_cop[batch_indices])
        }
        
        # Handle Jacobian shape
        if jacobians.ndim == 3:  # [num_samples, nv, 12]
            batch_jacobian = jnp.array(jacobians[batch_indices])
        else:  # [nv, 12] - broadcast to all samples
            batch_jacobian = jnp.array(jacobians)
        
        batches.append((batch, batch_jacobian))
    
    return batches


def train_model(
    args: argparse.Namespace,
    train_data: Tuple,
    val_data: Tuple,
    body_ids: Dict[str, int]
):
    """
    Main training loop.
    
    Args:
        args: Command line arguments
        train_data: (input_features, jacobians, grf_cop, torques)
        val_data: Same format as train_data
        body_ids: Body IDs for feet
    """
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # Unpack data
    train_inputs, train_jacobians, train_grf_cop, train_torques = train_data
    val_inputs, val_jacobians, val_grf_cop, val_torques = val_data
    
    # Get dimensions
    num_train = train_inputs.shape[0]
    input_size = train_inputs.shape[1]
    nv = train_torques.shape[1]
    
    print(f"\nDataset info:")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {val_inputs.shape[0]}")
    print(f"  Input size: {input_size}")
    print(f"  Output size: 12 (GRF + COP)")
    print(f"  Number of DOFs: {nv}")
    
    # Initialize YOUR MODEL HERE
    model = YourModelPlaceholder(
        input_size=input_size,
        output_size=12
    )
    
    # Create training state
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)
    
    state = create_train_state(
        init_rng,
        model,
        input_shape=(args.batch_size, input_size),
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    num_params = sum([p.size for p in jax.tree_util.tree_leaves(state.params)])
    print(f"\n✓ Model initialized with {num_params:,} parameters")
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_torque_rmse': [],
        'val_torque_rmse': []
    }
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Create batches
        rng, batch_rng = random.split(rng)
        train_batches = create_batches(
            train_inputs,
            train_jacobians,
            train_grf_cop,
            train_torques,
            args.batch_size,
            shuffle=True,
            rng=batch_rng
        )
        
        # Training
        train_metrics = {
            'total_loss': 0.0,
            'torque_rmse': 0.0,
            'grad_norm': 0.0
        }
        
        for batch, batch_jacobian in train_batches:
            state, metrics = train_step(
                state,
                batch,
                batch_jacobian,
                body_ids,
                args.lambda_torque,
                args.lambda_grf
            )
            
            for key in train_metrics.keys():
                train_metrics[key] += float(metrics[key])
        
        # Average metrics
        num_train_batches = len(train_batches)
        for key in train_metrics.keys():
            train_metrics[key] /= num_train_batches
        
        # Validation
        val_batches = create_batches(
            val_inputs,
            val_jacobians,
            val_grf_cop,
            val_torques,
            args.batch_size,
            shuffle=False
        )
        
        val_metrics = {
            'total_loss': 0.0,
            'torque_rmse': 0.0
        }
        
        for batch, batch_jacobian in val_batches:
            metrics = eval_step(
                state,
                batch,
                batch_jacobian,
                body_ids,
                args.lambda_torque,
                args.lambda_grf
            )
            
            for key in val_metrics.keys():
                val_metrics[key] += float(metrics[key])
        
        num_val_batches = len(val_batches)
        for key in val_metrics.keys():
            val_metrics[key] /= num_val_batches
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_torque_rmse'].append(train_metrics['torque_rmse'])
        history['val_torque_rmse'].append(val_metrics['torque_rmse'])
        
        # Print progress
        epoch_time = time.time() - epoch_start
        if epoch % args.print_every == 0 or epoch == args.epochs - 1:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Torque RMSE: {train_metrics['torque_rmse']:.4f} N·m, "
                  f"Grad Norm: {train_metrics['grad_norm']:.4f}")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Torque RMSE: {val_metrics['torque_rmse']:.4f} N·m")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint_path = Path(args.output_dir) / "best_model.npy"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(checkpoint_path, state.params)
            if epoch % args.print_every == 0:
                print(f"  ✓ Saved best model")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.npy"
            np.save(checkpoint_path, state.params)
    
    # Save final model and history
    final_path = Path(args.output_dir) / "final_model.npy"
    np.save(final_path, state.params)
    
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final validation torque RMSE: {history['val_torque_rmse'][-1]:.4f} N·m")
    print(f"  Models saved to: {args.output_dir}")
    print("="*70)
    
    return state, history


def main():
    parser = argparse.ArgumentParser(
        description="Train Your Custom Model with Physics-Informed Loss"
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Results',
                        help='Directory containing saved data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lambda_torque', type=float, default=1.0,
                        help='Physics loss weight')
    parser.add_argument('--lambda_grf', type=float, default=0.1,
                        help='GRF regularization weight')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_every', type=int, default=1,
                        help='Print progress every N epochs')
    
    args = parser.parse_args()
    
    # Load data
    print("\n" + "="*70)
    print("Loading Training Data")
    print("="*70)
    data = load_training_data(args.data_dir)
    
    # Prepare data
    # CUSTOMIZE THIS based on how you want to structure your inputs
    train_inputs, train_jacobians, train_grf_cop, train_torques = prepare_data(data)
    
    # For now, use same data for train and validation
    # TODO: Split into multiple samples/trials for proper train/val split
    val_inputs, val_jacobians, val_grf_cop, val_torques = prepare_data(data)
    
    print(f"\n⚠ Currently using same data for train and validation")
    print(f"  In production, split into separate trials/subjects")
    
    # Package data
    train_data = (train_inputs, train_jacobians, train_grf_cop, train_torques)
    val_data = (val_inputs, val_jacobians, val_grf_cop, val_torques)
    
    # Body IDs for feet (adjust to match your model)
    body_ids = {
        'calcn_l': 10,  # Left foot body ID
        'calcn_r': 15,  # Right foot body ID
        'pelvis': 1     # Pelvis body ID (reference frame)
    }
    
    # Train model
    state, history = train_model(args, train_data, val_data, body_ids)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
