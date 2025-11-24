"""
Visualization Script for Transformer Training Results

This script visualizes:
1. Training/validation loss curves
2. Torque RMSE over time
3. Sample predictions vs ground truth (GRF, COP, Torques)

Usage:
    python visualize_training.py --checkpoint checkpoints/best_model.npy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import jax
import jax.numpy as jnp

from TrashCode.physics_informed_transformer import GRFCOPTransformer, jacobian_to_joint_torques


def plot_training_history(history_path: str = "checkpoints/training_history.json"):
    """Plot training and validation curves."""
    print(f"Loading training history from {history_path}...")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Training Progress - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Torque RMSE curves
    axes[1].plot(epochs, history['train_torque_rmse'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_torque_rmse'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Torque RMSE (N·m)', fontsize=12)
    axes[1].set_title('Training Progress - Physics Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Target (1.0 N·m)')
    
    plt.tight_layout()
    
    output_path = Path("checkpoints/training_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {output_path}")
    plt.close()
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Final validation RMSE: {history['val_torque_rmse'][-1]:.4f} N·m")


def visualize_predictions(
    checkpoint_path: str,
    data_dir: str = "Results",
    num_samples: int = 3
):
    """
    Visualize model predictions on validation data.
    
    Args:
        checkpoint_path: Path to saved model parameters
        data_dir: Directory with validation data
        num_samples: Number of sample sequences to visualize
    """
    print(f"\nLoading model from {checkpoint_path}...")
    
    # Load model parameters
    params = np.load(checkpoint_path, allow_pickle=True).item()
    
    # Load validation data
    print("Loading validation data...")
    qpos = np.load(Path(data_dir) / "qpos_matrix.npy")
    qvel = np.load(Path(data_dir) / "qvel_matrix.npy")
    qacc = np.load(Path(data_dir) / "qacc_matrix.npy")
    grf_target = np.load(Path(data_dir) / "grf_matrix.npy")
    cop_target = np.load(Path(data_dir) / "cop_matrix.npy")
    
    jacobian_data = np.load(Path(data_dir) / "jacobian_data.npz")
    if 'J_calcn_l' in jacobian_data:
        jacobian = np.concatenate([jacobian_data['J_calcn_l'], jacobian_data['J_calcn_r']], axis=-1)
    else:
        jacobian = jacobian_data[list(jacobian_data.keys())[0]]
    
    nv = qpos.shape[1]
    
    # Concatenate kinematics
    kinematics = np.concatenate([qpos, qvel, qacc], axis=-1)
    
    # Initialize model
    model = GRFCOPTransformer(output_dim=12)
    
    # Select random samples
    seq_len = 200
    num_timesteps = kinematics.shape[0]
    max_start = num_timesteps - seq_len
    
    np.random.seed(42)
    start_indices = np.random.choice(max_start, size=num_samples, replace=False)
    
    body_ids = {'calcn_l': 10, 'calcn_r': 15, 'pelvis': 1}  # Adjust as needed
    
    for sample_idx, start_idx in enumerate(start_indices):
        print(f"\nVisualizing sample {sample_idx + 1}/{num_samples} (frames {start_idx}-{start_idx+seq_len})...")
        
        # Extract sequence
        kinematics_seq = kinematics[start_idx:start_idx+seq_len]
        grf_seq = grf_target[start_idx:start_idx+seq_len]
        cop_seq = cop_target[start_idx:start_idx+seq_len]
        jacobian_seq = jacobian[start_idx:start_idx+seq_len]
        
        # Add batch dimension
        kinematics_batch = jnp.array(kinematics_seq[None, :, :])
        
        # Predict
        predictions = model.apply(
            {'params': params},
            kinematics_batch,
            train=False
        )
        predictions = np.array(predictions[0])  # Remove batch dim
        
        # Extract predictions
        grf_pred_left = predictions[:, 0:3]
        grf_pred_right = predictions[:, 3:6]
        cop_pred_left = predictions[:, 6:9]
        cop_pred_right = predictions[:, 9:12]
        
        # Extract targets
        grf_target_left = grf_seq[:, 0:3]
        grf_target_right = grf_seq[:, 3:6]
        cop_target_left = cop_seq[:, 0:3]
        cop_target_right = cop_seq[:, 3:6]
        
        # Compute torques
        torques_pred = []
        for t in range(seq_len):
            tau = jacobian_to_joint_torques(
                jnp.array(predictions[t]),
                jnp.array(jacobian_seq[t]),
                body_ids
            )
            torques_pred.append(np.array(tau))
        torques_pred = np.array(torques_pred)
        
        # Time vector
        time_vec = np.arange(seq_len) * 0.01  # Assuming 100 Hz
        
        # Create figure
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        
        # Row 1: GRF Left
        for i, label in enumerate(['Fx', 'Fy', 'Fz']):
            axes[0, i].plot(time_vec, grf_target_left[:, i], 'r--', linewidth=2, label='Target', alpha=0.7)
            axes[0, i].plot(time_vec, grf_pred_left[:, i], 'b-', linewidth=2, label='Predicted', alpha=0.7)
            axes[0, i].set_ylabel('Force (N)', fontsize=10)
            axes[0, i].set_title(f'GRF Left - {label}', fontsize=11, fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend(fontsize=8)
        
        # Row 2: GRF Right
        for i, label in enumerate(['Fx', 'Fy', 'Fz']):
            axes[1, i].plot(time_vec, grf_target_right[:, i], 'r--', linewidth=2, label='Target', alpha=0.7)
            axes[1, i].plot(time_vec, grf_pred_right[:, i], 'b-', linewidth=2, label='Predicted', alpha=0.7)
            axes[1, i].set_ylabel('Force (N)', fontsize=10)
            axes[1, i].set_title(f'GRF Right - {label}', fontsize=11, fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend(fontsize=8)
        
        # Row 3: COP Left
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[2, i].plot(time_vec, cop_target_left[:, i], 'r--', linewidth=2, label='Target', alpha=0.7)
            axes[2, i].plot(time_vec, cop_pred_left[:, i], 'b-', linewidth=2, label='Predicted', alpha=0.7)
            axes[2, i].set_ylabel('Position (m)', fontsize=10)
            axes[2, i].set_title(f'COP Left - {label}', fontsize=11, fontweight='bold')
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].legend(fontsize=8)
        
        # Row 4: Sample joint torques
        sample_joints = [0, nv//2, nv-1]  # First, middle, last joint
        for plot_idx, joint_idx in enumerate(sample_joints):
            axes[3, plot_idx].plot(time_vec, torques_pred[:, joint_idx], 'g-', linewidth=2, label='Predicted Torque')
            axes[3, plot_idx].set_xlabel('Time (s)', fontsize=10)
            axes[3, plot_idx].set_ylabel('Torque (N·m)', fontsize=10)
            axes[3, plot_idx].set_title(f'Joint {joint_idx} Torque', fontsize=11, fontweight='bold')
            axes[3, plot_idx].grid(True, alpha=0.3)
            axes[3, plot_idx].legend(fontsize=8)
        
        plt.tight_layout()
        
        output_path = Path("checkpoints") / f"predictions_sample_{sample_idx+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
        # Calculate errors
        grf_rmse = np.sqrt(np.mean((predictions[:, :6] - np.concatenate([grf_target_left, grf_target_right], axis=-1))**2))
        cop_rmse = np.sqrt(np.mean((predictions[:, 6:] - np.concatenate([cop_target_left, cop_target_right], axis=-1))**2))
        
        print(f"  GRF RMSE: {grf_rmse:.3f} N")
        print(f"  COP RMSE: {cop_rmse:.3f} m")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.npy',
                        help='Path to model checkpoint')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json',
                        help='Path to training history')
    parser.add_argument('--data_dir', type=str, default='Results',
                        help='Directory with validation data')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of prediction samples to visualize')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Training Results Visualization")
    print("="*70)
    
    # Plot training curves
    if Path(args.history).exists():
        plot_training_history(args.history)
    else:
        print(f"⚠ Training history not found at {args.history}")
    
    # Visualize predictions
    if Path(args.checkpoint).exists():
        visualize_predictions(args.checkpoint, args.data_dir, args.num_samples)
    else:
        print(f"⚠ Checkpoint not found at {args.checkpoint}")
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("Check checkpoints/ directory for generated plots")
    print("="*70)


if __name__ == "__main__":
    main()
