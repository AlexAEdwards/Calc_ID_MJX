"""
Validation Script: Test JAX-Compatible Inverse Dynamics

This script:
1. Loads saved kinematics data (qpos, qvel, qacc)
2. Loads saved external forces and Jacobian data
3. Runs the JAX-compatible inverse dynamics
4. Compares results with reference tau.csv
5. Plots comparison and calculates RMSE

Usage:
    python validate_jax_inverse_dynamics.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
import time
from typing import Dict

# Import the JAX-compatible functions
from TrashCode.inverse_dynamics_jax_compatible import (
    compute_inverse_dynamics_single_frame,
    compute_moment_arm_and_forces,
    run_inverse_dynamics_jax
)


def load_saved_data(results_dir: str = "Results"):
    """Load all saved data from previous run."""
    print("Loading saved data...")
    
    data_dir = Path(results_dir)
    
    # Load kinematics
    qpos_matrix = np.load(data_dir / "qpos_matrix.npy")
    qvel_matrix = np.load(data_dir / "qvel_matrix.npy")
    qacc_matrix = np.load(data_dir / "qacc_matrix.npy")
    
    # Load external forces
    external_forces = np.load(data_dir / "external_forces.npy")
    
    # Load Jacobian data
    jacobian_data = np.load(data_dir / "jacobian_data.npz")
    
    # Load GRF and COP data
    grf_matrix = np.load(data_dir / "grf_matrix.npy")
    moment_matrix = np.load(data_dir / "moment_matrix.npy")
    cop_matrix = np.load(data_dir / "cop_matrix.npy")
    
    print(f"  qpos shape: {qpos_matrix.shape}")
    print(f"  qvel shape: {qvel_matrix.shape}")
    print(f"  qacc shape: {qacc_matrix.shape}")
    print(f"  external_forces shape: {external_forces.shape}")
    print(f"  grf_matrix shape: {grf_matrix.shape}")
    print(f"  Jacobian data keys: {list(jacobian_data.keys())}")
    
    return {
        'qpos_matrix': qpos_matrix,
        'qvel_matrix': qvel_matrix,
        'qacc_matrix': qacc_matrix,
        'external_forces': external_forces,
        'jacobian_data': dict(jacobian_data),
        'grf_matrix': grf_matrix,
        'moment_matrix': moment_matrix,
        'cop_matrix': cop_matrix
    }


def load_reference_tau(data_path: str = "PatientData/Falisse_2017_subject_01/tau.csv"):
    """Load reference tau data from CSV."""
    print(f"\nLoading reference tau from {data_path}...")
    
    tau_df = pd.read_csv(data_path)
    print(f"  Tau shape: {tau_df.shape}")
    print(f"  Columns: {list(tau_df.columns[:5])}... (showing first 5)")
    
    return tau_df


def setup_mjx_model(model_path: str = "Model/fullbody_falisse_corrected_scaled_FINAL_MUSCLES.xml"):
    """Load and setup MuJoCo/MJX model."""
    print(f"\nLoading MuJoCo model from {model_path}...")
    
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Convert to MJX
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    
    # Get body IDs
    body_ids = {
        'calcn_l': mj_model.body('calcn_l').id,
        'calcn_r': mj_model.body('calcn_r').id,
        'pelvis': mj_model.body('pelvis').id
    }
    
    print(f"  Model has {mjx_model.nv} DOFs")
    print(f"  Body IDs: calcn_l={body_ids['calcn_l']}, calcn_r={body_ids['calcn_r']}, pelvis={body_ids['pelvis']}")
    
    return mj_model, mjx_model, mjx_data, body_ids


def run_jax_inverse_dynamics_simple(
    mjx_model,
    mjx_data,
    qpos_matrix: np.ndarray,
    qvel_matrix: np.ndarray,
    qacc_matrix: np.ndarray,
    external_forces: np.ndarray,
    body_ids: Dict[str, int]
):
    """
    Run JAX-compatible inverse dynamics in a simple loop.
    
    This version doesn't use scan yet - just validates the core function works.
    """
    print("\nRunning JAX-compatible inverse dynamics...")
    
    num_timesteps = qacc_matrix.shape[0]
    nv = mjx_model.nv
    
    # Convert to JAX arrays
    qacc_jax = jnp.array(qacc_matrix)
    qvel_jax = jnp.array(qvel_matrix)
    qpos_jax = jnp.array(qpos_matrix)
    external_forces_jax = jnp.array(external_forces)
    
    # Pre-allocate results
    joint_forces_all = np.zeros((num_timesteps, nv))
    qfrc_constraint_all = np.zeros((num_timesteps, nv))
    qfrc_bias_all = np.zeros((num_timesteps, nv))
    
    # Run for each timestep
    print(f"Processing {num_timesteps} timesteps...")
    start_time = time.time()
    
    current_data = mjx_data
    
    for t in range(num_timesteps):
        if t % 50 == 0:
            print(f"  Timestep {t}/{num_timesteps}...")
        
        # Run inverse dynamics for this timestep
        current_data, joint_forces, outputs = compute_inverse_dynamics_single_frame(
            mjx_model,
            current_data,
            qacc_jax[t, :],
            qvel_jax[t, :],
            qpos_jax[t, :],
            external_forces_jax[:, :, t]
        )
        
        # Store results (convert to numpy)
        joint_forces_all[t, :] = np.array(joint_forces)
        qfrc_constraint_all[t, :] = np.array(outputs['qfrc_constraint'])
        qfrc_bias_all[t, :] = np.array(outputs['qfrc_bias'])
    
    duration = time.time() - start_time
    print(f"✓ Completed in {duration:.2f}s ({duration/num_timesteps*1000:.2f} ms/frame)")
    
    return {
        'joint_forces': joint_forces_all,
        'qfrc_constraint': qfrc_constraint_all,
        'qfrc_bias': qfrc_bias_all
    }


def map_tau_columns_to_model(tau_df: pd.DataFrame, mj_model):
    """
    Map tau.csv column names to model joint indices.
    
    Returns dict: {joint_name: (tau_column_name, model_dof_index)}
    """
    mapping = {}
    tau_columns = [col for col in tau_df.columns if col != 'time']
    
    for tau_col in tau_columns:
        # Try to find matching joint in model
        # tau.csv might have names like 'hip_flexion_r', model might have 'hip_flexion_r'
        for i in range(mj_model.nv):
            joint_name = mj_model.joint(mj_model.dof_jntid[i]).name
            
            # Match if names are similar
            if tau_col.lower() in joint_name.lower() or joint_name.lower() in tau_col.lower():
                mapping[joint_name] = (tau_col, i)
                break
    
    return mapping


def calculate_rmse(computed: np.ndarray, reference: np.ndarray) -> float:
    """Calculate root mean square error."""
    return np.sqrt(np.mean((computed - reference)**2))


def plot_comparison(
    computed_forces: np.ndarray,
    tau_df: pd.DataFrame,
    mj_model,
    time_vector: np.ndarray,
    output_dir: str = "Results"
):
    """
    Plot comparison between computed forces and reference tau.
    """
    print("\nPlotting comparison...")
    
    # Map columns
    mapping = map_tau_columns_to_model(tau_df, mj_model)
    
    if len(mapping) == 0:
        print("⚠ Warning: Could not map any tau columns to model joints")
        print("  Tau columns:", list(tau_df.columns[:10]))
        print("  Model joints:", [mj_model.joint(mj_model.dof_jntid[i]).name for i in range(min(10, mj_model.nv))])
        return
    
    print(f"Mapped {len(mapping)} joints")
    
    # Create figures (6 DOFs per figure)
    joints_per_fig = 6
    num_figs = int(np.ceil(len(mapping) / joints_per_fig))
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    joint_items = list(mapping.items())
    
    for fig_idx in range(num_figs):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        start_idx = fig_idx * joints_per_fig
        end_idx = min(start_idx + joints_per_fig, len(joint_items))
        
        for plot_idx, joint_idx in enumerate(range(start_idx, end_idx)):
            joint_name, (tau_col, dof_idx) = joint_items[joint_idx]
            ax = axes[plot_idx]
            
            # Get data
            computed = computed_forces[:, dof_idx]
            reference = tau_df[tau_col].values[:len(computed)]
            
            # Calculate RMSE
            rmse = calculate_rmse(computed, reference)
            
            # Plot
            ax.plot(time_vector, computed, 'b-', linewidth=2, label='JAX Computed', alpha=0.7)
            ax.plot(time_vector, reference, 'r--', linewidth=2, label='Reference (tau.csv)', alpha=0.7)
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Torque (N·m)', fontsize=10)
            ax.set_title(f'{joint_name}\nRMSE: {rmse:.3f} N·m', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(end_idx - start_idx, joints_per_fig):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = output_path / f'jax_validation_comparison_{fig_idx+1}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig_path}")
        plt.close()
    
    # Calculate overall statistics
    print("\nOverall Statistics:")
    all_rmse = []
    for joint_name, (tau_col, dof_idx) in mapping.items():
        computed = computed_forces[:, dof_idx]
        reference = tau_df[tau_col].values[:len(computed)]
        rmse = calculate_rmse(computed, reference)
        all_rmse.append(rmse)
    
    print(f"  Mean RMSE: {np.mean(all_rmse):.3f} N·m")
    print(f"  Median RMSE: {np.median(all_rmse):.3f} N·m")
    print(f"  Max RMSE: {np.max(all_rmse):.3f} N·m")
    print(f"  Min RMSE: {np.min(all_rmse):.3f} N·m")


def main():
    """Main validation script."""
    print("="*70)
    print("JAX-Compatible Inverse Dynamics Validation")
    print("="*70)
    
    # Load saved data
    data = load_saved_data()
    
    # Load reference tau
    tau_df = load_reference_tau()
    
    # Setup model
    mj_model, mjx_model, mjx_data, body_ids = setup_mjx_model()
    
    # Run JAX inverse dynamics
    results = run_jax_inverse_dynamics_simple(
        mjx_model,
        mjx_data,
        data['qpos_matrix'],
        data['qvel_matrix'],
        data['qacc_matrix'],
        data['external_forces'],
        body_ids
    )
    
    # Create time vector
    num_timesteps = data['qacc_matrix'].shape[0]
    time_vector = np.linspace(0, num_timesteps * 0.01, num_timesteps)  # Assuming 0.01s timestep
    
    # Plot comparison
    plot_comparison(
        results['joint_forces'],
        tau_df,
        mj_model,
        time_vector
    )
    
    # Save results
    output_dir = Path("Results")
    np.save(output_dir / "jax_joint_forces.npy", results['joint_forces'])
    print(f"\n✓ Saved JAX results to {output_dir / 'jax_joint_forces.npy'}")
    
    print("\n" + "="*70)
    print("Validation Complete!")
    print("="*70)
    print("\nCheck the generated plots in Results/ to compare:")
    print("  - JAX computed forces (blue solid)")
    print("  - Reference tau.csv (red dashed)")
    print("\nIf RMSE values are low (< 1.0 N·m), the JAX implementation is accurate!")


if __name__ == "__main__":
    main()
