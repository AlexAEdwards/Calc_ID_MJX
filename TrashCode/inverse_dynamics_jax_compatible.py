"""
JAX-Compatible Inverse Dynamics Loop for ML Integration

This module provides a fully JAX-compatible implementation of the inverse dynamics
computation loop that can be:
1. JIT-compiled for speed
2. Differentiated with jax.grad or jax.jacobian
3. Vectorized with jax.vmap
4. Used in ML model forward passes

Key changes from the original:
- Removed all Python control flow (if/else on traced values)
- Removed side effects (time.time(), print, try/except)
- Used jax.lax.cond for conditional operations
- Used jax.lax.fori_loop or jax.lax.scan for loops
- Removed float() conversions
- Removed _impl attribute access
"""

import jax
import jax.numpy as jnp
from mujoco import mjx
from typing import Dict, Tuple
import numpy as np

@jax.jit
def compute_inverse_dynamics_single_frame(
    model,
    data,
    qacc: jnp.ndarray,
    qvel: jnp.ndarray,
    qpos: jnp.ndarray,
    external_forces: jnp.ndarray
) -> Tuple:
    """
    Compute inverse dynamics for a single frame.
    
    Args:
        model: MJX model
        data: MJX data
        qacc: Joint accelerations (nv,)
        qvel: Joint velocities (nv,)
        qpos: Joint positions (nq,)
        external_forces: External forces on all bodies (nbody, 6)
        
    Returns:
        Tuple of (updated_data, joint_forces, additional_outputs)
    """
    # Update data with new state
    data = data.replace(qpos=qpos, qvel=qvel, qacc=qacc, xfrc_applied=external_forces)
    
    # Compute inverse dynamics
    data = mjx.inverse(model, data)
    
    # Extract results
    joint_forces = data.qfrc_inverse
    qfrc_constraint = data.qfrc_constraint
    qfrc_bias = data.qfrc_bias
    qfrc_passive = data.qfrc_passive
    qfrc_actuator = data.qfrc_actuator
    
    # Package outputs
    outputs = {
        'qfrc_inverse': joint_forces,
        'qfrc_constraint': qfrc_constraint,
        'qfrc_bias': qfrc_bias,
        'qfrc_passive': qfrc_passive,
        'qfrc_actuator': qfrc_actuator,
        'qacc': data.qacc,
        'qpos': data.qpos,
        'qvel': data.qvel,
        'xpos': data.xpos
    }
    
    return data, joint_forces, outputs


@jax.jit
def compute_moment_arm_and_forces(
    ankle_pos: jnp.ndarray,
    cop: jnp.ndarray,
    grf: jnp.ndarray,
    moment: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute moment arm vector and total moment from GRF.
    
    Args:
        ankle_pos: Ankle position (3,)
        cop: Center of pressure (3,)
        grf: Ground reaction force (3,)
        moment: Moment from GRF data (3,)
        
    Returns:
        Tuple of (r_vec, distance, total_moment)
    """
    # Check if COP is non-zero (contact active)
    cop_magnitude = jnp.linalg.norm(cop)
    is_contact = cop_magnitude > 0.0
    
    # Compute moment arm: r = COP - ankle
    r_vec_contact = cop - ankle_pos
    distance_contact = jnp.linalg.norm(r_vec_contact)
    moment_added_contact = jnp.cross(r_vec_contact, grf)
    
    # No contact case
    r_vec_no_contact = jnp.zeros(3)
    distance_no_contact = 0.0
    moment_added_no_contact = jnp.zeros(3)
    
    # Use jax.lax.cond for conditional (JAX-compatible)
    r_vec = jax.lax.cond(
        is_contact,
        lambda: r_vec_contact,
        lambda: r_vec_no_contact
    )
    
    distance = jax.lax.cond(
        is_contact,
        lambda: distance_contact,
        lambda: distance_no_contact
    )
    
    moment_added = jax.lax.cond(
        is_contact,
        lambda: moment_added_contact,
        lambda: moment_added_no_contact
    )
    
    total_moment = moment + moment_added
    
    return r_vec, distance, total_moment


def process_single_timestep_body(
    t: int,
    carry: Tuple,
    model,
    qacc_matrix: jnp.ndarray,
    qvel_matrix: jnp.ndarray,
    qpos_matrix: jnp.ndarray,
    external_forces: jnp.ndarray,
    grf_left: jnp.ndarray,
    grf_right: jnp.ndarray,
    moment_left: jnp.ndarray,
    moment_right: jnp.ndarray,
    cop_matrix: jnp.ndarray,
    calcn_l_id: int,
    calcn_r_id: int,
    pelvis_id: int
) -> Dict:
    """
    Process a single timestep (non-JIT body - for testing).
    
    This function contains the logic for one timestep but is NOT JIT-compiled
    itself (the inner functions are). Use this for clarity or wrap the whole
    thing in @jax.jit for speed.
    """
    data = carry
    
    # Get body positions from previous state
    ankle_pos_l = data.xpos[calcn_l_id]
    ankle_pos_r = data.xpos[calcn_r_id]
    pelvis_pos = data.xpos[pelvis_id]
    
    # Extract COP (with coordinate transform: OpenSim -> MuJoCo)
    cop_l = jnp.array([cop_matrix[t, 4], -cop_matrix[t, 6], 0.0])
    cop_r = jnp.array([cop_matrix[t, 1], -cop_matrix[t, 3], 0.0])
    
    # Compute moment arms and total moments
    r_vec_l, dist_l, total_moment_l = compute_moment_arm_and_forces(
        ankle_pos_l, cop_l, grf_left[t], moment_left[t]
    )
    r_vec_r, dist_r, total_moment_r = compute_moment_arm_and_forces(
        ankle_pos_r, cop_r, grf_right[t], moment_right[t]
    )
    
    # Update external forces with computed moments
    # external_forces is (nbody, 6, num_timesteps) where 6 = [torque(3), force(3)]
    external_forces_t = external_forces[:, :, t].at[calcn_l_id, 3:6].set(total_moment_l)
    external_forces_t = external_forces_t.at[calcn_r_id, 3:6].set(total_moment_r)
    
    # Compute inverse dynamics
    data, joint_forces, outputs = compute_inverse_dynamics_single_frame(
        model, data,
        qacc_matrix[t],
        qvel_matrix[t],
        qpos_matrix[t],
        external_forces_t
    )
    
    # Package results for this timestep
    results = {
        'joint_forces': joint_forces,
        'outputs': outputs,
        'r_vec_l': r_vec_l,
        'r_vec_r': r_vec_r,
        'distance_l': dist_l,
        'distance_r': dist_r,
        'moment_l': total_moment_l,
        'moment_r': total_moment_r,
        'pelvis_pos': pelvis_pos,
        'ankle_pos_l': ankle_pos_l,
        'ankle_pos_r': ankle_pos_r,
        'cop_l': cop_l,
        'cop_r': cop_r,
    }
    
    return data, results


@jax.jit
def inverse_dynamics_scan_fn(carry, inputs):
    """
    Scan function for jax.lax.scan - processes one timestep.
    
    This is the JAX-compatible loop body that can be traced and differentiated.
    
    Args:
        carry: Current MJX data state
        inputs: Tuple of all inputs for this timestep
            (qacc, qvel, qpos, grf_l, grf_r, moment_l, moment_r, cop_data, ext_forces_slice)
            
    Returns:
        (new_carry, outputs) where outputs contains all computed values
    """
    data = carry
    qacc, qvel, qpos, grf_l, grf_r, moment_l, moment_r, cop_data, ext_forces_t, \
        calcn_l_id, calcn_r_id, pelvis_id = inputs
    
    # Get body positions from current state
    ankle_pos_l = data.xpos[calcn_l_id]
    ankle_pos_r = data.xpos[calcn_r_id]
    pelvis_pos = data.xpos[pelvis_id]
    
    # Extract COP (coordinate transform: OpenSim -> MuJoCo)
    cop_l = jnp.array([cop_data[3], -cop_data[5], 0.0])  # [4, -6, 0] in 0-indexed
    cop_r = jnp.array([cop_data[0], -cop_data[2], 0.0])  # [1, -3, 0] in 0-indexed
    
    # Compute moment arms and total moments
    r_vec_l, dist_l, total_moment_l = compute_moment_arm_and_forces(
        ankle_pos_l, cop_l, grf_l, moment_l
    )
    r_vec_r, dist_r, total_moment_r = compute_moment_arm_and_forces(
        ankle_pos_r, cop_r, grf_r, moment_r
    )
    
    # Update external forces with computed moments
    ext_forces_t = ext_forces_t.at[calcn_l_id, 3:6].set(total_moment_l)
    ext_forces_t = ext_forces_t.at[calcn_r_id, 3:6].set(total_moment_r)
    
    # Compute inverse dynamics
    data, joint_forces, outputs_dict = compute_inverse_dynamics_single_frame(
        data._model,  # Access model from data
        data,
        qacc,
        qvel,
        qpos,
        ext_forces_t
    )
    
    # Package outputs for this timestep
    outputs = {
        'joint_forces': joint_forces,
        'qfrc_constraint': outputs_dict['qfrc_constraint'],
        'qfrc_bias': outputs_dict['qfrc_bias'],
        'qfrc_passive': outputs_dict['qfrc_passive'],
        'qfrc_actuator': outputs_dict['qfrc_actuator'],
        'qacc': outputs_dict['qacc'],
        'qpos': outputs_dict['qpos'],
        'qvel': outputs_dict['qvel'],
        'r_vec_l': r_vec_l,
        'r_vec_r': r_vec_r,
        'distance_l': dist_l,
        'distance_r': dist_r,
        'pelvis_pos': pelvis_pos,
        'ankle_pos_l': ankle_pos_l,
        'ankle_pos_r': ankle_pos_r,
    }
    
    return data, outputs


def run_inverse_dynamics_jax(
    model,
    initial_data,
    qacc_matrix: jnp.ndarray,  # (num_timesteps, nv)
    qvel_matrix: jnp.ndarray,  # (num_timesteps, nv)
    qpos_matrix: jnp.ndarray,  # (num_timesteps, nq)
    grf_left: jnp.ndarray,     # (num_timesteps, 3)
    grf_right: jnp.ndarray,    # (num_timesteps, 3)
    moment_left: jnp.ndarray,  # (num_timesteps, 3)
    moment_right: jnp.ndarray, # (num_timesteps, 3)
    cop_matrix: jnp.ndarray,   # (num_timesteps, 10)
    external_forces: jnp.ndarray,  # (nbody, 6, num_timesteps)
    body_ids: Dict[str, int]
) -> Dict:
    """
    Run inverse dynamics for all timesteps using jax.lax.scan.
    
    This function is fully JAX-compatible and can be:
    - JIT-compiled
    - Differentiated
    - Vectorized with vmap
    - Used in ML forward passes
    
    Args:
        model: MJX model
        initial_data: Initial MJX data state
        qacc_matrix: Joint accelerations for all timesteps
        qvel_matrix: Joint velocities for all timesteps
        qpos_matrix: Joint positions for all timesteps
        grf_left: Left foot GRF for all timesteps
        grf_right: Right foot GRF for all timesteps
        moment_left: Left foot moments for all timesteps
        moment_right: Right foot moments for all timesteps
        cop_matrix: COP data for all timesteps
        external_forces: External forces array
        body_ids: Dictionary with 'calcn_l', 'calcn_r', 'pelvis' body IDs
        
    Returns:
        Dictionary containing all computed values across all timesteps
    """
    num_timesteps = qacc_matrix.shape[0]
    calcn_l_id = body_ids['calcn_l']
    calcn_r_id = body_ids['calcn_r']
    pelvis_id = body_ids['pelvis']
    
    # Prepare inputs for scan (stack all per-timestep inputs)
    # Each element of inputs_tuple will have shape (num_timesteps, ...)
    inputs_tuple = (
        qacc_matrix,
        qvel_matrix,
        qpos_matrix,
        grf_left,
        grf_right,
        moment_left,
        moment_right,
        cop_matrix,
        jnp.moveaxis(external_forces, 2, 0),  # (num_timesteps, nbody, 6)
        jnp.array([calcn_l_id] * num_timesteps),
        jnp.array([calcn_r_id] * num_timesteps),
        jnp.array([pelvis_id] * num_timesteps),
    )
    
    # Run scan over all timesteps
    final_data, all_outputs = jax.lax.scan(
        inverse_dynamics_scan_fn,
        initial_data,
        inputs_tuple
    )
    
    return all_outputs


# Example usage showing ML integration:
"""
# In your ML model forward pass:

class InverseDynamicsModel(nn.Module):
    model: mjx.Model
    
    def setup(self):
        # Initialize with MJX model
        pass
    
    @nn.compact
    def __call__(self, kinematics_params):
        # kinematics_params could be output from another NN
        qacc, qvel, qpos, grfs, cops = kinematics_params
        
        # Run JAX-compatible inverse dynamics
        results = run_inverse_dynamics_jax(
            self.model,
            initial_data,
            qacc, qvel, qpos,
            grf_left, grf_right,
            moment_left, moment_right,
            cops,
            external_forces,
            body_ids
        )
        
        # Use joint forces in loss function
        joint_forces = results['joint_forces']
        return joint_forces

# Can now differentiate:
grad_fn = jax.grad(lambda params: loss_fn(model(params)))
gradients = grad_fn(params)

# Or JIT compile:
fast_forward = jax.jit(model.apply)
"""
