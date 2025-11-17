# Why Filtering Didn't Fix the Massive Torques

## Current Status

### Accelerations After Filtering:
```
Knee_R: [-89.030, 138.570] rad/s²  ← Still high!
Hip_R:  [-26.057, 39.451] rad/s²   ← Better
```

### Torques (Still Massive):
```
Knee_R: [-8605, 9188] N·m  ← Still ~60-100x too high!
Hip_R:  [-2970, 1912] N·m  ← Still ~20-30x too high!
```

### Expected Normal Values:
```
Knee: 50-150 N·m
Hip:  80-200 N·m
```

## Root Causes

### Issue 1: Filter Cutoff Still Too High ⚠️

Your 6 Hz cutoff reduced noise by 59%, but **138 rad/s² is still 2-3x higher** than typical gait (50-100 rad/s²).

**Solution**: Use a more aggressive filter:
```python
# Instead of:
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=6)

# Try:
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=3)  # More aggressive
```

### Issue 2: That Mysterious Line ⚠️⚠️

```python
qpos_matrix[1,:]=qpos_matrix[1,:]+.1  # ← THIS IS CAUSING PROBLEMS!
```

This adds 0.1 radians (~5.7°) to ALL joints in frame 1, causing:
- **Huge artificial velocities** between frames 0-2
- **Huge artificial accelerations** 
- **Massive spikes in torques** at t=0.01s

**This could be responsible for most of your high torques!**

### Issue 3: Missing Ground Reaction Forces 🦶

```python
# external_forces = jnp.zeros((nb, 6, num_timesteps))  # Commented out
# xfrc_applied = external_forces[:, :, t]              # Not applied
```

During stance phase:
- Ground supports ~70% of body weight
- Without GRFs, joints must generate ALL forces
- This inflates torques by 2-5x

### Issue 4: Coupled Joint Accelerations Not Filtered

You filtered `qacc_matrix` AFTER calculating coupled positions, but the coupled joints (indices 9, 10, 12, 13, 17-20, 24, 25, 27, 28, 32-35) have **zero acceleration** because they weren't in the original data!

This creates inconsistency.

## Solutions (Priority Order)

### SOLUTION 1: Remove That Bad Line (CRITICAL!)

```python
# DELETE THIS LINE:
# qpos_matrix[1,:]=qpos_matrix[1,:]+.1

# Or at minimum, comment it out completely
```

This alone could reduce your torques by 50-80%!

### SOLUTION 2: More Aggressive Filtering

```python
# Try cutoff = 3 Hz instead of 6 Hz
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=3, fs=100.5, order=4)
```

Expected result:
- Knee accel: 138 rad/s² → ~60-80 rad/s²
- Torques: ~9,000 N⋅m → ~300-500 N⋅m

### SOLUTION 3: Filter Positions and Recalculate Velocities/Accelerations

Instead of using the provided vel/acc data, derive them from filtered positions:

```python
from scipy.interpolate import UnivariateSpline

def smooth_and_differentiate(qpos_matrix, time_array):
    """
    Smooth positions and calculate consistent velocities/accelerations.
    """
    num_frames, num_dofs = qpos_matrix.shape
    qvel_matrix = np.zeros_like(qpos_matrix)
    qacc_matrix = np.zeros_like(qpos_matrix)
    
    # Smooth each DOF independently
    for dof in range(num_dofs):
        # Fit smoothing spline
        spline = UnivariateSpline(time_array, qpos_matrix[:, dof], s=0.001, k=5)
        
        # Get derivatives
        qpos_matrix[:, dof] = spline(time_array)  # Smoothed position
        qvel_matrix[:, dof] = spline.derivative(1)(time_array)
        qacc_matrix[:, dof] = spline.derivative(2)(time_array)
    
    return qpos_matrix, qvel_matrix, qacc_matrix

# Use it:
time_array = pos_data['time'].values
qpos_matrix, qvel_matrix, qacc_matrix = smooth_and_differentiate(qpos_matrix, time_array)
```

### SOLUTION 4: Add Ground Reaction Forces

Estimate GRFs from motion:

```python
def estimate_ground_forces(qpos, qvel, qacc, mass=65.6):
    """
    Estimate ground reaction forces during stance phase.
    """
    # Get pelvis vertical position and acceleration
    pelvis_y = qpos[1]  # Vertical position
    pelvis_ay = qacc[1]  # Vertical acceleration
    
    # Vertical force = m(g + a_y)
    F_vertical = mass * (9.81 + pelvis_ay)
    
    # Simple stance detection: pelvis height < threshold
    stance_threshold = 0.95  # meters
    is_stance = pelvis_y < stance_threshold
    
    if not is_stance:
        F_vertical = 0  # No ground contact
    
    # Create force array
    forces = np.zeros((mj_model.nbody, 6))
    
    # Apply to foot bodies (assuming indices)
    # You'll need to find the correct body indices for calcn_r and calcn_l
    foot_r_idx = 4  # Right foot body
    foot_l_idx = 9  # Left foot body
    
    # Split force between feet (simplified - could be more sophisticated)
    forces[foot_r_idx, 2] = F_vertical / 2  # Z-force (vertical)
    forces[foot_l_idx, 2] = F_vertical / 2
    
    return forces

# Then in your loop:
external_forces = estimate_ground_forces(
    qpos_matrix[t, :], 
    qvel_matrix[t, :], 
    qacc_matrix[t, :]
)

current_mjx_data = current_mjx_data.replace(
    qpos=qpos_matrix[t, :],
    qvel=qvel_matrix[t, :],
    qacc=qacc_matrix[t, :],
    xfrc_applied=jnp.array(external_forces)
)
```

## Quick Fix Implementation

Here's the minimum changes to try first:

```python
# 1. REMOVE THIS LINE (or comment it out):
# qpos_matrix[1,:]=qpos_matrix[1,:]+.1

# 2. Use more aggressive filtering:
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=3, fs=100.5, order=4)

# 3. Also filter velocities for consistency:
qvel_matrix = butter_lowpass_filter(qvel_matrix, cutoff=4, fs=100.5, order=4)

# 4. Gently filter positions to remove any discontinuities:
qpos_matrix = butter_lowpass_filter(qpos_matrix, cutoff=10, fs=100.5, order=2)
```

## Expected Results

### After Removing Bad Line + Aggressive Filtering:
```
Knee accelerations: ~50-80 rad/s²
Knee torques:       ~200-500 N·m  (better, but still high)
```

### After Adding GRFs:
```
Knee torques: ~50-150 N·m  ✓ REALISTIC!
Hip torques:  ~80-200 N·m  ✓ REALISTIC!
```

## Why Torques Are Still High (Even After Filtering)

The fundamental equation is:
```
τ = M(q)·q̈ + C(q,q̇)·q̇ + g(q) - J^T·F_ext
```

Where:
- M(q)·q̈ = Inertial forces (dominant)
- C(q,q̇)·q̇ = Coriolis/centrifugal
- g(q) = Gravity
- J^T·F_ext = External forces (GRFs)

Without F_ext (GRFs), the joints must generate:
1. Forces to accelerate body segments
2. Forces to overcome gravity
3. Forces to support body weight

This is why even with filtered data, torques are high without GRFs.

## Diagnostic: Check for Frame 1 Artifact

Run this to see if frame 1 is causing spikes:

```python
# After inverse dynamics
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(joint_forces_array[:, 11], label='Knee R')
plt.axvline(x=1, color='r', linestyle='--', label='Frame 1 (artifact?)')
plt.xlabel('Frame')
plt.ylabel('Torque (N⋅m)')
plt.legend()
plt.title('Check for Frame 1 Spike')
plt.savefig('frame1_check.png')
```

## Implementation Priority

1. **START HERE**: Remove `qpos_matrix[1,:]=qpos_matrix[1,:]+.1` line
2. **Then**: Try cutoff=3 Hz (more aggressive)
3. **Then**: Filter all kinematic data (pos, vel, acc) together
4. **Finally**: Add GRF estimation

Try them one at a time to see which has the biggest impact!

## Summary

Your filtering IS working (59% noise reduction), but:
- ❌ That mysterious +0.1 line is likely causing artifacts
- ❌ 6 Hz cutoff isn't aggressive enough (try 3 Hz)
- ❌ Missing ground reaction forces inflates torques by 2-5x
- ❌ Coupled joint accelerations aren't being properly handled

**Start by removing that `qpos_matrix[1,:]=qpos_matrix[1,:]+.1` line!**
