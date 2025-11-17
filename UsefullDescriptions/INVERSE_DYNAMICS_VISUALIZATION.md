# Inverse Dynamics Control Forces Visualization

## Overview

The script now automatically generates visualizations of the control forces calculated through inverse dynamics. These forces represent the joint torques/forces required to achieve the observed patient motion.

## Generated Plots

### 1. **inverse_dynamics_forces.png**
A focused 4x2 grid showing the 8 most important joints:

**Right Leg:**
- Hip Flexion R (DOF 6)
- Hip Adduction R (DOF 7)
- Knee Angle R (DOF 11)
- Ankle Angle R (DOF 14)

**Left Leg:**
- Hip Flexion L (DOF 21)
- Hip Adduction L (DOF 22)
- Knee Angle L (DOF 26)
- Ankle Angle L (DOF 29)

Each subplot shows:
- X-axis: Time (seconds) from 0 to ~2.06s
- Y-axis: Force/Torque (N·m)
- Blue line: Control force trajectory
- Dashed line at y=0 for reference
- Grid for easier reading

### 2. **all_joint_forces.png**
A comprehensive plot showing all 39 DOFs on a single graph:
- Overlaid line plots for each joint
- Color-coded with transparency
- Legend identifying each joint
- Useful for:
  - Comparing relative magnitudes across joints
  - Identifying dominant forces
  - Spotting patterns across the gait cycle

## Joint Force Interpretation

### What the Forces Represent:
- **Positive values**: Torque/force in the positive joint direction
- **Negative values**: Torque/force in the negative joint direction
- **Magnitude**: Strength of the control force required

### Expected Patterns:
- **Hip Flexion**: Should show alternating pattern during gait
- **Knee Angle**: Peak forces during stance phase (weight acceptance)
- **Ankle Angle**: Push-off forces during late stance
- **Pelvis DOFs**: Small forces for balance/stabilization
- **Coupled joints** (walker knee, patella): Computed from polynomial constraints

## File Locations

```
/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/
├── inverse_dynamics_forces.png    # Key joints (8 subplots)
└── all_joint_forces.png           # All DOFs (single plot)
```

## Visualization Features

### Plot 1 (Key Joints):
- **Resolution**: 300 DPI (publication quality)
- **Size**: 14" x 12"
- **Format**: PNG with tight bounding box
- **Style**: Professional with bold titles

### Plot 2 (All DOFs):
- **Resolution**: 300 DPI
- **Size**: 16" x 8" (wide format)
- **Legend**: External (right side) with all 39 joint names
- **Alpha**: 0.7 transparency for overlapping lines

## Code Features

```python
# Forces are stored in joint_forces_over_time list
# Converted to numpy array: (208 timesteps, 39 DOFs)
joint_forces_array = np.array([np.array(f) for f in joint_forces_over_time])

# Visualization automatically:
1. Extracts time array from patient data
2. Plots key joints in separate subplots
3. Creates comprehensive overlay plot
4. Saves both as high-resolution PNG files
5. Displays plots (non-blocking)
```

## Using the Visualizations

### Analysis Workflow:
1. **Run the script**: `python MJX_RunID.py`
2. **Check console**: "Saved plot: ..." messages confirm creation
3. **View images**: Open PNG files in image viewer
4. **Analyze patterns**: 
   - Look for periodicity (gait cycle ~1.03s)
   - Check force magnitudes (reasonable for human motion?)
   - Compare left/right symmetry
   - Identify peak forces during stance/swing

### Quality Checks:
- ✅ Forces should be finite (no NaN or Inf)
- ✅ Magnitudes should be reasonable (0-200 N·m typical for legs)
- ✅ Patterns should repeat (periodic gait)
- ✅ Left/right should be symmetric but phase-shifted

## Customization

To modify the visualization, edit `MJX_RunID.py`:

### Change plotted joints:
```python
key_joints = [
    (6, 'Hip Flexion R'),
    (7, 'Hip Adduction R'),
    # Add or remove joints here
]
```

### Adjust plot style:
```python
ax.plot(time_array, forces, linewidth=2, color='blue')  # Change color/width
fig.suptitle('Your Title', fontsize=16)  # Change title
plt.savefig('custom_name.png', dpi=300)  # Change filename/resolution
```

### Add more analysis:
```python
# After computing forces, you can:
max_force = np.max(np.abs(joint_forces_array), axis=0)
print(f"Max forces per joint: {max_force}")

# Or compute RMS
rms_force = np.sqrt(np.mean(joint_forces_array**2, axis=0))
print(f"RMS forces per joint: {rms_force}")
```

## Technical Details

### Inverse Dynamics Method:
- Uses MuJoCo's `mjx.inverse()` function
- Computes generalized forces from:
  - `qpos`: Joint positions
  - `qvel`: Joint velocities
  - `qacc`: Joint accelerations
  - `xfrc_applied`: External forces (currently zeros)

### Output:
- `qfrc_inverse`: Generalized forces for inverse dynamics
- These are the control forces needed to achieve the desired accelerations

### Timing:
- Computation: ~9 seconds for 208 frames (21.9 fps)
- Plotting: < 1 second
- Total: ~10 seconds for complete analysis + visualization

## Next Steps

### Validation:
1. Compare with patient's measured forces (if available)
2. Check against biomechanical literature values
3. Verify gait cycle patterns match expectations

### Further Analysis:
1. Calculate joint power (force × velocity)
2. Integrate for joint work
3. Compare stance vs swing phase
4. Analyze bilateral symmetry quantitatively

### Export Data:
```python
# Save forces to CSV for external analysis
df_forces = pd.DataFrame(joint_forces_array, columns=joint_names)
df_forces['time'] = time_array
df_forces.to_csv('inverse_dynamics_forces.csv', index=False)
```

## Troubleshooting

### No plots appear:
- Plots are saved regardless of display
- Check for PNG files in working directory
- Use `plt.show(block=True)` to force display

### Forces look wrong:
- Verify input data (pos, vel, acc) is correct
- Check coupled joint calculations
- Ensure polynomial constraints are evaluated
- Validate external forces (should be zeros for now)

### Memory issues:
- 208 frames × 39 DOFs = 8,112 values (minimal memory)
- If processing longer trials, consider downsampling
- JAX arrays are converted to NumPy for plotting

## Summary

Your script now provides:
1. ✅ Automatic inverse dynamics calculation
2. ✅ Frame-by-frame processing (no batching)
3. ✅ Comprehensive force visualization (2 plots)
4. ✅ High-resolution publication-quality images
5. ✅ Progress tracking with tqdm
6. ✅ Error handling and fallback

The control forces are visualized and saved as PNG files, ready for analysis, presentation, or publication!
