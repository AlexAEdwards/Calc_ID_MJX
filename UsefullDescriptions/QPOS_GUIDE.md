# Guide: Setting Generalized Coordinates (qpos) in MuJoCo

## Overview

Your model has **39 generalized coordinates** (qpos values) that define the configuration of the entire musculoskeletal system. This guide explains how to set these values to visualize different kinematics.

## Complete qpos Structure

```python
data.qpos[0-38]  # 39 total coordinates
```

## Detailed Breakdown

### 1. Pelvis (Floating Base) - Indices 0-5

The pelvis is a **6-DOF floating base** that defines the overall position and orientation:

```python
data.qpos[0] = x_position     # Forward(+)/Backward(-) in meters
data.qpos[1] = y_position     # Up(+)/Down(-) in meters (height)
data.qpos[2] = z_position     # Right(+)/Left(-) in meters  
data.qpos[3] = pelvis_tilt    # Forward(+)/Backward(-) rotation (radians)
data.qpos[4] = pelvis_list    # Right(+)/Left(-) side bend (radians)
data.qpos[5] = pelvis_rotation # Twist right(+)/left(-) (radians)
```

**Default standing:** `[0, 0.93, 0, 0, 0, 0]`

### 2. Right Leg - Indices 6-20

#### Right Hip (3 DOF) - Indices 6-8
```python
data.qpos[6]  = hip_flexion_r    # Flexion(+)/Extension(-) 
                                  # Range: [-0.524, 2.094] rad (-30° to 120°)
data.qpos[7]  = hip_adduction_r  # Adduction(+)/Abduction(-)
                                  # Range: [-0.873, 0.524] rad (-50° to 30°)
data.qpos[8]  = hip_rotation_r   # Internal(+)/External(-) rotation
                                  # Range: [-0.698, 0.698] rad (-40° to 40°)
```

#### Right Knee (5 DOF Complex Joint) - Indices 9-13
```python
data.qpos[9]  = walker_knee_r_translation1  # Coupled translation (usually auto)
data.qpos[10] = walker_knee_r_translation2  # Coupled translation (usually auto)
data.qpos[11] = knee_angle_r                # Main knee flexion angle
                                             # Range: [0, 2.094] rad (0° to 120°)
data.qpos[12] = walker_knee_r_rotation2     # Coupled rotation (usually auto)
data.qpos[13] = walker_knee_r_rotation3     # Coupled rotation (usually auto)
```

**Note:** For the knee, typically you only set `data.qpos[11]` (the main knee angle). The other DOFs are coupled and update automatically based on the knee angle.

#### Right Ankle & Foot - Indices 14-16
```python
data.qpos[14] = ankle_angle_r     # Dorsiflexion(+)/Plantarflexion(-)
                                   # Range: [-0.698, 0.524] rad (-40° to 30°)
data.qpos[15] = subtalar_angle_r  # Inversion(+)/Eversion(-)
                                   # Range: [-0.349, 0.349] rad (-20° to 20°)
data.qpos[16] = mtp_angle_r       # Toe extension(+)/flexion(-)
                                   # Range: [-0.524, 0.524] rad (-30° to 30°)
```

#### Right Patella (Coupled) - Indices 17-20
```python
# These are automatically coupled to knee angle - usually don't set manually
data.qpos[17] = patellofemoral_r_translation1
data.qpos[18] = patellofemoral_r_translation2
data.qpos[19] = patellofemoral_r_translation3
data.qpos[20] = patellofemoral_r_rotation1
```

### 3. Left Leg - Indices 21-35

Same structure as right leg, just with different indices:

```python
# Left Hip
data.qpos[21] = hip_flexion_l      # Range: [-0.524, 2.094]
data.qpos[22] = hip_adduction_l    # Range: [-0.873, 0.524]
data.qpos[23] = hip_rotation_l     # Range: [-0.698, 0.698]

# Left Knee
data.qpos[24] = walker_knee_l_translation1
data.qpos[25] = walker_knee_l_translation2
data.qpos[26] = knee_angle_l       # Main knee angle [0, 2.094]
data.qpos[27] = walker_knee_l_rotation2
data.qpos[28] = walker_knee_l_rotation3

# Left Ankle & Foot
data.qpos[29] = ankle_angle_l      # Range: [-0.698, 0.524]
data.qpos[30] = subtalar_angle_l   # Range: [-0.349, 0.349]
data.qpos[31] = mtp_angle_l        # Range: [-0.524, 0.524]

# Left Patella (coupled)
data.qpos[32] = patellofemoral_l_translation1
data.qpos[33] = patellofemoral_l_translation2
data.qpos[34] = patellofemoral_l_translation3
data.qpos[35] = patellofemoral_l_rotation1
```

### 4. Torso/Lumbar - Indices 36-38

```python
data.qpos[36] = lumbar_extension  # Forward(+)/Backward(-) bend
                                   # Range: [-1.571, 1.571] rad (-90° to 90°)
data.qpos[37] = lumbar_bending    # Right(+)/Left(-) side bend
                                   # Range: [-1.571, 1.571] rad (-90° to 90°)
data.qpos[38] = lumbar_rotation   # Twist right(+)/left(-)
                                   # Range: [-1.571, 1.571] rad (-90° to 90°)
```

## Example Poses

### Standing (Default)
```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('Results/scaled_model_cvt2.xml')
data = mujoco.MjData(model)

# Reset to default
mujoco.mj_resetDataKeyframe(model, data, 0)
# This sets: pelvis height = 0.93m, all joints = 0

# Update kinematics
mujoco.mj_forward(model, data)
```

### Right Leg Forward (Hip Flexion 90°)
```python
# Start from default
mujoco.mj_resetDataKeyframe(model, data, 0)

# Set right hip flexion to 90 degrees (1.571 radians)
data.qpos[6] = 1.571

# Update
mujoco.mj_forward(model, data)
```

### Squat Position
```python
mujoco.mj_resetDataKeyframe(model, data, 0)

# Lower pelvis
data.qpos[1] = 0.70  # Lower height

# Flex both hips (~70°)
data.qpos[6] = 1.22   # Right hip
data.qpos[21] = 1.22  # Left hip

# Flex both knees (~80°)
data.qpos[11] = 1.40  # Right knee
data.qpos[26] = 1.40  # Left knee

# Dorsiflex ankles (~15°)
data.qpos[14] = 0.26  # Right ankle
data.qpos[29] = 0.26  # Left ankle

mujoco.mj_forward(model, data)
```

### Walking Stance (Mid-Stride)
```python
mujoco.mj_resetDataKeyframe(model, data, 0)

# Right leg forward (trailing)
data.qpos[6] = -0.17   # Hip slightly extended
data.qpos[11] = 0.26   # Knee slightly flexed
data.qpos[14] = -0.17  # Ankle plantarflexed

# Left leg forward (leading)
data.qpos[21] = 0.52   # Hip flexed 30°
data.qpos[26] = 0.17   # Knee slightly flexed
data.qpos[29] = 0.17   # Ankle dorsiflexed

mujoco.mj_forward(model, data)
```

### Seated Position
```python
mujoco.mj_resetDataKeyframe(model, data, 0)

# Lower and tilt pelvis
data.qpos[1] = 0.50   # Lower height
data.qpos[3] = 0.52   # Tilt forward ~30°

# Flex both hips to 90°
data.qpos[6] = 1.57   # Right hip
data.qpos[21] = 1.57  # Left hip

# Flex both knees to 90°
data.qpos[11] = 1.57  # Right knee
data.qpos[26] = 1.57  # Left knee

mujoco.mj_forward(model, data)
```

## Angle Conversion Reference

| Degrees | Radians | Description |
|---------|---------|-------------|
| 0°      | 0       | Neutral |
| 15°     | 0.262   | Small |
| 30°     | 0.524   | Moderate |
| 45°     | 0.785   | Medium |
| 60°     | 1.047   | Large |
| 90°     | 1.571   | Right angle |
| 120°    | 2.094   | Very large |
| -30°    | -0.524  | Extension/abduction |

**Quick conversion:**
- Degrees to radians: `deg * π / 180` or `deg * 0.01745`
- Radians to degrees: `rad * 180 / π` or `rad * 57.296`

## Important Notes

1. **Always call `mujoco.mj_forward(model, data)` after setting qpos values** to update the kinematics.

2. **Respect joint limits** - Each joint has min/max ranges. Setting values outside these ranges may cause unexpected behavior.

3. **Coupled joints** (knee translations/rotations, patella) are typically computed automatically. You usually only need to set the main joint angles.

4. **Units:**
   - Positions: meters
   - Angles: radians
   - All values in qpos are in world/model coordinates

5. **Sign conventions:**
   - Hip flexion: positive = leg forward
   - Knee flexion: positive = knee bending
   - Ankle dorsiflexion: positive = toes up

## Viewing Your Poses

Use the interactive viewer script:
```bash
conda run -n myoconverter python view_kinematics.py
```

Press number keys 0-9 to load preset poses, or use the UI sliders to adjust joints manually.

## Programmatic Animation

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('Results/scaled_model_cvt2.xml')
data = mujoco.MjData(model)

# Animate knee flexion
for i in range(100):
    # Oscillate knee between 0 and 90 degrees
    angle = 1.571 * np.sin(2 * np.pi * i / 100)
    
    data.qpos[11] = max(0, angle)  # Right knee (non-negative)
    data.qpos[26] = max(0, angle)  # Left knee
    
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    
    # Render or save frame here
```

## Full qpos Index Quick Reference

```
0-5:   Pelvis (6 DOF: 3 translations + 3 rotations)
6-8:   Right hip (3 DOF)
9-13:  Right knee (5 DOF, mostly coupled)
14-16: Right ankle/foot (3 DOF)
17-20: Right patella (4 DOF, fully coupled)
21-23: Left hip (3 DOF)
24-28: Left knee (5 DOF, mostly coupled)
29-31: Left ankle/foot (3 DOF)
32-35: Left patella (4 DOF, fully coupled)
36-38: Lumbar spine (3 DOF)
```

## Tips for Realistic Kinematics

1. **Keep the pelvis height reasonable** (0.7-1.0m for standing/walking)
2. **Coordinate hip and knee angles** (knee usually flexes when hip flexes)
3. **Balance left and right** (for symmetric poses)
4. **Check ground contact** (feet should touch ground at y ≈ 0)
5. **Use small increments** when testing (easier to debug)

## Resources

- `qpos_reference.txt` - Complete listing of all indices and ranges
- `view_kinematics.py` - Interactive viewer with preset poses
- `analyze_qpos.py` - Script to analyze your model's qpos structure
