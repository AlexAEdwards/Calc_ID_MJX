# Knee Joint Polynomial Constraints

## Overview

The knee joint in the MuJoCo model has **coupled coordinates** that are computed as polynomial functions of the main knee angle. These polynomials approximate the complex biomechanics of the knee joint.

## Polynomial Format

Each coupled coordinate is calculated as:
```
q_coupled = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
```
where:
- `θ` = knee_angle (the main knee flexion angle)
- `c0, c1, c2, c3, c4` = polynomial coefficients

## Right Knee Constraints

### Walker Knee Right - Coupled to `knee_angle_r` (qpos[11])

**qpos[9] - walker_knee_r_translation1:**
```python
coeffs = [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07]
qpos[9] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[10] - walker_knee_r_translation2:**
```python
coeffs = [7.949e-11, 0.006076, -0.001298, -2.706e-06, 6.452e-07]
qpos[10] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[12] - walker_knee_r_rotation2:**
```python
coeffs = [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083]
qpos[12] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[13] - walker_knee_r_rotation3:**
```python
coeffs = [1.089e-08, 0.3695, -0.1695, 0.02516, 3.505e-07]
qpos[13] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

### Patellofemoral Right - Coupled to `knee_angle_r` (qpos[11])

**qpos[17] - patellofemoral_r_translation1:**
```python
coeffs = [0.05515, -0.0158, -0.03583, 0.01403, -0.000925]
qpos[17] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[18] - patellofemoral_r_translation2:**
```python
coeffs = [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621]
qpos[18] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[19] - patellofemoral_r_translation3:**
```python
# Locked coordinate
coeffs = [0.00284182, 0, 0, 0, 0]
qpos[19] = 0.00284182  # constant
```

**qpos[20] - patellofemoral_r_rotation1:**
```python
coeffs = [0.01051, 0.02476, -1.316, 0.7163, -0.1383]
qpos[20] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

## Left Knee Constraints

### Walker Knee Left - Coupled to `knee_angle_l` (qpos[26])

**qpos[24] - walker_knee_l_translation1:**
```python
coeffs = [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07]
qpos[24] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[25] - walker_knee_l_translation2:**
```python
coeffs = [-7.949e-11, -0.006076, 0.001298, 2.706e-06, -6.452e-07]
qpos[25] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[27] - walker_knee_l_rotation2:**
```python
coeffs = [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083]
qpos[27] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[28] - walker_knee_l_rotation3:**
```python
coeffs = [-1.089e-08, -0.3695, 0.1695, -0.02516, -3.505e-07]
qpos[28] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

### Patellofemoral Left - Coupled to `knee_angle_l` (qpos[26])

**qpos[32] - patellofemoral_l_translation1:**
```python
coeffs = [0.05515, -0.0158, -0.03583, 0.01403, -0.000925]
qpos[32] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[33] - patellofemoral_l_translation2:**
```python
coeffs = [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621]
qpos[33] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

**qpos[34] - patellofemoral_l_translation3:**
```python
# Locked coordinate  
coeffs = [-0.00284182, 0, 0, 0, 0]
qpos[34] = -0.00284182  # constant
```

**qpos[35] - patellofemoral_l_rotation1:**
```python
coeffs = [0.01051, 0.02476, -1.316, 0.7163, -0.1383]
qpos[35] = coeffs[0] + coeffs[1]*θ + coeffs[2]*θ² + coeffs[3]*θ³ + coeffs[4]*θ⁴
```

## Summary of Coupled Indices

### Right Knee (driven by qpos[11] = knee_angle_r):
- qpos[9]: walker_knee_r_translation1
- qpos[10]: walker_knee_r_translation2
- qpos[12]: walker_knee_r_rotation2
- qpos[13]: walker_knee_r_rotation3
- qpos[17]: patellofemoral_r_translation1
- qpos[18]: patellofemoral_r_translation2
- qpos[19]: patellofemoral_r_translation3 (constant)
- qpos[20]: patellofemoral_r_rotation1

### Left Knee (driven by qpos[26] = knee_angle_l):
- qpos[24]: walker_knee_l_translation1
- qpos[25]: walker_knee_l_translation2
- qpos[27]: walker_knee_l_rotation2
- qpos[28]: walker_knee_l_rotation3
- qpos[32]: patellofemoral_l_translation1
- qpos[33]: patellofemoral_l_translation2
- qpos[34]: patellofemoral_l_translation3 (constant)
- qpos[35]: patellofemoral_l_rotation1

## Usage in Code

See `calculate_knee_coupled_coords()` function in MJX_RunID.py
