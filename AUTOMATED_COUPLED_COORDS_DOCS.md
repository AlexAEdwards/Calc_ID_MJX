# Automated Coupled Coordinates Calculation

## Overview

The `calculate_coupled_coordinates_automated()` function automatically extracts polynomial constraints from a MuJoCo XML model file and populates coupled coordinates without requiring hardcoded polynomial coefficients.

## Function Signature

```python
def calculate_coupled_coordinates_automated(qpos_matrix, qvel_matrix, qacc_matrix, xml_path):
    """
    Automatically extract polynomial constraints from XML model and populate coupled coordinates.
    
    Parameters:
    -----------
    qpos_matrix : numpy array (num_timesteps, nq)
        Position matrix with main joint angles already filled
    qvel_matrix : numpy array (num_timesteps, nv)
        Velocity matrix with main joint velocities already filled
    qacc_matrix : numpy array (num_timesteps, nv)
        Acceleration matrix with main joint accelerations already filled
    xml_path : str
        Path to the MuJoCo XML model file
    
    Returns:
    --------
    tuple : (qpos_matrix, qvel_matrix, qacc_matrix)
        Updated matrices with all coupled coordinates populated
    """
```

## How It Works

### 1. XML Parsing
The function parses the `<equality>` section of the MuJoCo XML file to extract joint equality constraints defined as polynomials:

```xml
<equality>
  <joint joint1="walker_knee_r_translation1" 
         joint2="knee_angle_r" 
         polycoef="9.877e-08 0.00324 -0.00239 0.0005816 5.886e-07"/>
  ...
</equality>
```

### 2. Constraint Format
Each constraint defines a relationship between two joints:
- **joint1** (coupled joint): The dependent coordinate
- **joint2** (driver joint): The independent coordinate that drives joint1
- **polycoef**: Five coefficients [c0, c1, c2, c3, c4] for a 4th-order polynomial

The relationship is:
```
joint1 = c0 + c1*joint2 + c2*joint2² + c3*joint2³ + c4*joint2⁴
```

### 3. Derivative Calculations

#### Position
Directly from the polynomial:
```
q = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
```

#### Velocity
Using the chain rule:
```
dq/dt = dq/dθ * dθ/dt
      = (c1 + 2*c2*θ + 3*c3*θ² + 4*c4*θ³) * dθ/dt
```

#### Acceleration
Using the chain rule:
```
d²q/dt² = d²q/dθ² * (dθ/dt)² + dq/dθ * d²θ/dt²
        = (2*c2 + 6*c3*θ + 12*c4*θ²) * (dθ/dt)² + (c1 + 2*c2*θ + 3*c3*θ² + 4*c4*θ³) * d²θ/dt²
```

### 4. Special Cases

#### Locked Joints
Joints with no driver (joint2=None) or all coefficients except c0 equal to zero are treated as locked:
```xml
<joint name="mtp_angle_r_locked" 
       joint1="mtp_angle_r" 
       polycoef="0 0 0 0 0"/>
```
These are set to constant position (c0) with zero velocity and acceleration.

## Usage Example

### Basic Usage

```python
from MJX_IDworks import calculate_coupled_coordinates_automated
import numpy as np
import mujoco

# Load model to get dimensions
xml_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

# Prepare matrices with main joint states
num_timesteps = 1000
qpos_matrix = np.zeros((num_timesteps, model.nq))
qvel_matrix = np.zeros((num_timesteps, model.nv))
qacc_matrix = np.zeros((num_timesteps, model.nv))

# Fill in your main joint angles (e.g., from motion capture)
# For example, knee angles, hip angles, etc.
qpos_matrix[:, 11] = your_knee_angle_r_data  # Right knee
qpos_matrix[:, 26] = your_knee_angle_l_data  # Left knee
# ... fill other main joints ...

# Calculate all coupled coordinates automatically
qpos_updated, qvel_updated, qacc_updated = calculate_coupled_coordinates_automated(
    qpos_matrix, 
    qvel_matrix, 
    qacc_matrix, 
    xml_path
)

# Now use the updated matrices for inverse dynamics
```

### Integration with Inverse Dynamics

```python
# After populating coupled coordinates
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

# Set the state at each timestep and compute inverse dynamics
for t in range(num_timesteps):
    mjx_data = mjx_data.replace(
        qpos=qpos_updated[t],
        qvel=qvel_updated[t],
        qacc=qacc_updated[t]
    )
    mjx_data = mjx.inverse(mjx_model, mjx_data)
    
    # Extract joint torques
    torques[t] = mjx_data.qfrc_inverse
```

## Comparison with Hardcoded Version

### Old Approach: `calculate_knee_coupled_coords_all()`
```python
# Hardcoded polynomial coefficients
right_knee_polys = {
    9: [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07],
    10: [7.949e-11, 0.006076, -0.001298, -2.706e-06, 6.452e-07],
    # ... more hardcoded values ...
}
```

**Limitations:**
- ❌ Must manually update coefficients if model changes
- ❌ Requires knowing joint indices in advance
- ❌ Error-prone (typos, incorrect indices)
- ❌ Not portable across different models

### New Approach: `calculate_coupled_coordinates_automated()`
```python
# Automatically reads from XML
qpos, qvel, qacc = calculate_coupled_coordinates_automated(
    qpos_matrix, qvel_matrix, qacc_matrix, xml_path
)
```

**Advantages:**
- ✅ Automatically extracts coefficients from XML
- ✅ Discovers joint indices dynamically
- ✅ Works with any model that follows the XML format
- ✅ Updates automatically if XML model changes
- ✅ Less error-prone
- ✅ More maintainable

## Output

The function prints detailed information about the extraction process:

```
======================================================================
Automated Coupled Coordinate Calculation
======================================================================
Found 18 polynomial constraints
Driver joints: ['knee_angle_r', 'knee_angle_l']

Processing driver joint: knee_angle_r (qpos_idx=11, qvel_idx=11)
  Coupled joints: 9
    ✓ walker_knee_r_translation1 (qpos_idx=9, qvel_idx=9)
    ✓ walker_knee_r_translation2 (qpos_idx=10, qvel_idx=10)
    ✓ walker_knee_r_rotation2 (qpos_idx=12, qvel_idx=12)
    ...

Processing driver joint: knee_angle_l (qpos_idx=26, qvel_idx=26)
  Coupled joints: 9
    ✓ walker_knee_l_translation1 (qpos_idx=24, qvel_idx=24)
    ...

======================================================================
✓ Coupled coordinates calculated successfully
======================================================================
```

## Model Requirements

The XML model must have:
1. An `<equality>` section with joint polynomial constraints
2. Each constraint must have:
   - `joint1` attribute (coupled joint name)
   - `joint2` attribute (driver joint name, or omitted for locked joints)
   - `polycoef` attribute with 5 space-separated coefficients

Example XML structure:
```xml
<mujoco>
  ...
  <equality>
    <joint joint1="coupled_joint" 
           joint2="driver_joint" 
           polycoef="c0 c1 c2 c3 c4"/>
  </equality>
  ...
</mujoco>
```

## Error Handling

The function handles several edge cases:
- Missing `<equality>` section → Returns unchanged matrices with warning
- Unknown joint names → Skips that constraint with warning
- Locked joints (no driver) → Sets to constant value
- Non-5-coefficient polynomials → Skips that constraint

## Performance Considerations

- **Time Complexity**: O(n * m) where n = timesteps, m = number of constraints
- **Memory**: Creates copies of input matrices (safe, no side effects)
- **XML Parsing**: Only done once at the start, cached by ElementTree

## Tips for Use

1. **Always populate main joints first**: The function depends on driver joint values being set before calling it
2. **Check the output**: Review the printed joint mapping to ensure correct extraction
3. **Verify results**: For the first run, compare with hardcoded version if available
4. **Update XML**: If you modify polynomial constraints in the model, the function automatically picks up the changes

## Troubleshooting

### Problem: "No equality section found in XML"
**Solution**: Verify your XML file has an `<equality>` section with joint constraints

### Problem: "Driver joint 'xyz' not found in model"
**Solution**: Check that the joint2 name in the XML matches an actual joint in the model

### Problem: Incorrect values calculated
**Solution**: 
1. Verify main joint states are correctly populated before calling the function
2. Check that polynomial coefficients in XML are correct
3. Ensure joint indices match between qpos and qvel matrices

## See Also

- `calculate_knee_coupled_coords_all()` - Original hardcoded version (still available)
- `example_automated_coupled_coords.py` - Complete working example
- MuJoCo documentation on equality constraints: https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality
