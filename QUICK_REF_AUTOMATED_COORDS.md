# Quick Reference: calculate_coupled_coordinates_automated()

## Import
```python
from MJX_IDworks import calculate_coupled_coordinates_automated
```

## Basic Usage
```python
qpos, qvel, qacc = calculate_coupled_coordinates_automated(
    qpos_matrix,    # numpy array (timesteps, nq) - positions
    qvel_matrix,    # numpy array (timesteps, nv) - velocities  
    qacc_matrix,    # numpy array (timesteps, nv) - accelerations
    xml_path        # str - path to XML model file
)
```

## What It Does
✅ Reads polynomial constraints from XML `<equality>` section  
✅ Automatically finds joint name→index mappings  
✅ Calculates coupled coordinates using chain rule  
✅ Populates position, velocity, and acceleration matrices  
✅ Handles locked joints and special cases  

## Requirements
- Main joint angles already populated in qpos_matrix
- Main joint velocities already populated in qvel_matrix  
- Main joint accelerations already populated in qacc_matrix
- Valid XML model with `<equality>` section

## Example Workflow
```python
import numpy as np
import mujoco
from MJX_IDworks import calculate_coupled_coordinates_automated

# 1. Load model
xml_path = "model.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

# 2. Create matrices
num_timesteps = 1000
qpos = np.zeros((num_timesteps, model.nq))
qvel = np.zeros((num_timesteps, model.nv))
qacc = np.zeros((num_timesteps, model.nv))

# 3. Fill main joints (from motion capture, etc.)
qpos[:, 11] = knee_angle_r_data  # Right knee
qpos[:, 26] = knee_angle_l_data  # Left knee
qvel[:, 11] = knee_vel_r_data
qvel[:, 26] = knee_vel_l_data
qacc[:, 11] = knee_acc_r_data
qacc[:, 26] = knee_acc_l_data

# 4. Calculate coupled coordinates
qpos, qvel, qacc = calculate_coupled_coordinates_automated(
    qpos, qvel, qacc, xml_path
)

# 5. Use in inverse dynamics
# Now qpos/qvel/qacc are fully populated!
```

## Replace Old Hardcoded Version
```python
# OLD (hardcoded)
qpos, qvel, qacc = calculate_knee_coupled_coords_all(qpos, qvel, qacc)

# NEW (automated)
qpos, qvel, qacc = calculate_coupled_coordinates_automated(
    qpos, qvel, qacc, xml_path
)
```

## XML Format
Your model needs:
```xml
<equality>
  <joint joint1="coupled_joint" 
         joint2="driver_joint" 
         polycoef="c0 c1 c2 c3 c4"/>
</equality>
```

## Benefits vs Hardcoded
| Feature | Old | New |
|---------|-----|-----|
| **Source** | Manual | XML |
| **Updates** | Re-code | Automatic |
| **Models** | One | Any |
| **Errors** | Common | Rare |
| **Portable** | No | Yes |

## Verification
```bash
python verify_automated_vs_hardcoded.py
```

## Troubleshooting
| Error | Solution |
|-------|----------|
| "No equality section" | Add `<equality>` to XML |
| "Driver joint not found" | Check joint names in XML |
| Wrong values | Verify main joints populated first |

## Files
- **Function:** `MJX_IDworks.py` (line ~375)
- **Example:** `example_automated_coupled_coords.py`
- **Verify:** `verify_automated_vs_hardcoded.py`
- **Docs:** `AUTOMATED_COUPLED_COORDS_DOCS.md`

## Mathematical Details

**Position:**
```
q = c₀ + c₁θ + c₂θ² + c₃θ³ + c₄θ⁴
```

**Velocity (chain rule):**
```
dq/dt = (c₁ + 2c₂θ + 3c₃θ² + 4c₄θ³) · dθ/dt
```

**Acceleration (chain rule):**
```
d²q/dt² = (2c₂ + 6c₃θ + 12c₄θ²) · (dθ/dt)² 
          + (c₁ + 2c₂θ + 3c₃θ² + 4c₄θ³) · d²θ/dt²
```

## Output Example
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
    ...

======================================================================
✓ Coupled coordinates calculated successfully
======================================================================
```

## One-Line Summary
**Extracts polynomial constraints from XML and automatically calculates all coupled joint coordinates with their derivatives.**
