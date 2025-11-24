# updateModel.py - Function Documentation

## Overview

The `updateModel.py` module provides a function to fix and load MuJoCo models with MJX, handling common issues like zero masses, small inertias, and small armatures.

## Main Function: `update_model()`

### Usage

```python
from updateModel import update_model

mjx_model, mj_model, fixed_xml_path = update_model(
    xml_path="path/to/model.xml",
    min_mass=0.5,
    min_inertia=0.01,
    min_armature=0.1
)
```

### Parameters

- **`xml_path`** (str, required): Path to the input XML file
- **`min_mass`** (float, optional): Minimum body mass. Default: 0.5
- **`min_inertia`** (float, optional): Minimum body inertia. Default: 0.01
- **`min_armature`** (float, optional): Minimum joint armature. Default: 0.1

### Returns

A tuple containing:
1. **`mjx_model`**: MJX model ready for simulation (or None if failed)
2. **`mujoco_model`**: Standard MuJoCo model (or None if failed)
3. **`fixed_xml_path`**: Path to the fixed XML file (or None if failed)

### What It Does

The function performs the following steps automatically:

1. **Fixes XML file**:
   - Sets compiler bounds for mass and inertia
   - Updates default joint armature
   - Fixes individual joint armatures
   - Corrects body masses and inertias
   - Saves a new XML file with "_FIXED.xml" suffix

2. **Verifies fixes**:
   - Loads the fixed model
   - Checks that all masses, inertias, and armatures meet minimums
   - Reports any remaining issues

3. **Applies runtime fixes**:
   - Ensures armature values are correct
   - Applies additional mass fixes if needed
   - Sets optimal solver options

4. **Converts to MJX**:
   - Creates MJX model from MuJoCo model
   - Applies post-conversion fixes if needed
   - Ensures all parameters are JAX-compatible

5. **Tests the model**:
   - Verifies `mjx.inverse()` works
   - Verifies `mjx.step()` works
   - Reports success or failure

### Example Usage

#### Basic Usage with Defaults

```python
from updateModel import update_model

mjx_model, mj_model, xml_path = update_model("my_model.xml")

if mjx_model is not None:
    print("Model loaded successfully!")
    # Use mjx_model for simulation
```

#### Custom Parameters

```python
from updateModel import update_model

mjx_model, mj_model, xml_path = update_model(
    xml_path="Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml",
    min_mass=0.25,        # Lower minimum mass
    min_inertia=0.01,     # Keep default inertia
    min_armature=0.01     # Lower minimum armature
)

if mjx_model is not None:
    import mujoco.mjx as mjx
    
    # Create simulation data
    data = mjx.make_data(mjx_model)
    
    # Run simulation
    for _ in range(100):
        data = mjx.step(mjx_model, data)
    
    print(f"Simulation complete!")
```

#### Error Handling

```python
from updateModel import update_model

mjx_model, mj_model, xml_path = update_model("my_model.xml")

if mjx_model is None:
    print("Failed to load model. Check the error messages above.")
    print("Common issues:")
    print("  - Invalid XML file")
    print("  - Model has structural problems")
    print("  - Solver instability")
else:
    print(f"Success! Fixed model saved to: {xml_path}")
    print(f"Model has {mjx_model.nbody} bodies and {mjx_model.nv} DOFs")
```

### Output Files

The function creates a new XML file with the suffix `_FIXED.xml`:

- Input: `my_model.xml`
- Output: `my_model_FIXED.xml`

The fixed XML file contains all the corrections and can be used independently.

### Common Parameter Values

| Use Case | min_mass | min_inertia | min_armature |
|----------|----------|-------------|--------------|
| Default (safe) | 0.5 | 0.01 | 0.1 |
| Low-mass bodies | 0.25 | 0.01 | 0.01 |
| High-mass bodies | 1.0 | 0.05 | 0.5 |
| Stiff joints | 0.5 | 0.01 | 1.0 |

### Notes

- The function automatically configures JAX with `jax_enable_x64=True`
- Progress is printed to console at each step
- If the model fails tests, returns `(None, None, None)`
- The original XML file is never modified
- Fixed XML preserves all original model structure

### Troubleshooting

**Problem: Model returns None**
- Check the console output for specific error messages
- Try increasing `min_mass` and `min_armature` values
- Verify the XML file is valid and well-formed

**Problem: Solver instability**
- Increase `min_armature` to make joints stiffer
- Increase `min_mass` to avoid numerical issues
- Check for kinematic loops or redundant constraints

**Problem: ImportError**
- Ensure MuJoCo and JAX are installed: `pip install mujoco jax`
- For MJX support, use MuJoCo 3.0+

### See Also

- `example_import_updateModel.py` - Complete working example
- Helper functions in module:
  - `fix_xml_masses()` - XML modification only
  - `verify_fixes()` - Verification only
  - `fix_and_load_model()` - Complete workflow (called by update_model)
