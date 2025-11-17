# MuJoCo Model Structure Overview

This document explains the structure of your converted MuJoCo model from the OpenSim Rajagopal model.

## File Organization

Your Results folder contains:

```
Results/
├── scaled_model_cvt1.xml          # Step 1: Basic geometric conversion
├── scaled_model_cvt2.xml          # Step 2: With optimized muscle kinematics
├── scaled_model_conversion.log    # Conversion log file
├── Geometry/                      # Converted STL mesh files
│   └── *.stl
├── Step1_xmlConvert/              # Step 1 conversion artifacts
│   ├── jnt_ranges_mjc.pkl         # MuJoCo joint ranges
│   ├── jnt_ranges_osim.pkl        # OpenSim joint ranges
│   ├── custom_joints/             # Visualization of custom joint transforms
│   │   └── *.svg
│   └── end_points/                # Marker endpoint visualizations
│       └── *.svg
└── Step2_muscleKinematics/        # Step 2 muscle optimization data
    ├── config.pkl                 # Configuration file
    └── *.pkl                      # One file per muscle with kinematic data
```

## Conversion Steps

### Step 1: Geometric Conversion (`scaled_model_cvt1.xml`)
This file contains the basic skeletal structure without muscle optimization:
- **Bodies**: All skeletal segments with their meshes and inertial properties
- **Joints**: All joints with their DOFs and ranges of motion
- **Geometry**: Mesh files and wrapping objects
- **Constraints**: Kinematic constraints (e.g., patellofemoral coupling)

### Step 2: Muscle Kinematics (`scaled_model_cvt2.xml`)
This is your **final, optimized model** with:
- Optimized muscle moment arms and lengths
- Muscle actuators with realistic force-length-velocity properties
- Tendons representing muscle paths
- Fine-tuned muscle parameters

## MuJoCo XML Structure

### 1. Header Section
```xml
<mujoco model="template">
  <compiler>      <!-- Compilation settings -->
  <option>        <!-- Simulation options -->
  <visual>        <!-- Visualization settings -->
  <default>       <!-- Default properties for elements -->
  <size>          <!-- Model size parameters -->
```

**Key Settings:**
- `timestep="0.001"`: Simulation timestep (1ms)
- `angle="radian"`: All angles in radians
- `autolimits="true"`: Automatically compute actuator limits

### 2. Asset Section
```xml
<asset>
  <texture>     <!-- Skybox/textures -->
  <mesh>        <!-- 3D mesh files -->
```

**Mesh References:**
- Each body segment has associated STL mesh files
- Meshes include scaling factors from your subject-specific model
- Example: `<mesh name="femur_r_geom_1_r_femur" file="Geometry/r_femur.stl" scale="1.0521 1.03563 1.03339"/>`

### 3. Worldbody Section
```xml
<worldbody>
  <light>       <!-- Lighting -->
  <body>        <!-- Ground plane -->
    <body>      <!-- Pelvis (root body) -->
      <body>    <!-- Femur -->
        <body>  <!-- Tibia -->
          ...   <!-- Hierarchical body tree -->
```

**Body Hierarchy:**
- **Pelvis** (floating base with 6 DOF)
  - **Femur_r/l** (hip joint)
    - **Tibia_r/l** (knee joint)
      - **Talus_r/l** (ankle joint)
        - **Calcn_r/l** (subtalar joint)
          - **Toes_r/l** (MTP joint)
      - **Patella_r/l** (patellofemoral joint)
  - **Torso** (lumbar joint)

**Each Body Contains:**
```xml
<inertial>    <!-- Mass and inertia tensor -->
<geom>        <!-- Visual and collision geometry -->
<site>        <!-- Attachment points for muscles/sensors -->
<joint>       <!-- Degrees of freedom -->
```

### 4. Tendon Section
Tendons represent muscle paths:
```xml
<tendon>
  <spatial name="muscle_name_tendon">
    <site site="origin_site"/>
    <geom geom="wrapping_cylinder"/>  <!-- Optional: wrapping surfaces -->
    <site site="insertion_site"/>
  </spatial>
```

**Muscle Path Types:**
- **Direct paths**: Straight line from origin to insertion
- **Via points**: Path goes through intermediate points
- **Wrapping surfaces**: Path wraps around cylinders (e.g., around bones)

### 5. Actuator Section
Muscles are implemented as actuators:
```xml
<actuator>
  <general name="muscle_name" class="muscle" 
           tendon="muscle_name_tendon"
           lengthrange="0.01 1"
           gainprm="0.75 1.05 [MAX_FORCE] 200 0.5 1.6 1.5 1.3 1.2 0"
           biasprm="0.75 1.05 [MAX_FORCE] 200 0.5 1.6 1.5 1.3 1.2 0"/>
```

**Muscle Parameters:**
- `dyntype="muscle"`: Uses Hill-type muscle model
- `gainprm[2]`: Maximum isometric force (varies per muscle)
- `lengthrange`: Normalized muscle length range
- Force-length-velocity curves are encoded in the gain/bias parameters

### 6. Keyframe Section
Default pose of the model:
```xml
<keyframe>
  <key name="default-pose" qpos="..."/>
```

## Your Model Contains

### Joints (Degrees of Freedom)
- **Pelvis**: 6 DOF (3 translations + 3 rotations)
- **Hip_r/l**: 3 DOF (flexion, adduction, rotation)
- **Knee_r/l**: 5 DOF (custom joint with coupled translations/rotations)
- **Ankle_r/l**: 1 DOF (dorsi/plantar flexion)
- **Subtalar_r/l**: 1 DOF (inversion/eversion)
- **MTP_r/l**: 1 DOF (toe flexion)
- **Lumbar**: 3 DOF (extension, bending, rotation)

### Muscles (80 total - 40 per leg)
**Hip muscles:**
- Gluteus maximus (3 parts)
- Gluteus medius (3 parts)
- Gluteus minimus (3 parts)
- Iliopsoas (iliacus + psoas)
- Hip adductors (adductor brevis, longus, magnus)
- Deep rotators (piriformis, etc.)

**Thigh muscles:**
- Quadriceps (rectus femoris, vastus lateralis/medialis/intermedius)
- Hamstrings (biceps femoris, semimembranosus, semitendinosus)
- Sartorius, tensor fasciae latae, gracilis

**Shank muscles:**
- Gastrocnemius (medial/lateral heads)
- Soleus
- Tibialis anterior/posterior
- Peroneals (brevis/longus)
- Toe flexors/extensors (FDL, FHL, EDL, EHL)

### Geometry Features
- **Mesh files**: High-quality 3D meshes for each bone
- **Wrapping objects**: Cylindrical surfaces that muscles wrap around
- **Sites**: Muscle attachment points and landmarks
- **Ground plane**: For contact detection

## Key Differences Between cvt1 and cvt2

### `scaled_model_cvt1.xml`
- Direct conversion from OpenSim
- Simple muscle paths (may have inaccuracies)
- No muscle optimization
- Used as a baseline for Step 2

### `scaled_model_cvt2.xml` (FINAL MODEL)
- **Optimized muscle moment arms**: Matched to OpenSim through optimization
- **Correct muscle-tendon lengths**: Validated across joint ranges
- **Refined muscle paths**: Better wrapping and via points
- **Ready for simulation**: Can be used directly in MuJoCo/MJX

## Muscle Data Files (Step2_muscleKinematics/)

Each `.pkl` file contains optimized data for one muscle:
- Muscle-tendon length as a function of joint angles
- Moment arms for each DOF the muscle crosses
- Optimal fiber length and tendon slack length
- Maximum isometric force

Example: `recfem_r.pkl` contains data for the right rectus femoris muscle.

## Validation Visualizations

### Custom Joints SVG Files
Show the coordinate transforms for complex joints (e.g., knee):
- `walker_knee_r_rotation*.svg`: Rotation coupling functions
- `walker_knee_r_translation*.svg`: Translation coupling functions
- `patellofemoral_r_*.svg`: Patella tracking functions

### End Points SVG Files
Show marker positions throughout the motion range to validate kinematics.

## Using the Model

### In MuJoCo (Python)
```python
import mujoco

# Load the model
model = mujoco.MjModel.from_xml_path('Results/scaled_model_cvt2.xml')
data = mujoco.MjData(model)

# Run simulation
mujoco.mj_step(model, data)
```

### In MJX (JAX-based)
```python
import mujoco
from mujoco import mjx

# Load and convert to MJX
mj_model = mujoco.MjModel.from_xml_path('Results/scaled_model_cvt2.xml')
mjx_model = mjx.put_model(mj_model)
```

## Tips for Working with the Model

1. **Use cvt2 for simulations**: It has optimized muscle kinematics
2. **Check the log file**: `scaled_model_conversion.log` contains warnings/errors
3. **Muscle activation**: Control muscles with values in [0, 1]
4. **Joint limits**: All joints have physiologically realistic limits
5. **Contact**: Ground plane is set up for foot contact detection
6. **Timestep**: Start with 0.001s, can go smaller if needed

## Common Applications

- **Forward dynamics**: Apply muscle activations, simulate motion
- **Inverse dynamics**: Given motion, compute required muscle forces
- **Predictive simulation**: Optimize controls to achieve a task
- **Reinforcement learning**: Train policies for locomotion
- **Biomechanical analysis**: Study muscle forces, joint loads, etc.

## Next Steps

1. **Visualize the model**: Use MuJoCo's viewer
2. **Test simulation**: Run a simple forward simulation
3. **Validate kinematics**: Compare joint ranges to OpenSim
4. **Add sensors**: Add IMUs, force plates, etc. if needed
5. **Develop controllers**: Create muscle activation patterns

## References

- **MuJoCo Documentation**: https://mujoco.readthedocs.io/
- **MyoConverter GitHub**: https://github.com/MyoHub/myoconverter
- **Rajagopal et al. (2016)**: Original OpenSim model paper
- **MJX Documentation**: https://mujoco.readthedocs.io/en/stable/mjx.html
