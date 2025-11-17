"""
Analyze and document the generalized coordinates (qpos) structure
"""
import mujoco
import numpy as np

model_path = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml'

print("Loading model...")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("=" * 80)
print("GENERALIZED COORDINATES (qpos) STRUCTURE")
print("=" * 80)
print(f"Total qpos dimension: {model.nq}")
print(f"Total DOFs (nv):      {model.nv}")
print()

# Get joint information
print("=" * 80)
print("JOINT INDEX MAPPING")
print("=" * 80)
print(f"{'Index':<6} {'Joint Name':<35} {'Type':<10} {'qpos Range':<15} {'Limits'}")
print("-" * 80)

joint_types = {
    0: 'free',
    1: 'ball', 
    2: 'slide',
    3: 'hinge'
}

qpos_idx = 0
for jnt_id in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    jnt_type = model.jnt_type[jnt_id]
    jnt_type_name = joint_types.get(jnt_type, 'unknown')
    
    # Get qpos address
    qpos_adr = model.jnt_qposadr[jnt_id]
    
    # Get range
    jnt_range = model.jnt_range[jnt_id]
    range_str = f"[{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]"
    
    # Determine qpos size for this joint
    if jnt_type == 0:  # free joint
        qpos_size = 7  # 3 pos + 4 quat
        qpos_range = f"[{qpos_adr}:{qpos_adr+7}]"
    elif jnt_type == 1:  # ball joint
        qpos_size = 4  # quaternion
        qpos_range = f"[{qpos_adr}:{qpos_adr+4}]"
    else:  # slide or hinge
        qpos_size = 1
        qpos_range = f"[{qpos_adr}]"
    
    print(f"{qpos_adr:<6} {jnt_name:<35} {jnt_type_name:<10} {qpos_range:<15} {range_str}")

print()
print("=" * 80)
print("COORDINATE GROUPS BY BODY")
print("=" * 80)

# Reset to default
mujoco.mj_resetDataKeyframe(model, data, 0)

# Group joints by major body parts
joint_groups = {
    'Pelvis (Floating Base)': [],
    'Right Hip': [],
    'Right Knee': [],
    'Right Ankle/Foot': [],
    'Left Hip': [],
    'Left Knee': [],
    'Left Ankle/Foot': [],
    'Torso/Lumbar': [],
}

for jnt_id in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    qpos_adr = model.jnt_qposadr[jnt_id]
    
    if 'pelvis' in jnt_name.lower():
        joint_groups['Pelvis (Floating Base)'].append((jnt_id, jnt_name, qpos_adr))
    elif 'hip' in jnt_name.lower() and '_r' in jnt_name.lower():
        joint_groups['Right Hip'].append((jnt_id, jnt_name, qpos_adr))
    elif 'knee' in jnt_name.lower() and '_r' in jnt_name.lower():
        joint_groups['Right Knee'].append((jnt_id, jnt_name, qpos_adr))
    elif ('ankle' in jnt_name.lower() or 'subtalar' in jnt_name.lower() or 'mtp' in jnt_name.lower()) and '_r' in jnt_name.lower():
        joint_groups['Right Ankle/Foot'].append((jnt_id, jnt_name, qpos_adr))
    elif 'hip' in jnt_name.lower() and '_l' in jnt_name.lower():
        joint_groups['Left Hip'].append((jnt_id, jnt_name, qpos_adr))
    elif 'knee' in jnt_name.lower() and '_l' in jnt_name.lower():
        joint_groups['Left Knee'].append((jnt_id, jnt_name, qpos_adr))
    elif ('ankle' in jnt_name.lower() or 'subtalar' in jnt_name.lower() or 'mtp' in jnt_name.lower()) and '_l' in jnt_name.lower():
        joint_groups['Left Ankle/Foot'].append((jnt_id, jnt_name, qpos_adr))
    elif 'lumbar' in jnt_name.lower() or 'torso' in jnt_name.lower():
        joint_groups['Torso/Lumbar'].append((jnt_id, jnt_name, qpos_adr))

for group_name, joints in joint_groups.items():
    if joints:
        print(f"\n{group_name}:")
        for jnt_id, jnt_name, qpos_adr in joints:
            jnt_range = model.jnt_range[jnt_id]
            default_val = data.qpos[qpos_adr]
            print(f"  qpos[{qpos_adr:2d}] = {jnt_name:35s}  Range: [{jnt_range[0]:7.3f}, {jnt_range[1]:7.3f}]  Default: {default_val:7.4f}")

print()
print("=" * 80)
print("DEFAULT POSE (qpos values)")
print("=" * 80)
print("data.qpos =", np.array2string(data.qpos, precision=4, suppress_small=True, max_line_width=80))

print()
print("=" * 80)
print("QUICK REFERENCE GUIDE")
print("=" * 80)
print("""
To set joint positions in your viewer script:

1. PELVIS (Floating Base) - Indices 0-5:
   data.qpos[0] = x_position      # Forward/backward (meters)
   data.qpos[1] = y_position      # Up/down (meters, ~0.93 for standing)
   data.qpos[2] = z_position      # Left/right (meters)
   data.qpos[3] = tilt            # Forward/backward rotation (radians)
   data.qpos[4] = list            # Side bending (radians)
   data.qpos[5] = rotation        # Twisting (radians)

2. RIGHT LEG:
   data.qpos[6]  = hip_flexion_r     # Range: [-0.524, 2.094] rad (-30° to 120°)
   data.qpos[7]  = hip_adduction_r   # Range: [-0.873, 0.524] rad (-50° to 30°)
   data.qpos[8]  = hip_rotation_r    # Range: [-0.698, 0.698] rad (-40° to 40°)
   data.qpos[9-13] = knee complex joints (5 DOFs - usually leave default)
   data.qpos[14] = ankle_angle_r     # Range: [-0.698, 0.524] rad (-40° to 30°)
   data.qpos[15] = subtalar_angle_r  # Range: [-0.698, 0.698] rad (-40° to 40°)
   data.qpos[16] = mtp_angle_r       # Range: [-0.698, 0.873] rad (-40° to 50°)

3. LEFT LEG: (similar structure, different indices)

4. LUMBAR/TORSO: (if included)

IMPORTANT NOTES:
- All angles are in RADIANS (multiply degrees by π/180)
- After setting qpos, call: mujoco.mj_forward(model, data)
- Common conversions:
  * 90° = 1.571 rad
  * 45° = 0.785 rad  
  * 30° = 0.524 rad
  * -30° = -0.524 rad
""")

# Save this info to a file
output_file = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/qpos_reference.txt'
with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GENERALIZED COORDINATES (qpos) REFERENCE\n")
    f.write("=" * 80 + "\n\n")
    
    for jnt_id in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        qpos_adr = model.jnt_qposadr[jnt_id]
        jnt_range = model.jnt_range[jnt_id]
        default_val = data.qpos[qpos_adr]
        f.write(f"qpos[{qpos_adr:2d}] = {jnt_name:35s}  Range: [{jnt_range[0]:7.3f}, {jnt_range[1]:7.3f}]  Default: {default_val:7.4f}\n")

print(f"\n✓ Reference saved to: {output_file}")
