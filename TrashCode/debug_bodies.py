import mujoco
import numpy as np

# Load model
m = mujoco.MjModel.from_xml_path('Results/ModelNoMus/scaled_model_no_muscles_cvt2_FIXED.xml')

print('='*70)
print('BODY IDs:')
print('='*70)
print(f'calcn_r: {m.body("calcn_r").id}')
print(f'calcn_l: {m.body("calcn_l").id}')
print(f'pelvis: {m.body("pelvis").id}')
print(f'talus_r: {m.body("talus_r").id}')
print(f'talus_l: {m.body("talus_l").id}')

print('\n' + '='*70)
print('JOINT NAMES (first 20):')
print('='*70)
for i in range(min(20, m.njnt)):
    print(f'{i:2d}: {m.joint(i).name}')

print('\n' + '='*70)
print('CHECKING JOINT CONNECTIONS:')
print('='*70)
for i in range(min(20, m.njnt)):
    jnt_bodyid = m.jnt_bodyid[i]
    body_name = m.body(jnt_bodyid).name
    print(f'Joint {i:2d} ({m.joint(i).name:20s}) attached to body {jnt_bodyid:2d} ({body_name})')
