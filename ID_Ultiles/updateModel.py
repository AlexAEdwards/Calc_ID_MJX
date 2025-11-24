import xml.etree.ElementTree as ET
import mujoco
import mujoco.mjx as mjx
import jax.numpy as jnp

def fix_xml_masses(xml_path, output_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Fix zero masses, small inertias, and small armatures directly in XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print(f"\n{'='*70}")
    print(f"FIXING XML: {xml_path}")
    print(f"{'='*70}")
    
    # Update compiler to enforce minimum mass and inertia
    compiler = root.find('.//compiler')
    if compiler is not None:
        compiler.set('boundmass', str(min_mass))
        compiler.set('boundinertia', str(min_inertia))
        print(f"✓ Updated compiler:")
        print(f"    boundmass={min_mass}")
        print(f"    boundinertia={min_inertia}")
    else:
        print("⚠️  No compiler element found")
    
    # Fix default joint armature
    print(f"\n1. Fixing default joint armature...")
    default_joint = root.find('.//default/joint')
    if default_joint is not None:
        current_armature = default_joint.get('armature')
        if current_armature is not None:
            current_val = float(current_armature)
            if current_val < min_armature:
                default_joint.set('armature', str(min_armature))
                print(f"   ✓ Updated default joint armature: {current_val:.2e} -> {min_armature:.2e}")
        else:
            default_joint.set('armature', str(min_armature))
            print(f"   ✓ Added default joint armature: {min_armature:.2e}")
    else:
        # Create default/joint element if it doesn't exist
        default_elem = root.find('.//default')
        if default_elem is None:
            default_elem = ET.SubElement(root, 'default')
        joint_elem = ET.SubElement(default_elem, 'joint')
        joint_elem.set('armature', str(min_armature))
        print(f"   ✓ Created default joint with armature: {min_armature:.2e}")
    
    # Fix individual joint armatures
    print(f"\n2. Fixing individual joint armatures...")
    joint_armature_fixed = 0
    for joint in root.findall('.//joint'):
        armature_str = joint.get('armature')
        if armature_str is not None:
            armature = float(armature_str)
            if armature < min_armature:
                joint.set('armature', str(min_armature))
                joint_name = joint.get('name', 'unnamed')
                print(f"   Fixed joint '{joint_name}': {armature:.2e} -> {min_armature:.2e}")
                joint_armature_fixed += 1
    
    if joint_armature_fixed > 0:
        print(f"   ✓ Fixed {joint_armature_fixed} individual joints")
    else:
        print(f"   No individual joint armatures needed fixing")
    
    # Fix body masses
    print(f"\n3. Fixing body masses...")
    mass_fixed = 0
    for inertial in root.findall('.//inertial'):
        mass_str = inertial.get('mass')
        if mass_str is not None:
            mass = float(mass_str)
            if mass < min_mass:
                parent_body = inertial.find('..')
                body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                inertial.set('mass', str(min_mass))
                print(f"   Fixed body '{body_name}': {mass:.6f} -> {min_mass:.6f}")
                mass_fixed += 1
    
    if mass_fixed > 0:
        print(f"   ✓ Fixed {mass_fixed} bodies with small/zero mass")
    else:
        print(f"   No body masses needed fixing")
    
    # Fix body inertias
    print(f"\n4. Fixing body inertias...")
    inertia_fixed = 0
    for inertial in root.findall('.//inertial'):
        # Check fullinertia attribute (6 values: Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
        fullinertia_str = inertial.get('fullinertia')
        if fullinertia_str is not None:
            values = [float(x) for x in fullinertia_str.split()]
            if len(values) >= 3:  # Check diagonal elements
                if any(v < min_inertia for v in values[:3]):
                    # Fix diagonal elements
                    values[:3] = [max(v, min_inertia) for v in values[:3]]
                    inertial.set('fullinertia', ' '.join(str(v) for v in values))
                    parent_body = inertial.find('..')
                    body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                    print(f"   Fixed inertia for body '{body_name}'")
                    inertia_fixed += 1
        
        # Check diaginertia attribute (3 values: Ixx, Iyy, Izz)
        diaginertia_str = inertial.get('diaginertia')
        if diaginertia_str is not None:
            values = [float(x) for x in diaginertia_str.split()]
            if any(v < min_inertia for v in values):
                values = [max(v, min_inertia) for v in values]
                inertial.set('diaginertia', ' '.join(str(v) for v in values))
                parent_body = inertial.find('..')
                body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                print(f"   Fixed inertia for body '{body_name}'")
                inertia_fixed += 1
    
    if inertia_fixed > 0:
        print(f"   ✓ Fixed {inertia_fixed} bodies with small inertias")
    else:
        print(f"   No body inertias needed fixing")
    
    # Save modified XML
    tree.write(output_path)
    print(f"\n{'='*70}")
    print(f"✓ SAVED FIXED MODEL TO: {output_path}")
    print(f"{'='*70}")
    
    return output_path


def verify_fixes(xml_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Verify that the fixes were applied correctly."""
    print(f"\n{'='*70}")
    print(f"VERIFYING FIXES")
    print(f"{'='*70}")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Check masses
    masses = model.body_mass[1:]  # Skip world body
    print(f"\nBody Masses:")
    print(f"  Min: {masses.min():.6f}")
    print(f"  Max: {masses.max():.6f}")
    
    if masses.min() < min_mass:
        print(f"  ⚠️  WARNING: Min mass {masses.min():.6f} < {min_mass}")
        zero_mass = np.where(masses < min_mass)[0] + 1
        print(f"     Problematic bodies: {zero_mass[:5]}")
    else:
        print(f"  ✓ All masses >= {min_mass}")
    
    # Check inertias
    inertias = model.body_inertia[1:]
    inertia_min = inertias[inertias > 0].min() if (inertias > 0).any() else 0
    print(f"\nBody Inertias:")
    print(f"  Min (non-zero): {inertia_min:.6e}")
    
    if inertia_min < min_inertia and inertia_min > 0:
        print(f"  ⚠️  WARNING: Min inertia {inertia_min:.6e} < {min_inertia}")
    else:
        print(f"  ✓ All inertias >= {min_inertia}")
    
    # Check armature
    print(f"\nDOF Armature:")
    print(f"  Min: {model.dof_armature.min():.6e}")
    print(f"  Max: {model.dof_armature.max():.6e}")
    
    if model.dof_armature.min() < min_armature:
        print(f"  ⚠️  WARNING: Min armature {model.dof_armature.min():.6e} < {min_armature}")
        small_armature = np.where(model.dof_armature < min_armature)[0]
        print(f"     Problematic DOFs: {small_armature[:5]}")
    else:
        print(f"  ✓ All armatures >= {min_armature}")
    
    print(f"\n{'='*70}")
    
    return model


# Complete workflow
def fix_and_load_model(xml_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Complete workflow: fix XML, verify, and convert to MJX."""
    
    # Step 1: Fix XML
    fixed_xml_path = xml_path.replace('.xml', '_FIXED.xml')
    fix_xml_masses(xml_path, fixed_xml_path, min_mass, min_inertia, min_armature)
    
    # Step 2: Verify fixes
    model = verify_fixes(fixed_xml_path, min_mass, min_inertia, min_armature)
    
    # Step 3: Apply additional runtime fixes if needed
    print(f"\n{'='*70}")
    print(f"APPLYING RUNTIME FIXES")
    print(f"{'='*70}")
    
    # Ensure armature (sometimes doesn't transfer from XML)
    if model.dof_armature.min() < min_armature:
        print(f"Applying runtime armature fix...")
        model.dof_armature[:] = np.maximum(model.dof_armature, min_armature)
        print(f"  ✓ Armature now: {model.dof_armature.min():.6e}")
    
    # Ensure masses
    if model.body_mass.min() < min_mass:
        print(f"Applying runtime mass fix...")
        model.body_mass = np.maximum(model.body_mass, min_mass)
        print(f"  ✓ Mass now: {model.body_mass.min():.6f}")
    
    # Set solver options
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    model.opt.tolerance = 1e-6
    print(f"  ✓ Jacobian: SPARSE")
    print(f"  ✓ Tolerance: {model.opt.tolerance}")
    
    # Step 4: Convert to MJX
    print(f"\n{'='*70}")
    print(f"CONVERTING TO MJX")
    print(f"{'='*70}")
    
    mjx_model = mjx.put_model(model)
    
    print(f"MJX Model:")
    print(f"  Min body mass: {mjx_model.body_mass.min():.6f}")
    print(f"  Min armature: {mjx_model.dof_armature.min():.6e}")
    
    # Step 5: Post-MJX fixes if needed
    if mjx_model.body_mass.min() < min_mass or mjx_model.dof_armature.min() < min_armature:
        print(f"\nApplying post-MJX fixes...")
        mjx_model = mjx_model.tree_replace({
            'body_mass': jnp.maximum(mjx_model.body_mass, min_mass),
            'body_inertia': jnp.maximum(mjx_model.body_inertia, min_inertia),
            'dof_armature': jnp.maximum(mjx_model.dof_armature, min_armature),
        })
        print(f"  ✓ Final min mass: {mjx_model.body_mass.min():.6f}")
        print(f"  ✓ Final min armature: {mjx_model.dof_armature.min():.6e}")
    
    # Step 6: Test that it works
    print(f"\n{'='*70}")
    print(f"TESTING MODEL")
    print(f"{'='*70}")
    
    try:
        data = mjx.make_data(mjx_model)
        data = data.replace(
            qpos=jnp.zeros(mjx_model.nq),
            qvel=jnp.zeros(mjx_model.nv),
            qacc=jnp.zeros(mjx_model.nv)
        )
        
        # Test inverse dynamics
        data = mjx.inverse(mjx_model, data)
        print(f"  ✓ mjx.inverse() works!")
        
        # Test step
        data = mjx.step(mjx_model, data)
        print(f"  ✓ mjx.step() works!")
        
        print(f"\n{'='*70}")
        print(f"✓✓✓ MODEL READY TO USE ✓✓✓")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗✗✗ MODEL TEST FAILED ✗✗✗")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    return mjx_model, model, fixed_xml_path


# Usage Example
if __name__ == "__main__":
    import jax
    import numpy as np
    
    # Configure JAX
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU to avoid cuSolver issues
    
    print(f"User: AlexAEdwards")
    print(f"Date: 2025-11-19 18:11:59 UTC")
    print(f"JAX devices: {jax.devices()}")
    
    # Fix and load your model
    mjx_model, mj_model, fixed_xml_path = fix_and_load_model(
        xml_path="/scaled_model_no_muscles_cvt1.xml",
        min_mass=0.5,
        min_inertia=0.01,
        min_armature=0.1
    )
    
    if mjx_model is not None:
        print(f"\n✓ Model ready! Use 'mjx_model' for your simulations")
        print(f"✓ Fixed XML saved to: {fixed_xml_path}")
        print(f"\nModel stats:")
        print(f"  Bodies: {mjx_model.nbody}")
        print(f"  DOFs: {mjx_model.nv}")
        print(f"  Constraints: {mjx_model.neq}")