"""
Example script showing how to import and use update_model function.
"""

from updateModel import update_model
import jax

# Configure JAX (optional, update_model will do it if needed)
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# Use the update_model function with custom parameters
mjx_model, mj_model, fixed_xml_path = update_model(
    xml_path="Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml",
    min_mass=0.25,
    min_inertia=0.01,
    min_armature=0.01
)

# Check if model loaded successfully
if mjx_model is not None:
    print(f"\n✅ SUCCESS! Model is ready to use")
    print(f"   Fixed XML: {fixed_xml_path}")
    print(f"   Bodies: {mjx_model.nbody}")
    print(f"   DOFs: {mjx_model.nv}")
    
    # Now you can use mjx_model for your simulations
    # For example:
    import mujoco.mjx as mjx
    import jax.numpy as jnp
    
    # Create data
    data = mjx.make_data(mjx_model)
    
    # Run a simulation step
    data = mjx.step(mjx_model, data)
    
    print(f"\n   ✓ Successfully ran a simulation step!")
else:
    print(f"\n❌ FAILED to load model")
