#!/usr/bin/env python3
"""
GPU Verification Script for JAX/MJX
This script checks if your system is properly configured to use GPU acceleration.
"""

import sys

print("="*70)
print("GPU CONFIGURATION CHECK")
print("="*70)

# Check JAX installation
print("\n1. Checking JAX installation...")
try:
    import jax
    import jax.numpy as jnp
    print(f"   ✅ JAX {jax.__version__} installed")
except ImportError as e:
    print(f"   ❌ JAX not installed: {e}")
    sys.exit(1)

# Check JAX backend
print("\n2. Checking JAX backend...")
backend = jax.default_backend()
print(f"   Default backend: {backend}")

if backend in ['gpu', 'cuda']:
    print(f"   ✅ GPU backend detected")
else:
    print(f"   ⚠️  Non-GPU backend: {backend}")

# List all devices
print("\n3. Available JAX devices:")
devices = jax.devices()
for i, device in enumerate(devices):
    print(f"   Device {i}: {device}")
    print(f"     - Platform: {device.platform}")
    print(f"     - Device kind: {device.device_kind}")
    if hasattr(device, 'id'):
        print(f"     - ID: {device.id}")

# Check CUDA availability
print("\n4. Checking CUDA support...")
try:
    from jax.lib import xla_bridge
    print(f"   XLA backend: {xla_bridge.get_backend().platform}")
    
    if xla_bridge.get_backend().platform == 'gpu':
        print("   ✅ CUDA GPU available")
    else:
        print(f"   ⚠️  Running on: {xla_bridge.get_backend().platform}")
except Exception as e:
    print(f"   ❌ Error checking CUDA: {e}")

# Run GPU performance test
print("\n5. Running GPU performance test...")
try:
    import time
    
    # Create large arrays
    size = 5000
    a = jnp.ones((size, size))
    b = jnp.ones((size, size))
    
    # Warm-up
    _ = jnp.dot(a, b).block_until_ready()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        c = jnp.dot(a, b).block_until_ready()
    elapsed = time.time() - start
    
    print(f"   Matrix multiplication ({size}x{size}) x 10 iterations:")
    print(f"   Time: {elapsed:.4f} seconds")
    print(f"   Average: {elapsed/10:.4f} seconds per iteration")
    
    # Rough estimate: GPU should be much faster
    if backend in ['gpu', 'cuda']:
        if elapsed < 2.0:  # Arbitrary threshold
            print(f"   ✅ Performance looks good for GPU")
        else:
            print(f"   ⚠️  Slow performance - GPU may not be used")
    else:
        print(f"   ℹ️  Running on {backend}")
        
except Exception as e:
    print(f"   ❌ Performance test failed: {e}")

# Check MuJoco/MJX
print("\n6. Checking MuJoCo/MJX installation...")
try:
    import mujoco
    from mujoco import mjx
    print(f"   ✅ MuJoCo installed")
    print(f"   ✅ MJX (MuJoCo JAX) available")
except ImportError as e:
    print(f"   ❌ MuJoCo/MJX not available: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if backend in ['gpu', 'cuda']:
    print("✅ Your system is configured to use GPU acceleration!")
    print("\nYour MJX inverse dynamics code will run on GPU.")
else:
    print(f"⚠️  Currently using: {backend}")
    print("\nTo enable GPU acceleration:")
    print("\n1. Check NVIDIA GPU:")
    print("   nvidia-smi")
    print("\n2. Install JAX with CUDA support:")
    print("   pip install --upgrade jax[cuda12]  # For CUDA 12.x")
    print("   # or")
    print("   pip install --upgrade jax[cuda11]  # For CUDA 11.x")
    print("\n3. Verify installation:")
    print("   python -c 'import jax; print(jax.devices())'")

print("="*70)
