"""
Timing Analysis Script for MJX_RunID.py Performance

Add this code snippet to your MJX_RunID.py to measure performance:

1. Add at the start of the main loop (before "for t in range(num_timesteps):"):
```python
import time
loop_start_time = time.time()
print(f"Starting main computation loop for {num_timesteps} timesteps...")
```

2. Add at the end of the main loop (after the loop completes):
```python
loop_end_time = time.time()
loop_duration = loop_end_time - loop_start_time
print(f"\n{'='*70}")
print(f"PERFORMANCE METRICS:")
print(f"{'='*70}")
print(f"Total loop time: {loop_duration:.3f} seconds")
print(f"Time per timestep: {loop_duration/num_timesteps*1000:.2f} ms")
print(f"Timesteps per second: {num_timesteps/loop_duration:.1f}")
print(f"{'='*70}\n")
```

3. Expected Performance:
   - **Before optimization:** ~10-50 seconds total (50-250 ms per timestep)
   - **After optimization:** ~0.5-5 seconds total (2-25 ms per timestep)
   - **Speedup factor:** 5-50x depending on hardware and data size

4. To compare with old version:
   - Save old version as MJX_RunID_old.py
   - Run both versions and compare timing output
   - Verify results match using compare_with_reference_tau()

5. GPU vs CPU timing:
   - Check if JAX is using GPU: `print(jax.devices())`
   - GPU should show: [GpuDevice(id=0)]
   - CPU only shows: [CpuDevice(id=0)]
   - GPU acceleration can provide additional 10-100x speedup

6. Memory usage monitoring:
```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# ... run your computation ...

mem_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory used: {mem_after - mem_before:.1f} MB")
```

7. Profiling with JAX:
```python
# Add this to profile specific functions
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/jax-trace")

# ... run computation ...

# Stop profiling
jax.profiler.stop_trace()

# View with: tensorboard --logdir=/tmp/jax-trace
```

8. Quick benchmark script:
```python
def benchmark_loop():
    \"\"\"Benchmark just the main computation loop\"\"\"
    import time
    start = time.time()
    
    for t in range(num_timesteps):
        # ... your loop body ...
        pass
    
    duration = time.time() - start
    return duration

# Run multiple times to get average
durations = [benchmark_loop() for _ in range(3)]
avg_duration = sum(durations) / len(durations)
print(f"Average loop time: {avg_duration:.3f} seconds")
```

9. Detailed per-section timing:
```python
# Add timing for each major section
times = {}

t1 = time.time()
# Section 1: Setup external forces
times['setup_forces'] = time.time() - t1

t2 = time.time()
# Section 2: Compute moments and update external forces
times['compute_moments'] = time.time() - t2

t3 = time.time()
# Section 3: Run inverse dynamics
times['inverse_dynamics'] = time.time() - t3

t4 = time.time()
# Section 4: Store results
times['store_results'] = time.time() - t4

# Print timing breakdown
print("\\nTiming breakdown per iteration:")
for section, duration in times.items():
    print(f"  {section:20s}: {duration*1000:.2f} ms")
```

10. Check for bottlenecks:
```python
# Add this to see which operations are slow
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run your main loop ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```
"""

# Quick copy-paste timing code for immediate use
TIMING_CODE_START = """
# ============================================================================
# TIMING: Start performance measurement
# ============================================================================
import time
loop_start_time = time.time()
print(f"\\nStarting main computation loop for {num_timesteps} timesteps...")
print(f"Using JAX device: {jax.devices()[0]}")
"""

TIMING_CODE_END = """
# ============================================================================
# TIMING: End performance measurement and print results
# ============================================================================
loop_end_time = time.time()
loop_duration = loop_end_time - loop_start_time

print(f"\\n{'='*70}")
print(f"PERFORMANCE METRICS:")
print(f"{'='*70}")
print(f"Total loop time: {loop_duration:.3f} seconds")
print(f"Time per timestep: {loop_duration/num_timesteps*1000:.2f} ms")
print(f"Timesteps per second: {num_timesteps/loop_duration:.1f}")
print(f"Total timesteps: {num_timesteps}")
print(f"JAX device: {jax.devices()[0]}")
print(f"{'='*70}\\n")
"""

if __name__ == "__main__":
    print("=" * 70)
    print("JAX PERFORMANCE TIMING GUIDE")
    print("=" * 70)
    print()
    print("Quick Start:")
    print("1. Copy TIMING_CODE_START and paste before your main loop")
    print("2. Copy TIMING_CODE_END and paste after your main loop")
    print("3. Run your script and compare timing")
    print()
    print("=" * 70)
    print("CODE TO COPY (START):")
    print("=" * 70)
    print(TIMING_CODE_START)
    print()
    print("=" * 70)
    print("CODE TO COPY (END):")
    print("=" * 70)
    print(TIMING_CODE_END)
