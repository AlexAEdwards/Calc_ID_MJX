import sys
import os

print("Python Environment Info:")
print(f"  Python version: {sys.version}")
print(f"  Python executable: {sys.executable}")
print(f"  Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")

# Check installed packages
import subprocess
result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                       capture_output=True, text=True)
print("\nInstalled MuJoCo-related packages:")
for line in result.stdout.split('\n'):
    if any(x in line.lower() for x in ['mujoco', 'mjx', 'dm_control']):
        print(f"  {line}")