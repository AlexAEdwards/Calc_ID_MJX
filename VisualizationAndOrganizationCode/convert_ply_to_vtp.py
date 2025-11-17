"""
Convert .vtp.ply files to .vtp files in the Geometry folder
"""
import os
import pyvista as pv
from pathlib import Path

# Define the geometry folder
geometry_folder = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/Geometry'

# Find all .ply files
ply_files = list(Path(geometry_folder).glob('*.ply'))

print(f"Found {len(ply_files)} .ply files to convert")

# Convert each file
for ply_file in ply_files:
    print(f"Converting {ply_file.name}...")
    
    try:
        # Read the PLY file
        mesh = pv.read(str(ply_file))
        
        # Determine the output VTP filename
        # If it ends with .vtp.ply, remove both extensions and add .vtp
        # Otherwise, just replace .ply with .vtp
        if ply_file.name.endswith('.vtp.ply'):
            vtp_filename = ply_file.name[:-8] + '.vtp'  # Remove .vtp.ply, add .vtp
        else:
            vtp_filename = ply_file.name[:-4] + '.vtp'  # Remove .ply, add .vtp
        
        vtp_path = Path(geometry_folder) / vtp_filename
        
        # Save as VTP
        mesh.save(str(vtp_path))
        
        print(f"  ✓ Saved as {vtp_filename}")
        
    except Exception as e:
        print(f"  ✗ Error converting {ply_file.name}: {e}")

print("\nConversion complete!")
print(f"\nYou can now remove the .ply files if the conversion was successful:")
print(f"  rm {geometry_folder}/*.ply")
