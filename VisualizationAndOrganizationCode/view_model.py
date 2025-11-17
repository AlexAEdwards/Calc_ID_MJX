"""
Launch MuJoCo interactive viewer for the converted model
"""
import mujoco
import mujoco.viewer

# Path to your converted model
model_path = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/Results/scaled_model_cvt2.xml'

print("Loading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("✓ Model loaded successfully!")
print(f"  Bodies: {model.nbody}")
print(f"  Joints: {model.njnt}")
print(f"  Muscles: {model.nu}")
print()
print("Launching interactive viewer...")
print()
print("=" * 60)
print("VIEWER CONTROLS:")
print("=" * 60)
print("  Mouse:")
print("    - Left click + drag:     Rotate view")
print("    - Right click + drag:    Move view")
print("    - Scroll:                Zoom in/out")
print("    - Double click on body:  Select/track body")
print()
print("  Keyboard:")
print("    - Space:                 Pause/unpause simulation")
print("    - Backspace:             Reset simulation")
print("    - Tab:                   Toggle UI panels")
print("    - F1:                    Toggle help overlay")
print("    - Ctrl+P:                Take screenshot")
print("    - Esc:                   Close viewer")
print()
print("  Simulation:")
print("    - Use sliders in UI to control muscle activations")
print("    - Adjust joint positions manually")
print("    - Enable/disable contacts, forces, etc.")
print("=" * 60)
print()
print("Opening viewer... (Close the window or press Esc to exit)")
print()

# Launch the interactive viewer
# This will block until the viewer window is closed
mujoco.viewer.launch(model, data)

print("\nViewer closed. Goodbye!")
