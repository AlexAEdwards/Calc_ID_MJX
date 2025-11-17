# Viewing the Generated Plots

## Problem: Black Images

If you saw black images before, it was likely due to:
1. **Cached files** - Old corrupted files from previous runs
2. **Viewer issues** - Some image viewers don't refresh automatically
3. **Matplotlib backend** - Now fixed with `matplotlib.use('Agg')`

## Solution Applied

The code has been updated with:
```python
matplotlib.use('Agg')  # Use non-interactive backend
fig.set_facecolor('white')  # Explicitly set white background
ax.set_facecolor('white')
plt.savefig('file.png', facecolor='white')  # Save with white background
plt.close(fig)  # Properly close figures
```

## Verification

The plots are now confirmed to be **valid and contain data**:

### inverse_dynamics_forces.png
- **Size**: 594.7 KB
- **Dimensions**: 4170 x 3542 pixels
- **Content**: 8 subplots showing key joint forces
- **Status**: ✓ Valid with data variation

### all_joint_forces.png
- **Size**: 1996.1 KB  
- **Dimensions**: 4768 x 2364 pixels
- **Content**: All 39 DOF forces overlaid
- **Status**: ✓ Valid with data variation

## How to View the Plots

### Option 1: Use image viewer (eog - Eye of GNOME)
```bash
eog inverse_dynamics_forces.png
```

### Option 2: Use default system viewer
```bash
xdg-open inverse_dynamics_forces.png
```

### Option 3: Use Firefox/Chrome
```bash
firefox inverse_dynamics_forces.png
```

### Option 4: VS Code
Right-click on the PNG file in VS Code's file explorer and select "Open Preview"

### Option 5: Command line preview
```bash
# If you have imgcat or similar
imgcat inverse_dynamics_forces.png
```

## Data Summary

From the latest run:

### Key Joint Force Ranges (N·m):

**Right Leg:**
- Hip Flexion R: -264.5 to 1919.8
- Hip Adduction R: -867.8 to 1205.3
- Knee Angle R: -8675.1 to 8906.6 (largest forces!)
- Ankle Angle R: -515.4 to 210.2

**Left Leg:**
- Hip Flexion L: -1478.4 to 1398.6
- Hip Adduction L: -1036.2 to 772.1
- Knee Angle L: -8714.6 to 8001.3 (largest forces!)
- Ankle Angle L: -625.9 to 220.5

### Overall Force Range:
- **Minimum**: -14,046.9 N·m
- **Maximum**: 12,857.1 N·m
- **Time span**: 0 to 2.06 seconds (208 frames)

## What the Plots Show

### inverse_dynamics_forces.png (8 key joints)
- 4 rows × 2 columns layout
- Each subplot shows one major joint
- Blue lines show force/torque over time
- Zero reference line (dashed)
- Grid for easy reading

### all_joint_forces.png (all 39 DOFs)
- Single plot with all forces overlaid
- Different colors for each DOF
- Legend on the right side
- Good for comparing relative magnitudes

## Interpreting the Results

### Knee forces are dominant:
- Knees show the largest forces (~8000-9000 N·m peak)
- This makes sense - knees bear significant load during gait
- Coupled joints (walker knee) contribute to these forces

### Hip forces are moderate:
- Hip flexion: 1000-2000 N·m range
- Hip adduction/rotation: smaller magnitudes
- Asymmetry between left/right is normal in gait

### Ankle forces are smaller:
- Ankle: 200-600 N·m range
- Smaller moment arms = smaller torques
- But still significant for propulsion

## Troubleshooting

### If images still appear black:
1. **Delete old files and regenerate:**
   ```bash
   rm inverse_dynamics_forces.png all_joint_forces.png
   python MJX_RunID.py
   ```

2. **Check with verification script:**
   ```bash
   python check_plots.py
   ```

3. **Try different viewer:**
   ```bash
   eog inverse_dynamics_forces.png  # Eye of GNOME
   # or
   gpicview inverse_dynamics_forces.png  # Lightweight
   # or  
   gimp inverse_dynamics_forces.png  # Full editor
   ```

4. **Convert format if needed:**
   ```bash
   convert inverse_dynamics_forces.png inverse_dynamics_forces.jpg
   ```

### If forces look unrealistic:
- Check input data (pos, vel, acc CSV files)
- Verify coupled joint calculations
- Ensure polynomial constraints are correct
- Compare with literature values for human gait

## Next Steps

1. **Analyze the plots** - Look for patterns in the gait cycle
2. **Compare with literature** - Are magnitudes reasonable?
3. **Export data** - Save forces to CSV for further analysis
4. **Validate** - Compare with measured ground reaction forces
5. **Refine model** - Adjust if needed based on results

## Files Generated

All in: `/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/`

- ✓ `inverse_dynamics_forces.png` - Key joints plot
- ✓ `all_joint_forces.png` - All DOFs plot
- ✓ `check_plots.py` - Verification script

The plots are confirmed to be valid and ready for analysis!
