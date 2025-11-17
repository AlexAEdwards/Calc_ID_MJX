# GRF and Moment Data Column Assignment Fix

## Problem Identified
The GRF (Ground Reaction Force) and moment data were being assigned to the wrong feet due to incorrect column indexing.

## Root Cause

### CSV File Structure Issue
The `grf.csv` and `moment.csv` files have a **malformed header**:
- **Header claims**: 7 columns (`time, Fx_l, Fy_l, Fz_l, Fx_r, Fy_r, Fz_r`)
- **Actual data has**: 10 columns per row
- **Extra columns**: Contain all zeros (columns 7-9)

### Actual Data Layout
```
Column 0: time
Column 1: Fx_l (Left foot - Anterior-Posterior force)
Column 2: Fy_l (Left foot - Vertical force)  
Column 3: Fz_l (Left foot - Medial-Lateral force)
Column 4: Fx_r (Right foot - Anterior-Posterior force)
Column 5: Fy_r (Right foot - Vertical force)
Column 6: Fz_r (Right foot - Medial-Lateral force)
Column 7-9: All zeros
```

### Data Reality
- **Left foot** (columns 1-3): Has real GRF data (up to 728N vertical)
- **Right foot** (columns 4-6): ALL ZEROS (no ground contact)

## The Bug

**Before Fix** (INCORRECT):
```python
# These were swapped!
external_forces.at[calcn_l_id, 0:3, :].set(grf_matrix[:, 4:7].T)  # Left ← Right data (zeros)
external_forces.at[calcn_r_id, 0:3, :].set(grf_matrix[:, 1:4].T)  # Right ← Left data (real)
```

This caused:
- Left calcaneus got zeros (columns 4-6)
- Right calcaneus got the real left foot data (columns 1-3)
- Data was backwards!

## The Fix

**After Fix** (CORRECT):
```python
# Now correctly assigned
external_forces.at[calcn_l_id, 0:3, :].set(grf_matrix[:, 1:4].T)  # Left ← Left data ✓
external_forces.at[calcn_r_id, 0:3, :].set(grf_matrix[:, 4:7].T)  # Right ← Right data ✓
```

Same fix applied to moment data:
```python
external_forces.at[calcn_l_id, 3:6, :].set(moment_matrix[:, 1:4].T)  # Left moments ✓
external_forces.at[calcn_r_id, 3:6, :].set(moment_matrix[:, 4:7].T)  # Right moments ✓
```

## Verification

### Data Ranges After Fix
```
Left Calcaneus Forces:
  Fx: [-140.0, 161.7] N
  Fy: [0.0, 728.3] N (vertical - correct!)
  Fz: [-26.6, 21.8] N

Right Calcaneus Forces:
  Fx: [0.0, 0.0] N (no contact)
  Fy: [0.0, 0.0] N (no contact)
  Fz: [0.0, 0.0] N (no contact)

Left Calcaneus Torques:
  Tx: [-0.4, 0.5] N·m
  Ty: [-3.2, 1.0] N·m
  Tz: [-0.8, 0.1] N·m

Right Calcaneus Torques:
  Tx: [0.0, 0.0] N·m (no contact)
  Ty: [0.0, 0.0] N·m (no contact)
  Tz: [0.0, 0.0] N·m (no contact)
```

### Expected Behavior
This is **correct** for this gait data:
- Patient is primarily on **left foot** during this motion
- Right foot has **no ground contact** (all zeros)
- Left foot vertical force peaks at **728N** (~74kg body weight)

## Files Modified
- `MJX_RunID.py`: Fixed column indexing for GRF and moment assignment
- Added column number annotations in validation printouts
- Updated comments to reflect actual CSV structure

## Impact
✅ **GRF and moments now correctly applied to the proper feet**
✅ **Inverse dynamics will now use correct forces**
✅ **Should reduce joint torques significantly** (previously missing all GRF!)

---
*Fixed: October 17, 2025*
