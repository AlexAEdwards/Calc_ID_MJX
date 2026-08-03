"""
BatchConvertGRF.py

Batch converts .anc files to .mot (OpenSim) and .npy files for ground reaction forces.
This is a Python rewrite of the MATLAB ConverAnalogGRF.py script.

Author: Auto-generated
Date: January 2026
"""

import os
import glob
import numpy as np
from scipy import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


# ============================================================================
# Configuration Parameters
# ============================================================================

# Filtering parameters
FREQ_FILTERING = 12  # Low-pass cutoff frequency for force data (Hz)
BUTTERWORTH_ORDER = 4  # Order of Butterworth filter

# Force thresholds
ZERO_THRESHOLD = 20  # Forces below this (N) go to zero

# Force plate names for output
PLATE_NAMES_WALKING = ['R', 'L']  # Right foot, Left foot

# Data paths
# Define data root relative to the script location (ProcessScottData) -> Project Root -> Datasets_NAS
# This ensures it works regardless of where the script is called from
DATA_ROOT = Path(__file__).resolve().parent.parent / "Datasets_NAS/AddBiomechanicsDataset_All_npy/NeedsCleanedFromScott"
# DATA_ROOT = "../Datasets_NAS/AddBiomechanicsDataset_All_npy/AdditionalDataFromScott"
RELATIVE_EDITED_PATH = "Gait/Week1/Edited"
RELATIVE_CAL_PATH = "Gait/Week1"


# ============================================================================
# File I/O Functions
# ============================================================================

def parse_anc_file(filepath: str) -> Dict:
    """
    Parse a Cortex .anc (analog) file.
    
    Returns:
        Dictionary containing:
        - sample_rate: Sampling rate in Hz
        - channel_names: List of channel names
        - range_mv: Range in millivolts per channel
        - time: Time vector
        - data: Raw data matrix (samples x channels)
        - bit_depth: ADC bit depth
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Normalize line endings and split
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    
    # Join all header lines into one for parsing (header can span multiple lines)
    header_text = ' '.join(lines[:10])
    
    # Extract duration and number of channels from header
    duration_match = re.search(r'Duration\(Sec\.\)[:\s]+(\d+\.?\d*)', header_text)
    channels_match = re.search(r'#Channels[:\s]+(\d+)', header_text)
    bitdepth_match = re.search(r'BitDepth[:\s]+(\d+)', header_text)
    rate_match = re.search(r'PreciseRate[:\s]+(\d+\.?\d*)', header_text)
    
    duration = float(duration_match.group(1)) if duration_match else None
    num_channels = int(channels_match.group(1)) if channels_match else 23  # Default
    bit_depth = int(bitdepth_match.group(1)) if bitdepth_match else 16
    precise_rate = float(rate_match.group(1)) if rate_match else 2000.0
    
    # Parse channel names (line starting with "Name")
    name_line = None
    rate_line = None
    range_line = None
    data_start_idx = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('Name'):
            name_line = line_stripped
        elif line_stripped.startswith('Rate'):
            rate_line = line_stripped
        elif line_stripped.startswith('Range'):
            range_line = line_stripped
            data_start_idx = i + 1
            break
    
    if name_line is None or rate_line is None or range_line is None:
        raise ValueError(f"Could not parse header in {filepath}")
    
    # Parse channel names (tab-separated)
    channel_names = [x for x in re.split(r'\t+', name_line) if x and x != 'Name']
    
    # Parse sample rates (tab-separated)
    rate_parts = [x for x in re.split(r'\t+', rate_line) if x and x != 'Rate']
    rates = []
    for x in rate_parts:
        try:
            rates.append(float(x))
        except ValueError:
            pass
    sample_rate = rates[0] if rates else precise_rate
    
    # Parse ranges (tab-separated, in millivolts)
    range_parts = [x for x in re.split(r'\t+', range_line) if x and x != 'Range']
    ranges = []
    for x in range_parts:
        try:
            ranges.append(float(x))
        except ValueError:
            pass
    
    # Use number of channel names if num_channels wasn't parsed correctly
    if num_channels is None or num_channels <= 0:
        num_channels = len(channel_names)
    
    # Parse data
    data_lines = lines[data_start_idx:]
    data = []
    time = []
    
    for line in data_lines:
        line = line.strip()
        if line:
            # Split by tabs or whitespace
            parts = re.split(r'\t+|\s+', line)
            parts = [p for p in parts if p]  # Remove empty strings
            if len(parts) > 1:
                try:
                    time.append(float(parts[0]))
                    row_data = [int(x) for x in parts[1:num_channels+1]]
                    # Pad with zeros if not enough data
                    while len(row_data) < num_channels:
                        row_data.append(0)
                    data.append(row_data)
                except (ValueError, IndexError):
                    continue
    
    data = np.array(data, dtype=np.float64)
    time = np.array(time)
    ranges = np.array(ranges[:num_channels]) if len(ranges) >= num_channels else np.array(ranges + [5000]*(num_channels - len(ranges)))
    
    return {
        'sample_rate': sample_rate,
        'channel_names': channel_names[:num_channels],
        'range_mv': ranges,
        'time': time,
        'data': data,
        'bit_depth': bit_depth
    }


def parse_trc_file(filepath: str) -> Dict:
    """
    Parse a .trc (marker trajectory) file.
    
    Returns:
        Dictionary containing:
        - marker_names: List of marker names
        - sample_rate: Sampling rate in Hz
        - data: Marker data matrix (frames x coordinates)
        - time: Time vector
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    
    # Line 1: PathFileType header
    # Line 2: DataRate, CameraRate, NumFrames, NumMarkers, Units, etc.
    # Line 3: Values for above
    # Line 4: Frame#, Time, marker names...
    # Line 5: blank or X1, Y1, Z1, X2, Y2, Z2...
    # Line 6+: data
    
    # Parse header info from line 3 (tab-separated)
    header_parts = [x for x in re.split(r'\t+', lines[2].strip()) if x]
    sample_rate = float(header_parts[0]) if header_parts else 100.0
    num_markers = int(header_parts[3]) if len(header_parts) > 3 else 34
    
    # Parse marker names from line 4 (tab-separated, with empty tabs between xyz)
    marker_line_parts = re.split(r'\t+', lines[3].strip())
    # Filter out empty strings, Frame#, Time
    marker_names = [x for x in marker_line_parts if x and x not in ['Frame#', 'Time', '']]
    
    # Expected number of columns: frame + time + (num_markers * 3 for xyz)
    expected_cols = 2 + num_markers * 3
    
    # Find data start line (skip header lines and any blank lines)
    data_start_idx = 5
    for i in range(5, min(10, len(lines))):
        line = lines[i].strip()
        if line and not line.startswith('X') and not line.startswith('Frame'):
            # Check if this looks like data (starts with a number)
            try:
                float(line.split()[0])
                data_start_idx = i
                break
            except (ValueError, IndexError):
                continue
    
    # Parse data
    data_lines = lines[data_start_idx:]
    data = []
    time = []
    
    for line in data_lines:
        line = line.strip()
        if line:
            # Split by tabs
            parts = re.split(r'\t+', line)
            parts = [p for p in parts if p]  # Remove empty strings
            
            if len(parts) >= 3:  # At least frame, time, and some data
                try:
                    # Frame number is parts[0], time is parts[1], rest is marker data
                    time.append(float(parts[1]))
                    row_data = []
                    for val in parts[2:]:
                        try:
                            row_data.append(float(val))
                        except ValueError:
                            row_data.append(np.nan)
                    
                    # Ensure consistent row length (pad or truncate)
                    target_len = num_markers * 3
                    if len(row_data) < target_len:
                        row_data.extend([np.nan] * (target_len - len(row_data)))
                    elif len(row_data) > target_len:
                        row_data = row_data[:target_len]
                    
                    data.append(row_data)
                except (ValueError, IndexError):
                    continue
    
    data = np.array(data, dtype=np.float64)
    time = np.array(time)
    
    return {
        'marker_names': marker_names,
        'sample_rate': sample_rate,
        'data': data,
        'time': time,
        'num_markers': num_markers
    }


def parse_cal_file(filepath: str) -> Dict:
    """
    Parse a Cortex .cal calibration file to extract force plate configuration.
    
    Returns:
        Dictionary containing force plate calibration data for each plate:
        - sensitivity_matrix: 6x6 calibration matrix (V -> N, N*mm)
        - cop_offset: Center of pressure offset [x, y, z]
        - position: Position of force plate origin [x, y, z]
        - rotation_matrix: 3x3 rotation matrix for coordinate transformation
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the [Force Plate Config] section
    fp_section_match = re.search(r'\[Force Plate Config\](.*?)(?=\[|$)', content, re.DOTALL)
    if not fp_section_match:
        raise ValueError(f"Could not find [Force Plate Config] section in {filepath}")
    
    fp_section = fp_section_match.group(1)
    
    # Parse each force plate
    # Format for each plate:
    # <plate_number><type>
    # <scale> <width> <length>
    # <6x6 sensitivity matrix - 6 lines>
    # <cop_offset x y z>
    # <position x y z>
    # <rotation row 1>
    # <rotation row 2>
    # <rotation row 3>
    # ...
    
    plates = {}
    lines = fp_section.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for plate definition (e.g., "1Bertec", "2Bertec", "3Bertec")
        plate_match = re.match(r'^(\d+)(\w+)', line)
        if plate_match:
            plate_num = int(plate_match.group(1))
            plate_type = plate_match.group(2)
            
            i += 1
            # Scale, width, length (convert cm to meters)
            dims = np.array([float(x) for x in lines[i].split()])
            scale = dims[0] if len(dims) > 0 else 1.0
            width = dims[1] / 100.0 if len(dims) > 1 else 0.0
            length = dims[2] / 100.0 if len(dims) > 2 else 0.0
            
            # 6x6 sensitivity matrix
            sensitivity = []
            for j in range(6):
                i += 1
                row = [float(x) for x in lines[i].split()]
                sensitivity.append(row)
            sensitivity = np.array(sensitivity)
            
            # COP offset (convert cm to meters)
            i += 1
            cop_offset = np.array([float(x) for x in lines[i].split()]) / 100.0
            
            # Position (convert cm to meters)
            i += 1
            position = np.array([float(x) for x in lines[i].split()]) / 100.0
            
            # Rotation matrix (3 rows)
            rotation = []
            for j in range(3):
                i += 1
                row = [float(x) for x in lines[i].split()]
                rotation.append(row)
            rotation = np.array(rotation)
            
            plates[plate_num] = {
                'type': plate_type,
                'scale': scale,
                'width': width,
                'length': length,
                'sensitivity': sensitivity,
                'cop_offset': cop_offset,
                'position': position,
                'rotation': rotation
            }
        
        i += 1
    
    return plates


# ============================================================================
# Signal Processing Functions
# ============================================================================

def convert_raw_to_volts(data: np.ndarray, ranges: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """
    Convert raw ADC data to volts.
    
    Args:
        data: Raw data matrix (samples x channels)
        ranges: Range in millivolts for each channel
        bit_depth: ADC bit depth (default 16)
    
    Returns:
        Data in volts
    """
    # 16-bit system: 2^16 = 65536
    # Range is in millivolts, multiply by 0.001 to get volts
    scale = (2 * ranges * 0.001) / (2**bit_depth)
    return data * scale


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to data.
    
    Args:
        data: Input data (samples x channels)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered data
    """
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Handle multi-dimensional data
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered


def zero_below_threshold(data: np.ndarray, fz_col: int, threshold: float) -> np.ndarray:
    """
    Zero out force data when vertical force is below threshold.
    
    Args:
        data: Force data matrix
        fz_col: Column index for vertical force (Fz)
        threshold: Force threshold in Newtons
    
    Returns:
        Data with forces zeroed during non-contact
    """
    data_out = data.copy()
    below_threshold = np.abs(data[:, fz_col]) < threshold
    data_out[below_threshold, :] = 0
    return data_out


# ============================================================================
# Force Plate Processing Functions
# ============================================================================

def volts_to_forces_og(force_volts: np.ndarray, plate_config: Dict, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert filtered voltage signals to forces, global COPs, and free moments
    following the MATLAB Analog2Force_OG.m logic.
    
    Returns:
        f_lab: Forces in lab frame (N)
        cop_lab: COP in lab frame (meters)
        m_free_lab: Free moment (Tz) in lab frame (N*m)
    """
    # 1. Apply sensitivity matrix: forces_electrical = (CalibMat * volts')'
    sensitivity = plate_config['sensitivity']
    forces_electrical = force_volts @ sensitivity.T # Result is [Fx, Fy, Fz, Mx, My, Mz]
    
    # 2. Extract local forces and moments
    f_local = forces_electrical[:, 0:3]
    m_local = forces_electrical[:, 3:6]
    
    # 3. Rotate forces and moments to lab global frame
    rot = plate_config['rotation']
    f_lab = f_local @ rot.T
    m_lab_rot = m_local @ rot.T
    
    # 4. Apply Parallel Axis Theorem to shift moments to the Lab Origin
    # r is the vector from Lab Origin to Electronic Origin in Lab Frame
    # position = Lab location of plate center
    # cop_offset = Electronic-to-Physical surface offset in local frame (usually [0, 0, -0.006])
    pos = plate_config['position']
    off_local = plate_config['cop_offset']
    r_origin = pos + (off_local @ rot.T)
    
    # M_lab_origin = M_lab_rot + r x F_lab
    m_lab_origin = m_lab_rot + np.cross(r_origin, f_lab)
    
    # Convert Action Force (Platform Frame) to Reaction Force (GRF)
    # The platform measures the force of the subject pushing down (-Z)
    # We want the Ground Reaction Force pushing up (+Z)
    f_lab = -f_lab
    m_lab_origin = -m_lab_origin
    
    # 5. Calculate COP and Free Moment in Lab Frame
    n_samples = f_lab.shape[0]
    cop_lab = np.zeros((n_samples, 3))
    m_free_lab = np.zeros((n_samples, 3))
    
    # Vertical force is index 2 (Fz) in Lab Frame (Z-up)
    fz = f_lab[:, 2]
    is_contact = np.abs(fz) > threshold
    
    if np.any(is_contact):
        # COPx = -M_lab_y / Fz
        # COPy = M_lab_x / Fz
        # Note: These equations assume Fz is positive upwards. 
        # If Fz is negative (pushing down), signs might flip depending on convention.
        # Standard: COPx = -My/Fz, COPy = Mx/Fz for reaction force acting AT COP.
        
        cop_lab[is_contact, 0] = -m_lab_origin[is_contact, 1] / fz[is_contact]
        cop_lab[is_contact, 1] = m_lab_origin[is_contact, 0] / fz[is_contact]
        cop_lab[is_contact, 2] = 0 # Lab floor is Z=0
        
        # Free moment (vertical moment at COP)
        # Tz = M_lab_z - (COPx * F_lab_y - COPy * F_lab_x)
        # Cross product term (r x F)_z = (x*Fy - y*Fx)
        # So Mz_origin = Tz + (x*Fy - y*Fx)
        # Tz = Mz_origin - (x*Fy - y*Fx)
        m_free_lab[is_contact, 2] = m_lab_origin[is_contact, 2] - (
            cop_lab[is_contact, 0] * f_lab[is_contact, 1] - 
            cop_lab[is_contact, 1] * f_lab[is_contact, 0]
        )
        
    return f_lab, cop_lab, m_free_lab


def process_overground_forces(anc_data: Dict, plate_configs: Dict, 
                              filter_freq: float = 12, threshold: float = 20) -> Dict:
    """
    Process overground force plate data.
    """
    sample_rate = anc_data['sample_rate']
    channel_names = [name.upper() for name in anc_data['channel_names']]
    raw_data = anc_data['data']
    ranges = anc_data['range_mv']
    bit_depth = anc_data['bit_depth']
    time = anc_data['time']
    
    # Convert to volts
    data_volts = convert_raw_to_volts(raw_data, ranges, bit_depth)
    
    # Process each force plate
    results = {
        'time': time,
        'sample_rate': sample_rate,
        'plates': {}
    }
    
    for plate_num in sorted(plate_configs.keys()):
        # Find indices for this plate (Fx, Fy, Fz, Mx, My, Mz)
        ch_names = [f'F{plate_num}X', f'F{plate_num}Y', f'F{plate_num}Z', 
                    f'M{plate_num}X', f'M{plate_num}Y', f'M{plate_num}Z']
        
        indices = []
        missing = False
        for name in ch_names:
            try:
                indices.append(channel_names.index(name))
            except ValueError:
                missing = True
                break
        
        if missing:
            continue
            
        # Extract data in volts
        plate_data_volts = data_volts[:, indices]
        
        # Low-pass filter volts before conversion
        plate_volts_filt = lowpass_filter(plate_data_volts, filter_freq, sample_rate, BUTTERWORTH_ORDER)
        
        # Convert to forces/COPs/moments in Lab Frame
        config = plate_configs[plate_num]
        f_lab, cop_lab, m_free_lab = volts_to_forces_og(plate_volts_filt, config, threshold)
        
        # Zero out when below threshold
        is_below = np.abs(f_lab[:, 2]) < threshold
        f_lab[is_below, :] = 0
        cop_lab[is_below, :] = 0
        m_free_lab[is_below, :] = 0
        
        results['plates'][plate_num] = {
            'forces': f_lab,
            'cop': cop_lab,
            'moments': m_free_lab
        }
    
    return results


def assign_forces_to_feet(plate_data: Dict, trc_data: Dict, 
                          analog_sample_rate: float) -> Tuple[Dict, Dict]:
    """
    Assign force plate data to left and right feet based on marker positions.
    
    Uses calcaneous marker positions (R_CALC, L_CALC) to determine which foot is on which plate based on the closest distance to the Center of Pressure.
    
    Args:
        plate_data: Processed force plate data in Lab Global Frame
        trc_data: Marker trajectory data in Lab Global Frame
        analog_sample_rate: Analog sampling rate (Hz)
    
    Returns:
        Tuple of (right_foot_data, left_foot_data)
    """
    marker_names = trc_data['marker_names']
    marker_data = trc_data['data']
    marker_time = trc_data['time']
    # marker_rate = trc_data['sample_rate']
    
    # Find calcaneous markers
    r_calc_idx = None
    l_calc_idx = None
    
    for i, name in enumerate(marker_names):
        name_lower = name.lower()
        if name_lower == 'r_calc' or name_lower == 'r.calc':
            r_calc_idx = i * 3
        elif name_lower == 'l_calc' or name_lower == 'l.calc':
            l_calc_idx = i * 3
    
    if r_calc_idx is None or l_calc_idx is None:
        # Fallback to general calc markers if specific ones not found
        for i, name in enumerate(marker_names):
            if 'calc' in name.lower():
                if name.lower().startswith('r'): r_calc_idx = i * 3
                if name.lower().startswith('l'): l_calc_idx = i * 3

    if r_calc_idx is None or l_calc_idx is None:
        # raise ValueError(f"Could not find calcaneous markers in TRC file. Found: {marker_names}")
        # If no calc markers, assume order based on plate position or skip
        # For now, return empty
        print("    WARNING: No calcaneous markers found for foot assignment.")
        return ({'forces': None}, {'forces': None})
    
    # Get marker positions in meters (Lab frame)
    r_calc_pos = marker_data[:, r_calc_idx:r_calc_idx+3] / 1000.0
    l_calc_pos = marker_data[:, l_calc_idx:l_calc_idx+3] / 1000.0
    
    # Initialize output
    n_samples = len(plate_data['time'])
    
    # Data for combining multiple plates per foot
    # Using Lab frame coordinates here (Lab Z is vertical)
    right_f = np.zeros((n_samples, 3))
    right_m = np.zeros((n_samples, 3))
    right_cop_num = np.zeros((n_samples, 3))  # COP weighted by Fz
    right_fz_sum = np.zeros((n_samples, 1))
    
    left_f = np.zeros((n_samples, 3))
    left_m = np.zeros((n_samples, 3))
    left_cop_num = np.zeros((n_samples, 3))
    left_fz_sum = np.zeros((n_samples, 1))
    
    # For each plate, determine if it's left or right foot
    for plate_num, data in plate_data['plates'].items():
        cop_plate = data['cop']  # meters, Lab frame
        forces_plate = data['forces']  # N, Lab frame
        moments_plate = data['moments'] # N*m, Lab frame (Free moment)
        
        # Take values at marker time points for assignment decision
        # Resample COP/Forces to marker rate? No, just compare at marker indices?
        # Better: Interpolate marker data to analog time? Or subsample analog to marker time?
        # The logic is only needed to decide WHICH foot.
        # Let's subsample analog to marker indices for checking distance.
        
        indices = (marker_time * analog_sample_rate).astype(int)
        indices = indices[indices < n_samples]
        
        cop_ds = cop_plate[indices]
        forces_ds = forces_plate[indices]
        
        # Lab Z is vertical force index 2
        contact_frames = np.where(np.abs(forces_ds[:, 2]) > ZERO_THRESHOLD)[0]
        
        if len(contact_frames) < 5:
            continue
        
        # 2D distance on floor (Lab X, Y)
        cop_c = cop_ds[contact_frames]
        r_c = r_calc_pos[contact_frames]
        l_c = l_calc_pos[contact_frames]
        
        # Calculate mean distance during contact
        dist_to_r = np.nanmean(np.sqrt((cop_c[:, 0] - r_c[:, 0])**2 + (cop_c[:, 1] - r_c[:, 1])**2))
        dist_to_l = np.nanmean(np.sqrt((cop_c[:, 0] - l_c[:, 0])**2 + (cop_c[:, 1] - l_c[:, 1])**2))
        
        # Assign to closer foot
        fz = forces_plate[:, [2]]
        if dist_to_r < dist_to_l:
            # print(f"    Plate {plate_num}: Dist to R={dist_to_r:.3f}m, Dist to L={dist_to_l:.3f}m -> RIGHT")
            right_f += forces_plate
            right_m += moments_plate
            right_cop_num += cop_plate * fz
            right_fz_sum += fz
        else:
            # print(f"    Plate {plate_num}: Dist to R={dist_to_r:.3f}m, Dist to L={dist_to_l:.3f}m -> LEFT")
            left_f += forces_plate
            left_m += moments_plate
            left_cop_num += cop_plate * fz
            left_fz_sum += fz
            
        print(f"    [DEBUG] Plate {plate_num} Assignment:")
        print(f"      COP Mean: {np.nanmean(cop_c, axis=0)}")
        print(f"      R_Calc Mean: {np.nanmean(r_c, axis=0)}")
        print(f"      L_Calc Mean: {np.nanmean(l_c, axis=0)}")
        print(f"      Dist R: {dist_to_r:.3f}, Dist L: {dist_to_l:.3f} -> {'RIGHT' if dist_to_r < dist_to_l else 'LEFT'}")
            
    # Calculate final COP for each foot (weighted average)
    def finalize_cop(cop_num, fz_sum):
        cop = np.zeros_like(cop_num)
        # Use abs() because Fz is potentially negative (downward force)
        valid = np.abs(fz_sum[:, 0]) > ZERO_THRESHOLD
        cop[valid] = cop_num[valid] / fz_sum[valid]
        return cop

    right_cop = finalize_cop(right_cop_num, right_fz_sum)
    left_cop = finalize_cop(left_cop_num, left_fz_sum)
    
    # DEBUG: Check output
    r_force_max = np.max(np.abs(right_f))
    l_force_max = np.max(np.abs(left_f))
    r_cop_std = np.std(right_cop, axis=0)
    l_cop_std = np.std(left_cop, axis=0)
    
    if r_force_max > ZERO_THRESHOLD:
        print(f"    [DEBUG] Right Foot Assigned (Max F={r_force_max:.1f}N). COP Std: {r_cop_std}")
    else:
        print(f"    [DEBUG] Right Foot: NO FORCE ASSIGNED")
        
    if l_force_max > ZERO_THRESHOLD:
        print(f"    [DEBUG] Left Foot Assigned (Max F={l_force_max:.1f}N). COP Std: {l_cop_std}")
    else:
        print(f"    [DEBUG] Left Foot: NO FORCE ASSIGNED")

    if r_force_max > ZERO_THRESHOLD and l_force_max > ZERO_THRESHOLD:
        # Check for overlap if requested (if both assigned same plate? Logic prevents this as per-plate loop)
        # However, checking if both feet have suspicious similarity?
        pass

    return (
        {'forces': right_f, 'cop': right_cop, 'moments': right_m},
        {'forces': left_f, 'cop': left_cop, 'moments': left_m}
    )


def transform_to_opensim(forces: np.ndarray, cop: np.ndarray, 
                         moments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform forces/COP/moments to OpenSim coordinate system.
    
    Lab frame: X-forward, Y-lateral (Left), Z-up
    OpenSim frame: X-forward, Y-up, Z-lateral (Right)
    
    Transformation:
    X_osim = X_lab
    Y_osim = Z_lab
    Z_osim = -Y_lab
    
    Rotation Matrix R:
    [1  0  0]
    [0  0  1]
    [0 -1  0]
    
    v_osim = R * v_lab
    """
    R = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    
    f_osim = forces @ R.T
    cop_osim = cop @ R.T
    m_osim = moments @ R.T
    
    return f_osim, cop_osim, m_osim


def write_opensim_mot(filepath: str, time: np.ndarray, 
                      right_data: Dict, left_data: Dict):
    """
    Write data to OpenSim .mot file.
    """
    # Header
    header = "coordinates\nversion=1\nnRows={}\nnColumns={}\ninDegrees=yes\nendheader\n".format(
        len(time), 19
    )
    
    # Columns: time, R_ground_force_vx, vy, vz, px, py, pz, torque_x, y, z, ...
    cols = ["time"]
    sides = [("R", right_data), ("L", left_data)]
    
    data_matrix = [time]
    
    for side, data in sides:
        # Force (vx, vy, vz)
        cols.extend([f"{side}_ground_force_vx", f"{side}_ground_force_vy", f"{side}_ground_force_vz"])
        # COP (px, py, pz)
        cols.extend([f"{side}_ground_force_px", f"{side}_ground_force_py", f"{side}_ground_force_pz"])
        # Moment (torque_x, y, z)
        cols.extend([f"{side}_ground_torque_x", f"{side}_ground_torque_y", f"{side}_ground_torque_z"])
        
        if data['forces'] is None:
            # Zero fill
            zeros = np.zeros((len(time), 3))
            data_matrix.extend([zeros[:,0], zeros[:,1], zeros[:,2]])
            data_matrix.extend([zeros[:,0], zeros[:,1], zeros[:,2]])
            data_matrix.extend([zeros[:,0], zeros[:,1], zeros[:,2]])
        else:
            # Transform to OpenSim frame
            f_osim, cop_osim, m_osim = transform_to_opensim(
                data['forces'], data['cop'], data['moments']
            )
            data_matrix.extend([f_osim[:,0], f_osim[:,1], f_osim[:,2]])
            data_matrix.extend([cop_osim[:,0], cop_osim[:,1], cop_osim[:,2]])
            data_matrix.extend([m_osim[:,0], m_osim[:,1], m_osim[:,2]])
            
    # Write file
    data_arr = np.column_stack(data_matrix)
    
    with open(filepath, 'w') as f:
        f.write(header)
        f.write('\t'.join(cols) + '\n')
        np.savetxt(f, data_arr, delimiter='\t', fmt='%.6f')


def main():
    root_dir = Path(DATA_ROOT)
    
    # Find all subject folders
    subj_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("Subject")])
    
    # Pre-scan for common calibration files in the root directory
    common_cal_files = list(root_dir.glob("*.cal"))
    
    print("="*70)
    print(f"BatchConvertGRF.py - Batch GRF Conversion Tool")
    print("="*70)
    print(f"Data root: {root_dir}")
    print(f"Filter frequency: {FREQ_FILTERING} Hz")
    print(f"Force threshold: {ZERO_THRESHOLD} N")
    print("="*70)
    
    if not subj_dirs:
        print("No subject directories found!")
        return

    print(f"Found {len(subj_dirs)} subject directories")
    print("="*70)
    
    for subj_dir in subj_dirs:
        print(f"\nProcessing {subj_dir.name}...")
        
        # Load Calibration File
        # Expected location: Subject<#>/Gait/Week1/
        cal_dir = subj_dir / RELATIVE_CAL_PATH
        
        # 1. Look for Overground setup files first (OG_Setup*.cal)
        cal_files = sorted(list(cal_dir.glob("OG_Setup*.cal")))
        
        # 2. If not found, look for any .cal file in that specific directory
        if not cal_files:
            cal_files = sorted(list(cal_dir.glob("*.cal")))
        
        # 3. If still not found, try recursive search in subject dir
        if not cal_files:
            cal_files = sorted(list(subj_dir.rglob("*.cal")))
            
        # 4. Fallback to root directory
        if not cal_files:
             print(f"  ⚠️  No .cal file in {subj_dir.name}. Checking root...")
             cal_files = common_cal_files
        
        if not cal_files:
             print("  ERROR: No .cal file found anywhere. Skipping.")
             continue
            
        # Select the best file (prefer OG_Setup)
        cal_file = next((f for f in cal_files if "OG_Setup" in f.name), cal_files[0])
        print(f"  Using calibration: {cal_file} (Found {len(cal_files)} candidates)")
        
        try:
            plate_configs = parse_cal_file(str(cal_file))
            # Debug output for plate configs
            print("  [DEBUG] Loaded Plate Configurations:")
            for p_id, p_conf in plate_configs.items():
                pos = p_conf['position']
                print(f"    Plate {p_id} ({p_conf['type']}): Pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
        except Exception as e:
            print(f"  ERROR parsing calibration file: {e}")
            continue
            
        # Find .anc files in the Edited directory only
        edited_dir = subj_dir / RELATIVE_EDITED_PATH
        if not edited_dir.exists():
            print(f"  Skipping: Edited directory not found at {edited_dir}")
            continue
            
        anc_files = sorted(list(edited_dir.glob("*.anc")))
        print(f"  Found {len(anc_files)} .anc files in {RELATIVE_EDITED_PATH}")
        
        # Pre-scan for all .trc files to avoid repeated directory scanning
        print("  Scanning for .trc files...")
        all_trc_files = sorted(list(subj_dir.rglob("*.trc")))
        trc_file_map = {f.stem: f for f in all_trc_files}
        print(f"  Found {len(all_trc_files)} .trc files")
        
        for anc_file in anc_files:
            trial_name = anc_file.stem
            
            # Find corresponding .trc file from map
            if trial_name not in trc_file_map:
                # Try partial match? Or just skip
                # print(f"    Skipping: No matching .trc file for {trial_name}")
                continue
            
            trc_file = trc_file_map[trial_name]
            
            print(f"  Processing: {trial_name}")
            
            try:
                # Parse files
                anc_data = parse_anc_file(str(anc_file))
                trc_data = parse_trc_file(str(trc_file))
                
                # Check for F3X channel (Scott data specific check from logs)
                # "Skipping: Not an overground trial (no F3X channel)"
                if 'F3X' not in anc_data['channel_names'] and 'f3x' not in anc_data['channel_names']:
                     print(f"    Skipping: Not an overground trial (no F3X channel)")
                     continue
                
                # Process Forces
                processed_plates = process_overground_forces(
                    anc_data, plate_configs, 
                    filter_freq=FREQ_FILTERING, 
                    threshold=ZERO_THRESHOLD
                )
                
                # Assign to feet
                right_foot, left_foot = assign_forces_to_feet(
                    processed_plates, trc_data, anc_data['sample_rate']
                )
                
                time_vec = processed_plates['time']
                n_samples = len(time_vec)
                zeros_3 = np.zeros((n_samples, 3))
                
                if right_foot['forces'] is None:
                    # Failed assignment, fill with zeros
                    print(f"      WARNING: Foot assignment failed (missing markers). Outputting zeros.")
                    r_f_osim = zeros_3
                    r_cop_osim = zeros_3
                    r_m_osim = zeros_3
                    l_f_osim = zeros_3
                    l_cop_osim = zeros_3
                    l_m_osim = zeros_3
                else:
                    # Create Output Directory
                    # Structure: Data_Full_Cleaned/{Subject}/... replaced with current output path?
                    # Or just save next to .anc file?
                    # Log said: "Successfully processed..."
                    # Previous output paths were likely:
                    # {subj_dir}/calculatedInputs/External_Force.mot ?
                    # Or batch_convert_output folder?
                    
                    # Debug data content to verify non-zero values
                    f_max_r = np.max(np.abs(right_foot['forces']))
                    f_max_l = np.max(np.abs(left_foot['forces']))
                    print(f"      [DEBUG] Max Force Magnitude: R={f_max_r:.1f}N, L={f_max_l:.1f}N")

                    # Transform data to OpenSim frame for saving as .npy
                    # Visualize scripts expect OpenSim frame data (Y-up)
                    r_f_osim, r_cop_osim, r_m_osim = transform_to_opensim(
                        right_foot['forces'], right_foot['cop'], right_foot['moments']
                    )
                    l_f_osim, l_cop_osim, l_m_osim = transform_to_opensim(
                        left_foot['forces'], left_foot['cop'], left_foot['moments']
                    )
                
                # Save to specific output directory: SubjectXXX/Gait/Week1/Edited/GRFmot/TrialName
                out_dir = subj_dir / RELATIVE_EDITED_PATH / "GRFmot" / trial_name
                out_dir.mkdir(parents=True, exist_ok=True)
                
                out_path_mot = out_dir / f"{trial_name}_grf.mot"
                
                # Combined GRF [Right, Left] (N x 6)
                # First 3 cols: Right X, Y, Z. Last 3 cols: Left X, Y, Z
                grf_combined = np.hstack([r_f_osim, l_f_osim])
                
                # Combined Moment [Right, Left] (N x 6)
                moment_combined = np.hstack([r_m_osim, l_m_osim])
                
                # Combined COP [Right, Left] (N x 6)
                cop_combined = np.hstack([r_cop_osim, l_cop_osim])
                
                # Save .npy files
                np.save(str(out_dir / 'GRF.npy'), grf_combined)
                np.save(str(out_dir / 'Moment.npy'), moment_combined)
                np.save(str(out_dir / 'COP.npy'), cop_combined)
                
                # Save time vector
                np.save(str(out_dir / 'time.npy'), time_vec)
                
                # Save legacy individual files just in case (optional, but harmless)
                np.save(str(out_dir / 'right_forces.npy'), r_f_osim)
                np.save(str(out_dir / 'left_forces.npy'), l_f_osim)
                np.save(str(out_dir / 'right_cop.npy'), r_cop_osim)
                np.save(str(out_dir / 'left_cop.npy'), l_cop_osim)

                write_opensim_mot(str(out_path_mot), time_vec, right_foot, left_foot)
                
                print(f"    ✓ Success -> {out_dir}")
                # print(f"      Saved .mot and .npy (with Force, COP, Moment)")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
