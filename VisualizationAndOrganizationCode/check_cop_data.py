import pandas as pd
import numpy as np

# Load COP data
data_path = "PatientData/Falisse_2017_subject_01/"
cop_data_raw = pd.read_csv(data_path + "cop.csv", header=None, skiprows=1)
cop_matrix = cop_data_raw.values

print("="*70)
print("COP MATRIX DATA ANALYSIS")
print("="*70)
print(f"COP matrix shape: {cop_matrix.shape}")
print(f"\nColumn-by-column breakdown:")

for col_idx in range(cop_matrix.shape[1]):
    col_data = cop_matrix[:, col_idx]
    non_zero_count = np.count_nonzero(col_data)
    col_min = np.min(col_data)
    col_max = np.max(col_data)
    col_mean = np.mean(col_data[col_data != 0]) if non_zero_count > 0 else 0
    print(f"\nColumn {col_idx}:")
    print(f"  Non-zero values: {non_zero_count}/{len(col_data)} ({100*non_zero_count/len(col_data):.1f}%)")
    print(f"  Range: [{col_min:.4f}, {col_max:.4f}]")
    print(f"  Mean (non-zero): {col_mean:.4f}")

print("\n" + "="*70)
print("First 5 rows of COP data:")
print(cop_matrix[:5, :])
print("\n" + "="*70)
