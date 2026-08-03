"""Z-score normalisation of model inputs and targets.

Extracted verbatim from train.py in REFACTOR_PLAN.md Stage 5. train.py re-exports
these names, so every existing import keeps working.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


class Normalizer:
    def __init__(self, data: np.ndarray = None, eps: float = 1e-8, name: str = "unknown"):
        if data is not None:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
            # Check for near-zero std and warn
            below_eps = self.std < eps
            if np.any(below_eps):
                bad_indices = np.where(below_eps.flatten())[0]
                bad_stds = self.std.flatten()[bad_indices]
                print(f"   ⚠️  Normalizer '{name}': {len(bad_indices)} dim(s) have std < {eps}. "
                      f"Indices: {bad_indices.tolist()}, Stds: {bad_stds.tolist()}. Clamping to {eps}.", flush=True)
            self.std = np.where(self.std < eps, eps, self.std)
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def unnormalize(self, x):
        return x * self.std + self.mean


def normalize_batch(batch: Dict, normalizers: Dict) -> Dict:
    """Apply normalization to a batch."""
    normalized = {}
    for key, val in batch.items():
        if key == "input" and "input" in normalizers:
            normalized[key] = normalizers["input"].normalize(val)

        elif key == "static_context" and "static" in normalizers:
            normalized[key] = normalizers["static"].normalize(val)

        # Z-score outputs
        elif key == "cop" and "cop" in normalizers:
            normalized[key] = normalizers["cop"].normalize(val)
        elif key == "grf" and "grf" in normalizers:
            normalized[key] = normalizers["grf"].normalize(val)
        elif key == "moments" and "moments" in normalizers:
            normalized[key] = normalizers["moments"].normalize(val)

    # Z-score reconstruction curves too; they stay available for diagnostics.
        elif key == "cop_recon" and "cop" in normalizers:
            normalized[key] = normalizers["cop"].normalize(val)
        elif key == "grf_recon" and "grf" in normalizers:
            normalized[key] = normalizers["grf"].normalize(val)
        elif key in ["moment_recon", "moments_recon"] and "moments" in normalizers:
            normalized[key] = normalizers["moments"].normalize(val)

        # Leave torque in Nm (raw)
        elif key == "qfrc_grf_contribution":
            normalized[key] = val
        elif key == "jacobian_gt" and "jacobian" in normalizers:
            normalized[key] = normalizers["jacobian"].normalize(val)

        else:
            # pass through jacp, jacr, ankle_heights, contactBoolean, body_ids, height, mass, etc.
            normalized[key] = val

    return normalized
