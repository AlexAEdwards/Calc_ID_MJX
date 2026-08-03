"""Time-base resampling and stance-cycle normalisation.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. An AST closure
pass confirmed the cluster references nothing else defined in ProcessData.py.

Two different jobs live here, and they interpolate against different axes:

* ``resample_dataframes_to_uniform_timestep`` puts kinematics and force signals on
  one uniform clock. They arrive on separate time bases (motion capture and force
  plate sample at different rates), so it interpolates both onto a shared 100 Hz
  grid spanning only their overlap - the returned arrays are therefore usually
  shorter than either input.
* ``_interpolate_to_len`` / ``_interpolate_101`` resample against *normalised
  stance*, not time, which is what makes cycles of unequal duration averageable.
  101 points is the conventional 0-100 % grid.

``_interpolate_to_len`` accepts 1-D input but always returns 2-D (T, 1);
``_interpolate_101`` requires 2-D and returns zeros for inputs under two frames.

ProcessData.py re-exports every name, so its callers are unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def resample_dataframes_to_uniform_timestep(kin_time:    np.ndarray,
                                             force_time:  np.ndarray,
                                             pos:         np.ndarray,
                                             vel:         np.ndarray,
                                             accel:       np.ndarray,
                                             grf:         np.ndarray,
                                             grm:         np.ndarray,
                                             cop:         np.ndarray,
                                             dt:          float = 0.01
                                             ) -> tuple:
    """Resample all signals to a uniform 100 Hz grid."""
    t_start = max(kin_time[0],   force_time[0])
    t_end   = min(kin_time[-1],  force_time[-1])
    t_new   = np.arange(t_start, t_end, dt)

    def _interp(t_src, data):
        if data.ndim == 1:
            return interp1d(t_src, data, kind="linear",
                            fill_value="extrapolate", bounds_error=False)(t_new)
        out = np.empty((len(t_new), data.shape[1]))
        for c in range(data.shape[1]):
            out[:, c] = interp1d(t_src, data[:, c], kind="linear",
                                 fill_value="extrapolate", bounds_error=False)(t_new)
        return out

    return (
        t_new,
        _interp(kin_time,   pos),
        _interp(kin_time,   vel),
        _interp(kin_time,   accel),
        _interp(force_time, grf),
        _interp(force_time, grm),
        _interp(force_time, cop),
    )


def _interpolate_to_len(data, target_len: int) -> np.ndarray:
    """Linearly interpolate data (T, C) or (T,) to target_len frames."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    x_orig = np.linspace(0, 100, data.shape[0])
    x_new  = np.linspace(0, 100, target_len)
    out    = np.zeros((target_len, data.shape[1]))
    for c in range(data.shape[1]):
        out[:, c] = np.interp(x_new, x_orig, data[:, c])
    return out


def _interpolate_101(data: np.ndarray) -> np.ndarray:
    """Interpolate (T, C) data to exactly 101 points (0–100 % of stance)."""
    if data.shape[0] < 2:
        return np.zeros((101, data.shape[1]))
    x_orig = np.linspace(0, 100, data.shape[0])
    x_new  = np.linspace(0, 100, 101)
    out    = np.zeros((101, data.shape[1]))
    for c in range(data.shape[1]):
        out[:, c] = np.interp(x_new, x_orig, data[:, c])
    return out
