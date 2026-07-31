"""Memory-efficient data loader that loads trials on demand."""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from opensim_id_targets import load_aligned_opensim_id_target

NOISED_FILE_SUFFIX = "_noised"
GRAVITY_MPS2 = 9.8067
AUXILIARY_MODEL_INPUTS_ENABLED = False
VIDEO_SOURCE_NAMES = {"video", "processed", "processeddata", "opencap"}
MOCAP_SOURCE_NAMES = {"mocap", "motioncapture", "motion_capture"}
INDEPENDENT_MODEL_DOF_COUNT = 23
MODEL_31_TO_INDEPENDENT_INDICES = np.asarray(
    (
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 11, 14, 15, 16,
        17, 18, 19, 22, 25, 26, 27,
        28, 29, 30,
    ),
    dtype=np.int64,
)
MODEL_33_43_TO_INDEPENDENT_INDICES = np.asarray(
    (
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 12, 15, 16, 17,
        18, 19, 20, 24, 27, 28, 29,
        30, 31, 32,
    ),
    dtype=np.int64,
)
MODEL_39_TO_INDEPENDENT_INDICES = np.asarray(
    (
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 11, 14, 15, 16,
        21, 22, 23, 26, 29, 30, 31,
        36, 37, 38,
    ),
    dtype=np.int64,
)
INDEPENDENT_DOF_INDEX_BY_WIDTH = {
    31: MODEL_31_TO_INDEPENDENT_INDICES,
    33: MODEL_33_43_TO_INDEPENDENT_INDICES,
    39: MODEL_39_TO_INDEPENDENT_INDICES,
    43: MODEL_33_43_TO_INDEPENDENT_INDICES,
}


def _quiet_loader_logs() -> bool:
    return os.environ.get("MJX_DATALOADER_QUIET", "").strip().lower() in {"1", "true", "yes", "on"}


def _loader_log(message: str) -> None:
    if not _quiet_loader_logs():
        print(message)


def _verbose_dof_trim_logs() -> bool:
    return os.environ.get("MJX_DOFTRIM_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def coerce_independent_dof_width(arr: Any, *, label: str, trial_id: str = "") -> np.ndarray:
    """Return arr with the final DOF axis in the independent 23-DOF schema.

    ProcessData.py now saves Jacobian/qfrc/ID arrays at 23 DOFs, but older
    processed folders can still contain full-width 31/33/39/43 model arrays.
    This loader-side coercion lets training consume both without reprocessing.
    """
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim == 0:
        raise ValueError(f"{label} must have at least one dimension, got scalar")
    width = int(arr_np.shape[-1])
    if width == INDEPENDENT_MODEL_DOF_COUNT:
        return arr_np
    indices = INDEPENDENT_DOF_INDEX_BY_WIDTH.get(width)
    if indices is None:
        raise ValueError(
            f"{label} has unsupported DOF width {width}; expected 23 or one of "
            f"{sorted(INDEPENDENT_DOF_INDEX_BY_WIDTH)}"
        )
    if _verbose_dof_trim_logs():
        _loader_log(
            f"   [DOFTrim] {trial_id + ' ' if trial_id else ''}{label}: "
            f"{width} -> {INDEPENDENT_MODEL_DOF_COUNT}"
        )
    return np.take(arr_np, indices, axis=-1)


def normalize_input_source_name(source: str) -> str:
    """Normalize source aliases to the two processed sources in the OpenCap layout."""
    source_norm = str(source or "video").strip().lower()
    if source_norm in MOCAP_SOURCE_NAMES:
        return "mocap"
    return "video"


def subject_group_id(subject: str) -> str:
    """Group trunk-sway condition folders with their base subject for LOSO."""
    subject_str = str(subject)
    return subject_str[:-3] if subject_str.endswith("_TS") else subject_str


def resolve_trial_root_from_path(path: Path) -> Path:
    """Return the trial root for either a trial path or a source ProcessedData path."""
    path = Path(path)
    if path.name == "ProcessedData" and path.parent.name in {"Video", "MoCap"}:
        return path.parent.parent
    if path.name in {"Video", "MoCap"} and path.parent.name.startswith("trial_"):
        return path.parent
    if path.name == "ProcessedData":
        return path.parent
    return path


def source_processed_dir(trial_root: Path, source: str = "video") -> Path:
    """Resolve the selected processed source, including the legacy OpenCap layout."""
    trial_root = resolve_trial_root_from_path(Path(trial_root))
    source_norm = normalize_input_source_name(source)
    folder = "MoCap" if source_norm == "mocap" else "Video"
    nested = trial_root / folder / "ProcessedData"
    if nested.exists():
        return nested
    # Older OpenCap exports store video files directly in ProcessedData/ and
    # marker files directly in MoCap/ rather than under source/ProcessedData/.
    legacy = trial_root / ("MoCap" if source_norm == "mocap" else "ProcessedData")
    if legacy.exists():
        return legacy
    return nested


def video_processed_dir(trial_root: Path) -> Path:
    return source_processed_dir(trial_root, "video")


def mocap_processed_dir(trial_root: Path) -> Path:
    return source_processed_dir(trial_root, "mocap")


def direct_processed_dir(trial_root: Path) -> Path:
    """Resolve the TrustedDataSet-style ProcessedData directory for a trial root."""
    return resolve_trial_root_from_path(Path(trial_root)) / "ProcessedData"

# Files whose contents depend on the kinematic velocity/acceleration derivation and therefore
# have an OpenSim-filtered ("_OSfilt") variant produced by ProcessData --os-filtering. Positions
# (pos_inputs/pos_mjx) and force/COP files are identical across filtering methods.
OS_FILTER_FILES = frozenset({
    "vel_inputs.npy", "acc_inputs.npy",
    "qvel_mjx.npy", "qacc_mjx.npy",
    "qfrc_inverse.npy", "ID_GT_MJX.npy",
})


def _with_file_suffix(filename: str, suffix: str = NOISED_FILE_SUFFIX) -> str:
    path = Path(filename)
    if path.suffix:
        return f"{path.stem}{suffix}{path.suffix}"
    return f"{filename}{suffix}"


def compute_bodyweight_height_norm_factor(
    mass: Any,
    height: Any,
    *,
    xp=np,
    eps: float = 1e-8,
) -> Any:
    """Return BW*H = mass * g * height with a small lower clamp."""
    mass_arr = xp.asarray(mass, dtype=xp.float32)
    height_arr = xp.asarray(height, dtype=xp.float32)
    return xp.maximum(mass_arr * GRAVITY_MPS2 * height_arr, eps)


def normalize_qfrc_inverse_by_bw_height(
    qfrc_inverse: Any,
    mass: Any,
    height: Any,
    *,
    xp=np,
    eps: float = 1e-8,
) -> Any:
    """Convert qfrc_inverse from Nm to Nm / (BW * H)."""
    qfrc_arr = xp.asarray(qfrc_inverse, dtype=xp.float32)
    return qfrc_arr / compute_bodyweight_height_norm_factor(mass, height, xp=xp, eps=eps)


def unnormalize_qfrc_inverse_by_bw_height(
    qfrc_inverse_normalized: Any,
    mass: Any,
    height: Any,
    *,
    xp=np,
    eps: float = 1e-8,
) -> Any:
    """Convert qfrc_inverse from Nm / (BW * H) back to Nm."""
    qfrc_arr = xp.asarray(qfrc_inverse_normalized, dtype=xp.float32)
    return qfrc_arr * compute_bodyweight_height_norm_factor(mass, height, xp=xp, eps=eps)


def _resolve_subject_model_xml_path(trial_root: Path) -> Optional[Path]:
    """Resolve the subject-level model XML from a trial root."""
    trial_root = resolve_trial_root_from_path(Path(trial_root))
    subject_root = trial_root.parent
    candidates = [
        subject_root / "MyosuiteModel_Video_FIXED.xml",
        subject_root / "MyosuiteModel_MoCap_FIXED.xml",
        subject_root / "MyosuiteModel_FIXED.xml",
        subject_root / "MyosuiteModel.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def group_trials_by_subject(trials: List[Dict]) -> Dict[str, List[Dict]]:
    """Group discovered trial dicts by subject while preserving input order."""
    grouped: Dict[str, List[Dict]] = {}
    for trial in trials:
        subject = str(trial.get("subject", "")).strip()
        grouped.setdefault(subject, []).append(trial)
    return grouped


def select_pos_input_columns(pos: np.ndarray, include_pelvis_euler: bool = True) -> np.ndarray:
    """Optionally drop pelvis_tilt/list/rotation from pos_inputs."""
    pos = np.asarray(pos, dtype=np.float32)
    if include_pelvis_euler or pos.ndim != 2 or pos.shape[1] <= 3:
        return pos
    return pos[:, 3:]


def flatten_jacobian_components(jacp: np.ndarray, jacr: np.ndarray) -> np.ndarray:
    """Flatten jacp/jacr to a per-frame feature vector: [jacp_flat, jacr_flat]."""
    jacp = np.asarray(jacp, dtype=np.float32)
    jacr = np.asarray(jacr, dtype=np.float32)
    leading_shape = tuple(jacp.shape[:-3])
    jacp_flat = jacp.reshape(leading_shape + (-1,))
    jacr_flat = jacr.reshape(leading_shape + (-1,))
    return np.concatenate([jacp_flat, jacr_flat], axis=-1)


def flatten_rotation_matrices(rot: np.ndarray) -> np.ndarray:
    """Flatten (..., 2, 3, 3) rotation bundles to (..., 18)."""
    rot = np.asarray(rot, dtype=np.float32)
    leading_shape = tuple(rot.shape[:-3])
    return rot.reshape(leading_shape + (-1,))


def validate_prediction_margin(window_size: int, prediction_margin_frames: int) -> None:
    """Validate the central supervision crop configuration for a fixed window."""
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if prediction_margin_frames < 0:
        raise ValueError(
            f"prediction_margin_frames must be >= 0, got {prediction_margin_frames}"
        )
    if window_size <= 2 * prediction_margin_frames:
        raise ValueError(
            "window_size must be greater than 2 * prediction_margin_frames "
            f"(got window_size={window_size}, prediction_margin_frames={prediction_margin_frames})"
        )


def build_trial_edge_mask(trial_length: int, prediction_margin_frames: int) -> np.ndarray:
    """Return a boolean mask for trial frames eligible for numeric evaluation."""
    if trial_length <= 0:
        return np.zeros((0,), dtype=bool)
    if prediction_margin_frames <= 0:
        return np.ones((trial_length,), dtype=bool)
    frame_idx = np.arange(trial_length, dtype=np.int32)
    return (
        (frame_idx >= prediction_margin_frames)
        & (frame_idx < (trial_length - prediction_margin_frames))
    )


def build_window_supervision_mask(
    window_size: int,
    window_start_idx: int,
    trial_length: int,
    prediction_margin_frames: int,
) -> np.ndarray:
    """
    Build a per-window supervision mask that keeps only the center of the window
    and excludes the first/last `prediction_margin_frames` of the full trial.
    """
    validate_prediction_margin(window_size, prediction_margin_frames)
    if trial_length < 0:
        raise ValueError(f"trial_length must be >= 0, got {trial_length}")

    local_idx = np.arange(window_size, dtype=np.int32)
    local_valid = (
        (local_idx >= prediction_margin_frames)
        & (local_idx < (window_size - prediction_margin_frames))
    )
    absolute_idx = int(window_start_idx) + local_idx
    trial_valid = (
        (absolute_idx >= prediction_margin_frames)
        & (absolute_idx < (trial_length - prediction_margin_frames))
    )
    return (local_valid & trial_valid).astype(np.float32)[:, np.newaxis]


#: Edge-frame handling policies.
#:
#: ``legacy``
#:     Historical behaviour. ``build_window_supervision_mask`` keeps only the
#:     centre of each window (an inset of ``prediction_margin_frames`` at both
#:     window ends) *and* drops the first/last ``prediction_margin_frames`` of the
#:     trial. Edge frames therefore never receive a prediction at all.
#: ``train``
#:     Trim ``edge_trim_frames`` off each end of the trial *before* windowing, then
#:     supervise every frame of every window. Trials with fewer than
#:     ``window_size`` frames left after trimming are dropped rather than padded.
#: ``infer``
#:     No trim; window the full trial and supervise every frame, so edge frames do
#:     get predictions. Excluding them from accuracy is the caller's job (see
#:     ``scoring_mask`` in infer_directTorque.py).
EDGE_MODES = ("legacy", "train", "infer")


def build_full_window_supervision_mask(
    window_size: int,
    window_start_idx: int,
    valid_lo: int,
    valid_hi: int,
) -> np.ndarray:
    """Supervise every frame of the window that falls inside ``[valid_lo, valid_hi)``.

    Unlike ``build_window_supervision_mask`` there is no inset at the window ends -
    the model is trained on, and evaluated at, every frame it sees. The bounds still
    exclude anything outside the usable region, which is what keeps edge-padded
    frames of a short trial out of the loss.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    absolute_idx = int(window_start_idx) + np.arange(window_size, dtype=np.int32)
    valid = (absolute_idx >= int(valid_lo)) & (absolute_idx < int(valid_hi))
    return valid.astype(np.float32)[:, np.newaxis]


def build_window_start_indices(seq_len: int, window_size: int, stride: int) -> List[int]:
    """Build fixed-window start indices and append a tail window when needed."""
    if seq_len <= 0:
        return []
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if seq_len <= window_size:
        return [0]

    starts = list(range(0, seq_len - window_size + 1, stride))
    tail_start = seq_len - window_size
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def compute_balance_weights(
    speeds: List[float],
    genders: List[float],
    window_counts: List[int],
    *,
    bin_width: float = 0.05,
    speed_power: float = 0.5,
    clip_ratio: float = 3.0,
    gender_balance: bool = True,
):
    """Compute a per-trial loss weight that balances gender and up-weights rare speeds.

    Each trial contributes ``window_counts[i]`` windows that all share its mean
    walking speed and gender. Weights are computed on a per-window basis (so a
    trial with more windows carries proportionally more mass) and returned as one
    scalar per trial.

    Returns
    -------
    (trial_weights, stats)
        trial_weights: list[float], one weight per input trial (window-weighted mean ~1).
        stats: dict with gender counts/weights, the speed histogram, per-bin speed
               weights, the config used, and a summary of the final weights.
    """
    n = len(window_counts)
    speeds = np.asarray(speeds, dtype=np.float64)
    genders = np.asarray(genders, dtype=np.float64)
    counts = np.asarray(window_counts, dtype=np.float64)
    total_windows = float(counts.sum())

    if n == 0 or total_windows <= 0:
        return [1.0] * n, {
            "config": {
                "bin_width": bin_width,
                "speed_power": speed_power,
                "clip_ratio": clip_ratio,
                "gender_balance": gender_balance,
            },
            "gender_counts": {"male": 0.0, "female": 0.0, "unknown": 0.0},
            "gender_weights": {"male": 1.0, "female": 1.0, "unknown": 1.0},
            "speed_bins": [],
            "weight_summary": {"n_trials": 0, "min": 1.0, "max": 1.0, "mean": 1.0,
                               "clipped_fraction": 0.0},
        }

    clip_ratio = max(float(clip_ratio), 1.0)

    # --- Gender balancing (per-window mass) ---
    is_male = np.isclose(genders, 1.0)
    is_female = np.isclose(genders, 0.0)
    is_unknown = ~(is_male | is_female)
    n_male = float(counts[is_male].sum())
    n_female = float(counts[is_female].sum())
    n_unknown = float(counts[is_unknown].sum())

    w_male = w_female = w_unknown = 1.0
    gender_w = np.ones(n, dtype=np.float64)
    if gender_balance and n_male > 0 and n_female > 0:
        n_known = n_male + n_female
        w_male = n_known / (2.0 * n_male)
        w_female = n_known / (2.0 * n_female)
        gender_w[is_male] = w_male
        gender_w[is_female] = w_female
        # unknown stays neutral (1.0)

    # --- Speed balancing: w ~ (1 / bin_count)^power, per 0.05 m/s bin ---
    bin_idx = np.floor(speeds / bin_width).astype(np.int64)
    speed_w = np.ones(n, dtype=np.float64)
    bin_window_counts: Dict[int, float] = {}
    for b, c in zip(bin_idx, counts):
        bin_window_counts[int(b)] = bin_window_counts.get(int(b), 0.0) + float(c)
    if speed_power != 0.0:
        per_bin_weight = {b: (1.0 / c) ** speed_power if c > 0 else 1.0
                          for b, c in bin_window_counts.items()}
        speed_w = np.array([per_bin_weight[int(b)] for b in bin_idx], dtype=np.float64)
    else:
        per_bin_weight = {b: 1.0 for b in bin_window_counts}

    # --- Combine, normalize to window-weighted mean 1, clip, renormalize ---
    def _window_mean(w):
        return float((w * counts).sum() / total_windows)

    # Normalize to window-weighted mean 1 first, then clip as the authoritative
    # final bound. (Renormalizing after the clip would re-inflate weights past
    # clip_ratio, defeating the safety knob; the post-clip mean is reported below.)
    combined = gender_w * speed_w
    combined /= max(_window_mean(combined), 1e-12)
    lo, hi = 1.0 / clip_ratio, clip_ratio
    clipped_mask = (combined < lo) | (combined > hi)
    combined = np.clip(combined, lo, hi)

    # --- Build the speed histogram report (contiguous bins) ---
    speed_bins = []
    if bin_window_counts:
        b_min, b_max = min(bin_window_counts), max(bin_window_counts)
        for b in range(b_min, b_max + 1):
            sel = bin_idx == b
            sel_m, sel_f, sel_u = sel & is_male, sel & is_female, sel & is_unknown
            # Weighted ("effective") window mass = sum of per-trial final weights
            # times each trial's window count. This is what the loss actually sees.
            speed_bins.append({
                "left_edge": round(b * bin_width, 6),
                "right_edge": round((b + 1) * bin_width, 6),
                "n_windows": float(bin_window_counts.get(b, 0.0)),
                "n_windows_male": float(counts[sel_m].sum()),
                "n_windows_female": float(counts[sel_f].sum()),
                "n_windows_unknown": float(counts[sel_u].sum()),
                "w_windows": float((counts[sel] * combined[sel]).sum()),
                "w_windows_male": float((counts[sel_m] * combined[sel_m]).sum()),
                "w_windows_female": float((counts[sel_f] * combined[sel_f]).sum()),
                "w_windows_unknown": float((counts[sel_u] * combined[sel_u]).sum()),
                "speed_weight": float(per_bin_weight.get(b, 1.0)),
            })

    stats = {
        "config": {
            "bin_width": bin_width,
            "speed_power": speed_power,
            "clip_ratio": clip_ratio,
            "gender_balance": gender_balance,
        },
        "gender_counts": {"male": n_male, "female": n_female, "unknown": n_unknown},
        "gender_weights": {"male": w_male, "female": w_female, "unknown": w_unknown},
        "speed_bins": speed_bins,
        "weight_summary": {
            "n_trials": int(n),
            "total_windows": total_windows,
            "min": float(combined.min()),
            "max": float(combined.max()),
            "mean": _window_mean(combined),
            "clipped_fraction": float(clipped_mask.mean()),
        },
    }
    return combined.tolist(), stats


def _safe_abs_corr_matrix(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Compute |corr| matrix between columns of ref (T,m) and src (T,n)."""
    ref = np.asarray(ref, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    if ref.ndim != 2 or src.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    t = min(ref.shape[0], src.shape[0])
    if t <= 1:
        return np.zeros((ref.shape[1], src.shape[1]), dtype=np.float32)
    ref_t = ref[:t]
    src_t = src[:t]
    ref_c = ref_t - np.mean(ref_t, axis=0, keepdims=True)
    src_c = src_t - np.mean(src_t, axis=0, keepdims=True)
    ref_std = np.std(ref_c, axis=0, keepdims=True)
    src_std = np.std(src_c, axis=0, keepdims=True)
    ref_std = np.where(ref_std < 1e-8, 1.0, ref_std)
    src_std = np.where(src_std < 1e-8, 1.0, src_std)
    ref_n = ref_c / ref_std
    src_n = src_c / src_std
    corr = (ref_n.T @ src_n) / float(max(t - 1, 1))
    return np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)


def _order_preserving_subset_indices(src: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """
    Select an order-preserving subset of src columns that best matches ref columns.
    Returns indices of length ref_dim, or None if src has fewer columns.
    """
    if src.ndim != 2 or ref.ndim != 2:
        return None
    src_dim = int(src.shape[1])
    ref_dim = int(ref.shape[1])
    if src_dim < ref_dim:
        return None
    if src_dim == ref_dim:
        return np.arange(src_dim, dtype=np.int32)

    score = _safe_abs_corr_matrix(ref, src)  # (ref_dim, src_dim)
    m, n = score.shape
    neg_inf = -1e30
    dp = np.full((m, n), neg_inf, dtype=np.float32)
    prev = np.full((m, n), -1, dtype=np.int32)

    # First row
    j_max0 = n - (m - 1)
    for j in range(0, j_max0):
        dp[0, j] = score[0, j]

    # Remaining rows; O(m*n) with running best over previous row.
    for i in range(1, m):
        running_best = neg_inf
        running_idx = -1
        j_start = i
        j_stop = n - (m - i)  # exclusive upper bound
        for j in range(j_start, j_stop):
            k = j - 1
            if dp[i - 1, k] > running_best:
                running_best = dp[i - 1, k]
                running_idx = k
            if running_idx >= 0:
                dp[i, j] = running_best + score[i, j]
                prev[i, j] = running_idx

    # Best end state in last valid segment.
    end_start = m - 1
    end_stop = n
    end_j = int(np.argmax(dp[m - 1, end_start:end_stop]) + end_start)
    if dp[m - 1, end_j] <= neg_inf / 2:
        return None

    # Backtrack
    idx = np.zeros(m, dtype=np.int32)
    idx[m - 1] = end_j
    for i in range(m - 1, 0, -1):
        pj = int(prev[i, idx[i]])
        if pj < 0:
            return None
        idx[i - 1] = pj
    return idx


def _align_mocap_to_processed_columns(
    source_arr: np.ndarray,
    processed_arr: np.ndarray,
    label: str,
    trial_id: str,
) -> np.ndarray:
    """Align MoCap feature columns to ProcessedData column layout when dims differ."""
    if source_arr.ndim != 2 or processed_arr.ndim != 2:
        return source_arr
    src_dim = int(source_arr.shape[1])
    ref_dim = int(processed_arr.shape[1])
    if src_dim == ref_dim:
        return source_arr
    if src_dim < ref_dim:
        print(
            f"   ⚠️ {label} dims smaller in MoCap than ProcessedData "
            f"({src_dim} < {ref_dim}) for {trial_id}; keeping MoCap as-is"
        )
        return source_arr

    idx = _order_preserving_subset_indices(source_arr, processed_arr)
    if idx is None or idx.shape[0] != ref_dim:
        print(
            f"   ⚠️ Could not align MoCap {label} columns to ProcessedData layout "
            f"for {trial_id}; keeping MoCap as-is"
        )
        return source_arr

    aligned = source_arr[:, idx]
    print(
        f"   🔧 Aligned MoCap {label} columns to ProcessedData layout: "
        f"{src_dim} -> {ref_dim} (trial: {trial_id})"
    )
    return aligned


def _extract_cop_xz_channels(cop_arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert COP array to 4-channel [Rx, Rz, Lx, Lz].

    Accepted input shapes:
    - (T, 6): [Rx, Ry, Rz, Lx, Ly, Lz]
    - (T, 4): already [Rx, Rz, Lx, Lz]
    """
    if cop_arr is None:
        return None
    cop_arr = np.asarray(cop_arr)
    if cop_arr.ndim != 2:
        return None
    if cop_arr.shape[1] >= 6:
        return cop_arr[:, [0, 2, 3, 5]]
    if cop_arr.shape[1] == 4:
        return cop_arr
    return None


def _cop_target_filename(use_grf_norm_cop: bool) -> str:
    return (
        "COP_CalcFrame_GroundAligned_GRFNorm.npy"
        if use_grf_norm_cop
        else "COP_CalcFrame_GroundAligned.npy"
    )


def _build_grf_norm_cop_target(
    cop_ground_aligned: np.ndarray,
    grf: np.ndarray,
    mass: np.ndarray,
    height: np.ndarray,
) -> Optional[np.ndarray]:
    """Build (COP / height) * (|GRF| / body_weight) when the file is absent."""
    cop = np.asarray(cop_ground_aligned, dtype=np.float32)
    grf_arr = np.asarray(grf, dtype=np.float32)
    if cop.ndim != 2 or cop.shape[1] < 6 or grf_arr.ndim != 2 or grf_arr.shape[1] < 6:
        return None
    T = int(cop.shape[0])
    if grf_arr.shape[0] != T:
        return None

    mass_arr = np.asarray(mass, dtype=np.float32).reshape(-1)
    height_arr = np.asarray(height, dtype=np.float32).reshape(-1)
    if mass_arr.size == 1:
        mass_arr = np.full(T, float(mass_arr[0]), dtype=np.float32)
    if height_arr.size == 1:
        height_arr = np.full(T, float(height_arr[0]), dtype=np.float32)
    if mass_arr.size != T or height_arr.size != T:
        return None
    body_weight = mass_arr * np.float32(9.8067)
    if np.any(~np.isfinite(body_weight)) or np.any(body_weight <= 0.0):
        return None
    if np.any(~np.isfinite(height_arr)) or np.any(height_arr <= 0.0):
        return None

    grf_mag_r = np.linalg.norm(grf_arr[:, 0:3], axis=1)
    grf_mag_l = np.linalg.norm(grf_arr[:, 3:6], axis=1)
    out = cop[:, :6] / height_arr[:, None]
    out[:, 0:3] *= (grf_mag_r / body_weight)[:, None]
    out[:, 3:6] *= (grf_mag_l / body_weight)[:, None]
    return out.astype(np.float32, copy=False)


def _opensim_to_mujoco_6col(arr_6: np.ndarray) -> np.ndarray:
    """Convert per-foot [X,Y,Z] blocks to MuJoCo [X,-Z,Y] for 6-column arrays."""
    arr_6 = np.asarray(arr_6, dtype=np.float32)
    out = arr_6.copy()
    out[:, 0] = arr_6[:, 0]
    out[:, 1] = -arr_6[:, 2]
    out[:, 2] = arr_6[:, 1]
    out[:, 3] = arr_6[:, 3]
    out[:, 4] = -arr_6[:, 5]
    out[:, 5] = arr_6[:, 4]
    return out

def load_single_trial(
    path: Path,
    use_secondary: bool = False,
    trim_cop: bool = False,
    deviation_learning: bool = False,
    opencap_val: bool = False,
    input_source: str = "processed",
    use_noised: bool = False,
    noised_gt: bool = False,
    use_grf_norm_cop: bool = False,
    use_grf_nofilt: Optional[bool] = None,
    use_os_filtering: bool = False,
    use_opensim_id_gt: bool = False,
    use_recalculated_opensim_id_gt: bool = False,
    force_gt_grf_contribution: bool = False,
    grf_grm_from_processed: bool = False,
    subtract_ankle_height_knee_vecs: bool = False,
    allow_missing_noised: bool = False,
) -> Optional[Dict]:
    """
    Load a single trial's data with robust path handling and physiological normalization.
    
    Physiological Normalization:
    - COP: Ground-aligned calc frame [Rx,Rz,Lx,Lz], divided by subject height (m)
      When use_grf_norm_cop=True, the COP target is instead
      (COP / height) * (|GRF| / body_weight), loaded from
      COP_CalcFrame_GroundAligned_GRFNorm.npy.
    - GRF: Divided by subject mass (kg)
    - Moments: Divided by subject mass (kg)
    
    Returns a dictionary containing:
    - pos/vel/acc: Kinematic inputs
    - pelvis_rot: 6-dim pelvis orientation features (first two columns of R_total), shape (T, 6)
    - cop/grf/moments: Physiologically normalized targets
    - _raw fields: Raw physical units (m, N, Nm) for plotting/physics
    - qfrc_grf_contribution: Raw torque contribution from external forces (Nm)
    - When `noised_gt=True`, COP / calc-frame GT rotations / qfrc_inverse are
      sourced from the `_noised.npy` bundle when available for ProcessedData GT.
    - `allow_missing_noised=True` lets a trial that carries *no* `_noised.npy`
      bundle at all fall back to its clean files instead of being skipped. A
      partially-written bundle still fails strictly, because that indicates a
      broken noising run rather than a dataset that was never noised.
    """
    trial_root = resolve_trial_root_from_path(Path(path))
    trial_id = str(trial_root)
    direct_dir = direct_processed_dir(trial_root)
    opencap_video_dir = video_processed_dir(trial_root)
    mocap_dir = mocap_processed_dir(trial_root)
    uses_direct_processed_layout = bool(direct_dir.exists() and not opencap_val)
    processed_dir = direct_dir if uses_direct_processed_layout else opencap_video_dir
    if opencap_val and not mocap_dir.exists():
        subject_name = trial_root.parent.name
        trial_name = trial_root.name
        print(
            f"🛑 FORCE STOP: OpenCap validation requested but MoCap/ProcessedData is missing "
            f"for {subject_name}/{trial_name}"
        )
        raise FileNotFoundError(f"Missing MoCap/ProcessedData directory for OpenCap validation: {trial_id}")

    try:
        use_mocap_gt = bool(opencap_val and mocap_dir.exists())

        # Choose input source (video/OpenCap vs marker/MoCap).
        input_source_norm = normalize_input_source_name(input_source)
        if input_source_norm == "mocap" and mocap_dir.exists():
            load_path = mocap_dir
            input_source_label = "MoCap"
        else:
            load_path = processed_dir
            input_source_label = "ProcessedData" if uses_direct_processed_layout else "Video"
        input_use_noised = bool(use_noised and load_path == processed_dir)

        if not load_path.exists():
            return None

        ground_truth_base_dir = (
            mocap_dir
            if (use_mocap_gt or (input_source_norm == "mocap" and mocap_dir.exists()))
            else processed_dir
        )
        if not ground_truth_base_dir.exists():
            return None
        ground_truth_source_label = "MoCap" if ground_truth_base_dir == mocap_dir else "Video"
        ground_truth_use_noised = bool(
            noised_gt and ground_truth_base_dir == processed_dir and not use_mocap_gt
        )

        # Datasets that were never noised (e.g. the OpenCap validation cohort) carry
        # no `_noised.npy` files at all. Without this fallback every one of their
        # trials is silently dropped whenever use_noised/noised_gt is on.
        # Deliberately all-or-nothing: a *partial* bundle means the noising run
        # broke, and quietly mixing noised and clean sources there would corrupt
        # training in a way that is very hard to notice later.
        if allow_missing_noised and (input_use_noised or ground_truth_use_noised):
            def _lacks_noised_bundle(directory: Path) -> bool:
                return not any(directory.glob(f"*{NOISED_FILE_SUFFIX}.npy"))

            fell_back = []
            if input_use_noised and _lacks_noised_bundle(load_path):
                input_use_noised = False
                fell_back.append(f"inputs<-{input_source_label}")
            if ground_truth_use_noised and _lacks_noised_bundle(ground_truth_base_dir):
                ground_truth_use_noised = False
                fell_back.append(f"gt<-{ground_truth_source_label}")
            if fell_back and not _quiet_loader_logs():
                print(
                    f"   ↩︎ No noised bundle; using clean files ({', '.join(fell_back)}): {trial_id}",
                    flush=True,
                )

        def _os_name(filename: str) -> str:
            # When OpenSim filtering is requested, redirect each velocity/acceleration-derived
            # file to its _OSfilt variant (produced by ProcessData --os-filtering). Positions
            # (pos_inputs/pos_mjx) and force/COP files are unaffected by the filtering method.
            if use_os_filtering and filename in OS_FILTER_FILES:
                return filename[:-4] + "_OSfilt.npy"
            return filename

        def load_npy_robust(filename, allow_pickle=False, base_dir: Optional[Path] = None):
            base = load_path if base_dir is None else base_dir
            p = base / filename
            if p.exists():
                return np.load(p, allow_pickle=allow_pickle)
            return None

        def load_input_file(filename: str, allow_pickle: bool = False, base_dir: Optional[Path] = None):
            base = load_path if base_dir is None else base_dir
            os_fn = _os_name(filename)
            if os_fn != filename:  # OS-filtered variant has no _noised counterpart
                return load_npy_robust(os_fn, allow_pickle=allow_pickle, base_dir=base)
            if input_use_noised:
                arr = load_npy_robust(_with_file_suffix(filename), allow_pickle=allow_pickle, base_dir=base)
                if arr is not None:
                    return arr
                return None
            return load_npy_robust(filename, allow_pickle=allow_pickle, base_dir=base)

        def load_input_candidates(candidates: List[str], allow_pickle: bool = False, base_dir: Optional[Path] = None):
            base = load_path if base_dir is None else base_dir
            if use_os_filtering and any(c in OS_FILTER_FILES for c in candidates):
                for name in candidates:
                    arr = load_npy_robust(_os_name(name), allow_pickle=allow_pickle, base_dir=base)
                    if arr is not None:
                        return arr
                return None
            if input_use_noised:
                for name in candidates:
                    arr = load_npy_robust(_with_file_suffix(name), allow_pickle=allow_pickle, base_dir=base)
                    if arr is not None:
                        return arr
                return None
            for name in candidates:
                arr = load_npy_robust(name, allow_pickle=allow_pickle, base_dir=base)
                if arr is not None:
                    return arr
            return None

        def load_gt_file(
            filename: str,
            allow_pickle: bool = False,
            base_dir: Optional[Path] = None,
            use_noised_variant: bool = False,
        ):
            base = ground_truth_base_dir if base_dir is None else base_dir
            os_fn = _os_name(filename)
            if os_fn != filename:  # OS-filtered GT variant has no _noised counterpart
                return load_npy_robust(os_fn, allow_pickle=allow_pickle, base_dir=base)
            if use_noised_variant and ground_truth_use_noised:
                arr = load_npy_robust(_with_file_suffix(filename), allow_pickle=allow_pickle, base_dir=base)
                if arr is not None:
                    return arr
                return None
            return load_npy_robust(filename, allow_pickle=allow_pickle, base_dir=base)

        def load_gt_candidates(
            candidates: List[str],
            allow_pickle: bool = False,
            base_dir: Optional[Path] = None,
            use_noised_variant: bool = False,
        ):
            base = ground_truth_base_dir if base_dir is None else base_dir
            if use_noised_variant and ground_truth_use_noised:
                for name in candidates:
                    arr = load_npy_robust(_with_file_suffix(name), allow_pickle=allow_pickle, base_dir=base)
                    if arr is not None:
                        return arr
                return None
            for name in candidates:
                arr = load_npy_robust(name, allow_pickle=allow_pickle, base_dir=base)
                if arr is not None:
                    return arr
            return None

        # Temporal inputs come from selected input source (ProcessedData or MoCap).
        pos = load_input_file("pos_inputs.npy")
        vel = load_input_file("vel_inputs.npy")
        acc = load_input_file("acc_inputs.npy")
        pos_gt = load_gt_file("pos_inputs.npy", use_noised_variant=False)
        vel_gt = load_gt_file("vel_inputs.npy", use_noised_variant=False)
        acc_gt = load_gt_file("acc_inputs.npy", use_noised_variant=False)
        qpos_mjx_input = load_input_file("pos_mjx.npy")
        qvel_mjx_input = load_input_file("qvel_mjx.npy")
        qacc_mjx_input = load_input_file("qacc_mjx.npy")

        if pos is None or vel is None or acc is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised temporal inputs): {trial_id}")
            return None

        # If loading MoCap inputs for OpenCap comparison, align kinematic column
        # layout to ProcessedData when ProcessedData has a reduced feature mask
        # (e.g., 16/19 instead of 18/21).
        if input_source_norm == "mocap":
            if processed_dir.exists():
                try:
                    pos_ref = np.load(processed_dir / "pos_inputs.npy") if (processed_dir / "pos_inputs.npy").exists() else None
                    vel_ref = np.load(processed_dir / "vel_inputs.npy") if (processed_dir / "vel_inputs.npy").exists() else None
                    acc_ref = np.load(processed_dir / "acc_inputs.npy") if (processed_dir / "acc_inputs.npy").exists() else None
                except Exception:
                    pos_ref = vel_ref = acc_ref = None

                if pos_ref is not None:
                    pos = _align_mocap_to_processed_columns(pos, pos_ref, "pos_inputs", trial_id)
                if vel_ref is not None:
                    vel = _align_mocap_to_processed_columns(vel, vel_ref, "vel_inputs", trial_id)
                if acc_ref is not None:
                    acc = _align_mocap_to_processed_columns(acc, acc_ref, "acc_inputs", trial_id)

        # Statics are sourced from the selected input source (with subject-level fallbacks).
        height = load_npy_robust("Height_m.npy")
        mass = load_npy_robust("Mass_kg.npy")
        if height is None or mass is None:
            return None

        # PatientSize is often stored at subject-level (one level above trials).
        patient_size = load_npy_robust("PatientSize.npy")
        if patient_size is None:
            current = trial_root
            for _ in range(4):
                candidate = current / "PatientSize.npy"
                if candidate.exists():
                    patient_size = np.load(candidate)
                    break
                current = current.parent
        if patient_size is None:
            patient_size_vec = np.zeros(4, dtype=np.float32)
        else:
            patient_size_arr = np.asarray(patient_size, dtype=np.float32).reshape(-1)
            patient_size_vec = np.zeros(4, dtype=np.float32)
            take_n = min(4, patient_size_arr.shape[0])
            patient_size_vec[:take_n] = patient_size_arr[:take_n]

        # forwardVel is trial-level in ProcessedData and may be a vector over time.
        forward_vel = load_input_file("forwardVel.npy")
        if forward_vel is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised forward velocity): {trial_id}")
                return None
            forward_vel_scalar = 0.0
        else:
            forward_vel_arr = np.asarray(forward_vel, dtype=np.float32).reshape(-1)
            if forward_vel_arr.size == 0:
                forward_vel_scalar = 0.0
            else:
                forward_vel_scalar = float(np.nanmean(forward_vel_arr))
                if not np.isfinite(forward_vel_scalar):
                    forward_vel_scalar = 0.0
        
        # Load Patient Info (BiologicalSex)
        gender_val = 0.5 # Default fallback
        current_md_path = load_path
        found_md = False
        for _ in range(5):
            md_path = current_md_path / "Patient_MD.json"
            if md_path.exists():
                try:
                    with open(md_path, 'r') as f:
                        md = json.load(f)
                        sex = md.get("BiologicalSex", "").lower()
                        if sex == "male": gender_val = 1.0
                        elif sex == "female": gender_val = 0.0
                        found_md = True
                except: pass
                break
            current_md_path = current_md_path.parent

        com_l = load_input_file("COM_l.npy")
        com_r = load_input_file("COM_r.npy")
        if com_l is None or com_r is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised COM inputs): {trial_id}")
                return None
            if com_l is None:
                com_l = np.zeros((len(pos), 3), dtype=np.float32)
            if com_r is None:
                com_r = np.zeros((len(pos), 3), dtype=np.float32)
        contactBoolean = load_gt_file("contactBoolean.npy")
        if contactBoolean is None:
            contactBoolean = np.zeros((len(pos), 2), dtype=np.float32)
        else:
            contactBoolean = np.asarray(contactBoolean, dtype=np.float32)
            if contactBoolean.ndim == 1:
                contactBoolean = contactBoolean[:, np.newaxis]
            if contactBoolean.shape[1] == 1:
                contactBoolean = np.repeat(contactBoolean, 2, axis=1)
            elif contactBoolean.shape[1] > 2:
                contactBoolean = contactBoolean[:, :2]

        qpos_mjx_gt = load_gt_file("pos_mjx.npy", use_noised_variant=False)
        qvel_mjx_gt = load_gt_file("qvel_mjx.npy", use_noised_variant=False)
        qacc_mjx_gt = load_gt_file("qacc_mjx.npy", use_noised_variant=False)
        qfrc_inverse_gt_clean = load_gt_file("qfrc_inverse.npy", use_noised_variant=False)
        if qfrc_inverse_gt_clean is None:
            print(f"   ⚠️ Skipping trial (missing clean {ground_truth_source_label} qfrc_inverse): {trial_id}")
            return None
        qfrc_inverse_gt_clean = coerce_independent_dof_width(
            qfrc_inverse_gt_clean,
            label="clean qfrc_inverse GT",
            trial_id=trial_id,
        )

        # Load Pelvis Rotation Matrix — expected shape (T, 6), already-extracted features
        pelvis_rot = load_input_file("pelvis_rot_matrix.npy")
        if pelvis_rot is None:
            print(f"   ⚠️ Skipping trial (missing {'noised ' if input_use_noised else ''}{input_source_label} pelvis rotation): {trial_id}")
            return None
        # Enforce (T, 6) shape — ProcessData.py saves it pre-extracted
        if pelvis_rot.ndim == 3:
            # Legacy (T,3,3): extract first two columns and flatten
            pelvis_rot = pelvis_rot[:, :, :2].reshape(len(pelvis_rot), 6)

        # Prediction-side kinematics bundle. When UseNoised is enabled, the
        # temporal inputs, Jacobians, and calc-frame rotations come from the
        # suffixed `_noised` files.
        rot_w_to_ga = load_input_file("WorldToGroundAlignedCalcnRotation.npy")
        if rot_w_to_ga is None:
            print(f"   ⚠️ Skipping trial (missing {'noised ' if input_use_noised else ''}{input_source_label} rotation bundle): {trial_id}")
            return None

        # Validate rotation array shape: (T,2,3,3)
        rot_w_to_ga = np.asarray(rot_w_to_ga, dtype=np.float32)
        if rot_w_to_ga.ndim != 4 or rot_w_to_ga.shape[1:] != (2, 3, 3):
            print(f"   ⚠️ Skipping trial (invalid {'noised ' if input_use_noised else ''}{input_source_label} WorldToGroundAlignedCalcnRotation shape): {trial_id}")
            return None

        ankle_pos_r = load_input_file("ankle_pos_r.npy")
        ankle_pos_l = load_input_file("ankle_pos_l.npy")
        knee_pos_r = load_input_file("knee_pos_r.npy")
        knee_pos_l = load_input_file("knee_pos_l.npy")
        if ankle_pos_r is None or ankle_pos_l is None or knee_pos_r is None or knee_pos_l is None:
            suffix = "noised " if input_use_noised else ""
            print(f"   ⚠️ Skipping trial (missing {suffix}{input_source_label} ankle/knee global positions): {trial_id}")
            return None
        ankle_pos = np.stack(
            [np.asarray(ankle_pos_r, dtype=np.float32), np.asarray(ankle_pos_l, dtype=np.float32)],
            axis=1,
        )
        knee_pos = np.stack(
            [np.asarray(knee_pos_r, dtype=np.float32), np.asarray(knee_pos_l, dtype=np.float32)],
            axis=1,
        )
        if ankle_pos.ndim != 3 or ankle_pos.shape[1:] != (2, 3) or knee_pos.ndim != 3 or knee_pos.shape[1:] != (2, 3):
            print(f"   ⚠️ Skipping trial (invalid {input_source_label} ankle/knee global position shape): {trial_id}")
            return None

        cop_target_filename = _cop_target_filename(use_grf_norm_cop)
        gt_cop_ground_aligned_full = load_gt_file(
            "COP_CalcFrame_GroundAligned.npy",
            use_noised_variant=True,
        )
        gt_cop_target_full = load_gt_file(
            cop_target_filename,
            use_noised_variant=True,
        )
        gt_rot_w_to_ga_physics = load_gt_file(
            "WorldToGroundAlignedCalcnRotation.npy",
            use_noised_variant=True,
        )
        gt_rot_w_to_ga = load_gt_file("WorldToGroundAlignedCalcnRotation.npy", use_noised_variant=False)
        if gt_rot_w_to_ga is None:
            print(f"   ⚠️ Skipping trial (missing clean {ground_truth_source_label} WorldToGroundAlignedCalcnRotation): {trial_id}")
            return None
        gt_ankle_pos_r = load_gt_file("ankle_pos_r.npy", use_noised_variant=True)
        gt_ankle_pos_l = load_gt_file("ankle_pos_l.npy", use_noised_variant=True)
        gt_knee_pos_r = load_gt_file("knee_pos_r.npy", use_noised_variant=True)
        gt_knee_pos_l = load_gt_file("knee_pos_l.npy", use_noised_variant=True)
        if gt_ankle_pos_r is None or gt_ankle_pos_l is None or gt_knee_pos_r is None or gt_knee_pos_l is None:
            # Legacy OpenCap exports may omit the MoCap knee-position files. The
            # video-based LOSO does not use those MoCap positions for reconstruction;
            # retain a complete auxiliary bundle by falling back to the matching
            # video/input positions already loaded above.
            gt_ankle_pos_r = ankle_pos_r if gt_ankle_pos_r is None else gt_ankle_pos_r
            gt_ankle_pos_l = ankle_pos_l if gt_ankle_pos_l is None else gt_ankle_pos_l
            gt_knee_pos_r = knee_pos_r if gt_knee_pos_r is None else gt_knee_pos_r
            gt_knee_pos_l = knee_pos_l if gt_knee_pos_l is None else gt_knee_pos_l
            print(
                f"   ⚠️ Missing some {ground_truth_source_label} ankle/knee positions; "
                f"using {input_source_label} positions for the absent auxiliary files: {trial_id}"
            )
        gt_ankle_pos = np.stack(
            [np.asarray(gt_ankle_pos_r, dtype=np.float32), np.asarray(gt_ankle_pos_l, dtype=np.float32)],
            axis=1,
        )
        gt_knee_pos = np.stack(
            [np.asarray(gt_knee_pos_r, dtype=np.float32), np.asarray(gt_knee_pos_l, dtype=np.float32)],
            axis=1,
        )
        if gt_ankle_pos.ndim != 3 or gt_ankle_pos.shape[1:] != (2, 3) or gt_knee_pos.ndim != 3 or gt_knee_pos.shape[1:] != (2, 3):
            print(f"   ⚠️ Skipping trial (invalid {ground_truth_source_label} ankle/knee global position shape): {trial_id}")
            return None
        # GRF / GRM (free moment) are force-plate measurements, not kinematic. When
        # grf_grm_from_processed is set (e.g. a MoCap-input LOSO), source them from
        # ProcessedData/ so the force target stays the OpenCap-processed signal while
        # only the kinematics come from MoCap. MoCap and ProcessedData are frame-aligned.
        grf_grm_base_dir = processed_dir if (grf_grm_from_processed and processed_dir.exists()) else ground_truth_base_dir
        grf_nofilt_path = grf_grm_base_dir / "GRF_NoFilt_Trimmed.npy"
        grf_cleaned_path = grf_grm_base_dir / "GRF_Cleaned.npy"
        if use_grf_nofilt is True:
            grf_source_filename = "GRF_NoFilt_Trimmed.npy"
            if not grf_nofilt_path.exists():
                print(
                    f"   ⚠️ Skipping trial (requested GRF_NoFilt_Trimmed.npy but it is missing): {trial_id}"
                )
                return None
        elif use_grf_nofilt is False:
            grf_source_filename = "GRF_Cleaned.npy"
        else:
            grf_source_filename = (
                "GRF_NoFilt_Trimmed.npy" if grf_nofilt_path.exists() else "GRF_Cleaned.npy"
            )

        grf_source_path = grf_grm_base_dir / grf_source_filename
        grf = load_gt_file(grf_source_filename, base_dir=grf_grm_base_dir)
        if grf is None:
            if grf_source_filename != "GRF_Cleaned.npy":
                print(
                    f"   ⚠️ Skipping trial (missing {grf_grm_base_dir.name} {grf_source_filename}): {trial_id}"
                )
                return None
            grf_filtered = load_gt_file("GRF_Filtered.npy", base_dir=grf_grm_base_dir)
            if grf_filtered is not None:
                grf = _opensim_to_mujoco_6col(grf_filtered)
                grf_source_filename = "GRF_Filtered.npy"
                grf_source_path = grf_grm_base_dir / grf_source_filename

        moments = load_gt_file("Moment_Cleaned.npy", base_dir=grf_grm_base_dir)
        if moments is None:
            grm_filtered = load_gt_file("GRM_Filtered.npy", base_dir=grf_grm_base_dir)
            if grm_filtered is not None:
                moments = _opensim_to_mujoco_6col(grm_filtered)

        if use_grf_norm_cop and gt_cop_target_full is None and gt_cop_ground_aligned_full is not None and grf is not None:
            gt_mass = load_gt_file("Mass_kg.npy")
            gt_height = load_gt_file("Height_m.npy")
            gt_cop_target_full = _build_grf_norm_cop_target(
                gt_cop_ground_aligned_full,
                grf,
                gt_mass if gt_mass is not None else mass,
                gt_height if gt_height is not None else height,
            )
            if gt_cop_target_full is not None:
                print(
                    f"   ℹ️ Built missing {ground_truth_source_label} {cop_target_filename} in memory: {trial_id}"
                )

        if gt_cop_ground_aligned_full is None or gt_cop_target_full is None or gt_rot_w_to_ga_physics is None or grf is None or moments is None:
            print(f"   ⚠️ Skipping trial (missing {ground_truth_source_label} ground-truth bundle): {trial_id}")
            return None

        cop = _extract_cop_xz_channels(gt_cop_target_full)
        if cop is None:
            print(f"   ⚠️ Skipping trial (invalid {ground_truth_source_label} {cop_target_filename} shape): {trial_id}")
            return None
        cop_length_units = _extract_cop_xz_channels(gt_cop_ground_aligned_full)
        if cop_length_units is None:
            print(f"   ⚠️ Skipping trial (invalid {ground_truth_source_label} COP_CalcFrame_GroundAligned shape): {trial_id}")
            return None

        gt_rot_w_to_ga_physics = np.asarray(gt_rot_w_to_ga_physics, dtype=np.float32)
        if gt_rot_w_to_ga_physics.ndim != 4 or gt_rot_w_to_ga_physics.shape[1:] != (2, 3, 3):
            print(f"   ⚠️ Skipping trial (invalid {ground_truth_source_label} WorldToGroundAlignedCalcnRotation shape): {trial_id}")
            return None
        gt_rot_w_to_ga = np.asarray(gt_rot_w_to_ga, dtype=np.float32)
        if gt_rot_w_to_ga.ndim != 4 or gt_rot_w_to_ga.shape[1:] != (2, 3, 3):
            print(f"   ⚠️ Skipping trial (invalid clean {ground_truth_source_label} WorldToGroundAlignedCalcnRotation shape): {trial_id}")
            return None

        # Reconstructed curves are always loaded as temporal input features.
        # For moment reconstruction, only vertical components (Rz, Lz) are fed to the model input.
        grf_recon = load_npy_robust("GRF_average_reconstructed.npy")
        cop_recon = load_npy_robust("COP_CalcFrame_GroundAligned_average_reconstructed.npy")
        if cop_recon is None:
            cop_recon = load_npy_robust("COP_average_reconstructed.npy")
        moment_recon = load_npy_robust("Moment_average_reconstructed.npy")
        if grf_recon is None:
            grf_recon = np.zeros_like(grf)
        if cop_recon is None:
            cop_recon = np.zeros_like(cop)
        else:
            cop_recon_xz = _extract_cop_xz_channels(cop_recon)
            if cop_recon_xz is None:
                cop_recon = np.zeros_like(cop)
            else:
                cop_recon = cop_recon_xz
        if moment_recon is None:
            moment_recon = np.zeros_like(moments)
        
        # COM Acceleration (in ProcessedData/)
        com_accel = load_input_file("COM_Acc_Global.npy")
        if com_accel is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised COM acceleration): {trial_id}")
                return None
            com_accel = np.zeros((len(pos), 3), dtype=np.float32)

        # Prediction-side Jacobian bundle.
        # Jacobian must be consistent with the input kinematics: the Jacobian is a
        # kinematic quantity (body positions/orientations → generalized force mapping),
        # so it must be derived from the same skeleton state as pos/vel/acc.
        # For MoCap inputs, prefer MoCap/Jacobian.npy; only fall back to
        # ProcessedData/ or Motion/ if MoCap doesn't carry it.
        jacobian_base_dir = load_path
        if input_source_norm == "mocap" and not (load_path / "Jacobian.npy").exists():
            motion_dir = trial_root / "MoCap" / "Motion"
            if (processed_dir / "Jacobian.npy").exists():
                jacobian_base_dir = processed_dir
                _loader_log(
                    "   🎯 MoCap/Jacobian.npy not found; falling back to ProcessedData Jacobian"
                )
            elif (motion_dir / "Jacobian.npy").exists():
                jacobian_base_dir = motion_dir
                _loader_log(
                    "   🎯 MoCap/Jacobian.npy not found; falling back to Motion-folder Jacobian"
                )
        elif input_source_norm == "mocap":
            _loader_log("   🎯 Using MoCap/Jacobian.npy (kinematically consistent with MoCap inputs)")
        jacobian_data = load_input_candidates(
            ["Jacobian.npy", "Jacobian_Data.npy"],
            allow_pickle=True,
            base_dir=jacobian_base_dir,
        )
        if jacobian_data is None:
            jacobian_source_label = (
                str(jacobian_base_dir.name)
                if input_source_norm == "mocap" and jacobian_base_dir != load_path
                else input_source_label
            )
            print(
                f"   ⚠️ Skipping trial (missing {'noised ' if input_use_noised else ''}"
                f"{jacobian_source_label} Jacobian): {trial_id}"
            )
            return None
        jacobian_data = jacobian_data.item()
        jacp = coerce_independent_dof_width(
            np.array(jacobian_data['jacp']),
            label="input Jacobian.jacp",
            trial_id=trial_id,
        )
        jacr = coerce_independent_dof_width(
            np.array(jacobian_data['jacr']),
            label="input Jacobian.jacr",
            trial_id=trial_id,
        )
        body_ids = np.array(jacobian_data['body_ids'])

        gt_jacobian_data = load_gt_candidates(["Jacobian.npy", "Jacobian_Data.npy"], allow_pickle=True)
        if gt_jacobian_data is None:
            print(f"   ⚠️ Skipping trial (missing {ground_truth_source_label} Jacobian GT): {trial_id}")
            return None
        gt_jacobian_data = gt_jacobian_data.item()
        gt_jacp = coerce_independent_dof_width(
            np.array(gt_jacobian_data['jacp']),
            label="GT Jacobian.jacp",
            trial_id=trial_id,
        )
        gt_jacr = coerce_independent_dof_width(
            np.array(gt_jacobian_data['jacr']),
            label="GT Jacobian.jacr",
            trial_id=trial_id,
        )

        ankle_heights = load_input_file("ankle_heights.npy")
        if ankle_heights is None:
            print(f"   ⚠️ Skipping trial (missing {'noised ' if input_use_noised else ''}{input_source_label} ankle heights): {trial_id}")
            return None

        # New foot geometry features
        foot_progression_angle = load_input_candidates(["Foot_ProgressionAngle.npy", "FootProgressionAngle.npy"])
        if foot_progression_angle is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised foot progression angle): {trial_id}")
                return None
            foot_progression_angle = np.zeros((len(pos), 2), dtype=np.float32)
        else:
            foot_progression_angle = np.asarray(foot_progression_angle, dtype=np.float32)
            if foot_progression_angle.ndim == 1:
                foot_progression_angle = foot_progression_angle[:, np.newaxis]
            if foot_progression_angle.shape[1] < 2:
                foot_progression_angle = np.tile(foot_progression_angle, (1, 2))[:, :2]

        calcn_to_floor_angle = load_input_file("CalcnToFloor_AngleDeg.npy")
        if calcn_to_floor_angle is None:
            if input_use_noised:
                print(f"   ⚠️ Skipping trial (missing noised calcaneus-floor angle): {trial_id}")
                return None
            calcn_to_floor_angle = np.zeros((len(pos), 2), dtype=np.float32)
        else:
            calcn_to_floor_angle = np.asarray(calcn_to_floor_angle, dtype=np.float32)
            if calcn_to_floor_angle.ndim == 1:
                calcn_to_floor_angle = calcn_to_floor_angle[:, np.newaxis]
            if calcn_to_floor_angle.shape[1] < 2:
                calcn_to_floor_angle = np.tile(calcn_to_floor_angle, (1, 2))[:, :2]

        # Optional fields for inference comparison
        id_gt_mjx = load_gt_file("ID_GT_MJX.npy")
        if id_gt_mjx is not None:
            id_gt_mjx = coerce_independent_dof_width(
                id_gt_mjx,
                label="ID_GT_MJX",
                trial_id=trial_id,
            )
        opensim_id_bundle = None
        if use_opensim_id_gt:
            opensim_id_bundle = load_aligned_opensim_id_target(
                trial_root,
                target_len=len(pos),
            )
            id_gt_mjx = np.asarray(opensim_id_bundle["id"], dtype=np.float32)
            _loader_log(
                "   🎯 Loading strictly aligned OpenSim ID GT: "
                f"{Path(opensim_id_bundle['source_path']).name}"
            )
        recalculated_opensim_id_gt = None
        qfrc_inverse = load_input_file("qfrc_inverse.npy")
        if qfrc_inverse is None:
            print(f"   ⚠️ Skipping trial (missing {'noised ' if input_use_noised else ''}{input_source_label} qfrc_inverse): {trial_id}")
            return None
        qfrc_inverse = coerce_independent_dof_width(
            qfrc_inverse,
            label="input qfrc_inverse",
            trial_id=trial_id,
        )
        qfrc_inverse_gt_processed = load_gt_file(
            "qfrc_inverse.npy",
            base_dir=processed_dir,
            use_noised_variant=True,
        )
        qfrc_inverse_processed = (
            coerce_independent_dof_width(
                qfrc_inverse_gt_processed,
                label="processed qfrc_inverse",
                trial_id=trial_id,
            )
            if qfrc_inverse_gt_processed is not None
            else (
                None
                if ground_truth_use_noised
                else (
                    qfrc_inverse.copy()
                    if qfrc_inverse is not None and input_source_label == "Video"
                    else load_input_file("qfrc_inverse.npy", base_dir=processed_dir)
                )
            )
        )
        if qfrc_inverse_processed is not None:
            qfrc_inverse_processed = coerce_independent_dof_width(
                qfrc_inverse_processed,
                label="processed qfrc_inverse",
                trial_id=trial_id,
            )
        qfrc_inverse_mocap = (
            qfrc_inverse.copy()
            if qfrc_inverse is not None and input_source_label == "MoCap"
            else (
                load_npy_robust("qfrc_inverse.npy", base_dir=mocap_dir)
                if mocap_dir.exists()
                else None
            )
        )
        if qfrc_inverse_mocap is not None:
            qfrc_inverse_mocap = coerce_independent_dof_width(
                qfrc_inverse_mocap,
                label="mocap qfrc_inverse",
                trial_id=trial_id,
            )

        # MoCap redirection for ID Ground Truth
        if use_mocap_gt:
            mocap_id_path = mocap_dir / "ID_GT_MJX.npy"
            if mocap_id_path.exists():
                id_gt_mjx = coerce_independent_dof_width(
                    np.load(mocap_id_path),
                    label="MoCap ID_GT_MJX",
                    trial_id=trial_id,
                )
                _loader_log(f"   🎯 Loading MoCap ID GT: {mocap_id_path.name}")
            recalculated_candidates = [
                mocap_dir / "OpenSim_ID_recalculated.npy",
                mocap_dir / "ID_GT_OpenSim_recalculated.npy",
                mocap_dir / "inverse_dynamics_recalculated.npy",
                trial_root / "OpenSimResults_recalculated" / "OpenSim_ID_recalculated.npy",
            ]
            for candidate in recalculated_candidates:
                    if candidate.exists():
                        recalculated_opensim_id_gt = coerce_independent_dof_width(
                            np.load(candidate),
                            label="recalculated OpenSim ID GT",
                            trial_id=trial_id,
                        )
                        _loader_log(f"   🎯 Loading recalculated OpenSim ID GT: {candidate.name}")
                        break
            if use_recalculated_opensim_id_gt and recalculated_opensim_id_gt is None:
                raise FileNotFoundError(
                    "Recalculated OpenSim ID GT requested but no supported file was found. "
                    f"Checked: {', '.join(str(p) for p in recalculated_candidates)}"
                )

            # Keep both base-ID sources for OpenCap comparisons:
            # - ProcessedData qfrc_inverse for OpenCap-input predictions
            # - MoCap qfrc_inverse for MoCap-input predictions
            if input_source_norm != "mocap":
                qfrc_inverse = qfrc_inverse_processed
            elif qfrc_inverse_mocap is None and qfrc_inverse is not None:
                qfrc_inverse_mocap = np.asarray(qfrc_inverse, dtype=np.float32).copy()

        # Sync lengths. Reconstructed helper curves should only constrain trial
        # length when deviation learning actually uses them as model inputs.
        arrays_to_sync = [
            pos, vel, acc, cop, cop_length_units, gt_cop_ground_aligned_full, gt_cop_target_full, grf, moments,
            jacp, jacr, gt_jacp, gt_jacr, ankle_heights, pelvis_rot, com_accel,
            contactBoolean, com_l, com_r, rot_w_to_ga, gt_rot_w_to_ga_physics,
            gt_rot_w_to_ga, ankle_pos, knee_pos, gt_ankle_pos, gt_knee_pos, qfrc_inverse,
            foot_progression_angle, calcn_to_floor_angle
        ]
        if pos_gt is not None:
            arrays_to_sync.append(pos_gt)
        if vel_gt is not None:
            arrays_to_sync.append(vel_gt)
        if acc_gt is not None:
            arrays_to_sync.append(acc_gt)
        if qpos_mjx_input is not None:
            arrays_to_sync.append(qpos_mjx_input)
        if qvel_mjx_input is not None:
            arrays_to_sync.append(qvel_mjx_input)
        if qacc_mjx_input is not None:
            arrays_to_sync.append(qacc_mjx_input)
        if qpos_mjx_gt is not None:
            arrays_to_sync.append(qpos_mjx_gt)
        if qvel_mjx_gt is not None:
            arrays_to_sync.append(qvel_mjx_gt)
        if qacc_mjx_gt is not None:
            arrays_to_sync.append(qacc_mjx_gt)
        if qfrc_inverse_gt_clean is not None:
            arrays_to_sync.append(qfrc_inverse_gt_clean)
        if deviation_learning:
            arrays_to_sync.extend([grf_recon, cop_recon, moment_recon])
        min_len = min(len(x) for x in arrays_to_sync)

        pos, vel, acc = pos[:min_len], vel[:min_len], acc[:min_len]
        if pos_gt is not None:
            pos_gt = pos_gt[:min_len]
        if vel_gt is not None:
            vel_gt = vel_gt[:min_len]
        if acc_gt is not None:
            acc_gt = acc_gt[:min_len]
        cop = cop[:min_len]
        cop_length_units = cop_length_units[:min_len]
        gt_cop_ground_aligned_full = gt_cop_ground_aligned_full[:min_len]
        gt_cop_target_full = gt_cop_target_full[:min_len]
        grf, moments = grf[:min_len], moments[:min_len]
        grf_recon, cop_recon, moment_recon = grf_recon[:min_len], cop_recon[:min_len], moment_recon[:min_len]
        jacp, jacr = jacp[:min_len], jacr[:min_len]
        gt_jacp, gt_jacr = gt_jacp[:min_len], gt_jacr[:min_len]
        ankle_heights, pelvis_rot, com_accel = ankle_heights[:min_len], pelvis_rot[:min_len], com_accel[:min_len]
        contactBoolean = contactBoolean[:min_len]
        com_l, com_r = com_l[:min_len], com_r[:min_len]
        rot_w_to_ga = rot_w_to_ga[:min_len]
        gt_rot_w_to_ga_physics = gt_rot_w_to_ga_physics[:min_len]
        gt_rot_w_to_ga = gt_rot_w_to_ga[:min_len]
        ankle_pos = ankle_pos[:min_len]
        knee_pos = knee_pos[:min_len]
        gt_ankle_pos = gt_ankle_pos[:min_len]
        gt_knee_pos = gt_knee_pos[:min_len]
        qfrc_inverse = qfrc_inverse[:min_len]
        if qpos_mjx_input is not None:
            qpos_mjx_input = qpos_mjx_input[:min_len]
        if qvel_mjx_input is not None:
            qvel_mjx_input = qvel_mjx_input[:min_len]
        if qacc_mjx_input is not None:
            qacc_mjx_input = qacc_mjx_input[:min_len]
        if qpos_mjx_gt is not None:
            qpos_mjx_gt = qpos_mjx_gt[:min_len]
        if qvel_mjx_gt is not None:
            qvel_mjx_gt = qvel_mjx_gt[:min_len]
        if qacc_mjx_gt is not None:
            qacc_mjx_gt = qacc_mjx_gt[:min_len]
        if qfrc_inverse_gt_clean is not None:
            qfrc_inverse_gt_clean = qfrc_inverse_gt_clean[:min_len]
        foot_progression_angle = foot_progression_angle[:min_len]
        calcn_to_floor_angle = calcn_to_floor_angle[:min_len]

        # --- Optional COP Trimming ---
        if trim_cop and contactBoolean is not None:
            is_ds = (contactBoolean[:, 0] == 1) & (contactBoolean[:, 1] == 1)
            is_ds_padded = np.concatenate(([0], is_ds.astype(int), [0]))
            diffs = np.diff(is_ds_padded)
            ds_starts = np.where(diffs == 1)[0]
            ds_ends = np.where(diffs == -1)[0]
            valid_lengths = [e - s for s, e in zip(ds_starts, ds_ends) if s != 0 and e != len(contactBoolean)]
            if valid_lengths:
                trim_amt = int(np.mean(valid_lengths))
                if len(pos) > 2 * trim_amt:
                    # Apply trim to all synced arrays
                    slc = slice(trim_amt, -trim_amt)
                    pos, vel, acc = pos[slc], vel[slc], acc[slc]
                    if pos_gt is not None: pos_gt = pos_gt[slc]
                    if vel_gt is not None: vel_gt = vel_gt[slc]
                    if acc_gt is not None: acc_gt = acc_gt[slc]
                    cop = cop[slc]
                    cop_length_units = cop_length_units[slc]
                    gt_cop_ground_aligned_full = gt_cop_ground_aligned_full[slc]
                    gt_cop_target_full = gt_cop_target_full[slc]
                    grf, moments = grf[slc], moments[slc]
                    grf_recon, cop_recon, moment_recon = grf_recon[slc], cop_recon[slc], moment_recon[slc]
                    jacp, jacr = jacp[slc], jacr[slc]
                    gt_jacp, gt_jacr = gt_jacp[slc], gt_jacr[slc]
                    ankle_heights, pelvis_rot, com_accel = ankle_heights[slc], pelvis_rot[slc], com_accel[slc]
                    contactBoolean = contactBoolean[slc]
                    com_l, com_r = com_l[slc], com_r[slc]
                    rot_w_to_ga = rot_w_to_ga[slc]
                    gt_rot_w_to_ga_physics = gt_rot_w_to_ga_physics[slc]
                    gt_rot_w_to_ga = gt_rot_w_to_ga[slc]
                    ankle_pos = ankle_pos[slc]
                    knee_pos = knee_pos[slc]
                    gt_ankle_pos = gt_ankle_pos[slc]
                    gt_knee_pos = gt_knee_pos[slc]
                    qfrc_inverse = qfrc_inverse[slc]
                    if qpos_mjx_input is not None: qpos_mjx_input = qpos_mjx_input[slc]
                    if qvel_mjx_input is not None: qvel_mjx_input = qvel_mjx_input[slc]
                    if qacc_mjx_input is not None: qacc_mjx_input = qacc_mjx_input[slc]
                    if qpos_mjx_gt is not None: qpos_mjx_gt = qpos_mjx_gt[slc]
                    if qvel_mjx_gt is not None: qvel_mjx_gt = qvel_mjx_gt[slc]
                    if qacc_mjx_gt is not None: qacc_mjx_gt = qacc_mjx_gt[slc]
                    if qfrc_inverse_gt_clean is not None: qfrc_inverse_gt_clean = qfrc_inverse_gt_clean[slc]
                    foot_progression_angle = foot_progression_angle[slc]
                    calcn_to_floor_angle = calcn_to_floor_angle[slc]
                    if id_gt_mjx is not None: id_gt_mjx = id_gt_mjx[slc]
                    if qfrc_inverse is not None: qfrc_inverse = qfrc_inverse[slc]
                    if qfrc_inverse_processed is not None: qfrc_inverse_processed = qfrc_inverse_processed[slc]
                    if qfrc_inverse_mocap is not None: qfrc_inverse_mocap = qfrc_inverse_mocap[slc]
                    min_len = len(pos)

        # Standardize shapes
        if height.ndim == 0: height = height.reshape(1, 1)
        if mass.ndim == 0: mass = mass.reshape(1, 1)
        if height.ndim == 1: height = height[:, np.newaxis]
        if mass.ndim == 1: mass = mass[:, np.newaxis]
        
        # Compute qfrc_grf_contribution.
        # Ground truth COP is transformed back to world from the stored
        # ground-aligned calc frame using R_ga->w = R_w->ga^T.
        F_R, M_free_R = grf[:, 0:3], moments[:, 0:3]
        F_L, M_free_L = grf[:, 3:6], moments[:, 3:6]

        rot_ga_to_w_r = np.transpose(gt_rot_w_to_ga_physics[:, 0], (0, 2, 1))
        rot_ga_to_w_l = np.transpose(gt_rot_w_to_ga_physics[:, 1], (0, 2, 1))
        cop_r_world = np.einsum("tij,tj->ti", rot_ga_to_w_r, gt_cop_ground_aligned_full[:, 0:3])
        cop_l_world = np.einsum("tij,tj->ti", rot_ga_to_w_l, gt_cop_ground_aligned_full[:, 3:6])

        r_vec_R = cop_r_world
        M_total_R = M_free_R + np.cross(r_vec_R, F_R)
        r_vec_L = cop_l_world
        M_total_L = M_free_L + np.cross(r_vec_L, F_L)
        
        tau_R = np.einsum('tji,tj->ti', gt_jacp[:, 0], F_R) + np.einsum('tji,tj->ti', gt_jacr[:, 0], M_total_R)
        tau_L = np.einsum('tji,tj->ti', gt_jacp[:, 1], F_L) + np.einsum('tji,tj->ti', gt_jacr[:, 1], M_total_L)
        qfrc_grf_contribution = (tau_R + tau_L).astype(np.float32)

        if not use_mocap_gt:
            qfrc_grf_contribution_file = load_gt_file(
                "qfrc_grf_contribution.npy",
                use_noised_variant=True,
            )
            if qfrc_grf_contribution_file is not None:
                qfrc_grf_contribution = np.asarray(qfrc_grf_contribution_file)
                source_qfrc_name = (
                    _with_file_suffix("qfrc_grf_contribution.npy")
                    if ground_truth_use_noised
                    else "qfrc_grf_contribution.npy"
                )
                _loader_log(f"   🎯 Loading {ground_truth_source_label} qfrc_grf_contribution: {source_qfrc_name}")

        if id_gt_mjx is None and qfrc_inverse_gt_clean is not None:
            id_gt_mjx = np.asarray(qfrc_inverse_gt_clean, dtype=np.float32) - np.asarray(qfrc_grf_contribution, dtype=np.float32)

        # Final safety sync: make sure all torque/ID arrays match the trial length.
        final_len_candidates = [min_len, len(qfrc_grf_contribution)]
        if id_gt_mjx is not None:
            final_len_candidates.append(len(id_gt_mjx))
        if qfrc_inverse is not None:
            final_len_candidates.append(len(qfrc_inverse))
        if qfrc_inverse_processed is not None:
            final_len_candidates.append(len(qfrc_inverse_processed))
        if qfrc_inverse_mocap is not None:
            final_len_candidates.append(len(qfrc_inverse_mocap))
        if qfrc_inverse_gt_clean is not None:
            final_len_candidates.append(len(qfrc_inverse_gt_clean))
        if recalculated_opensim_id_gt is not None:
            final_len_candidates.append(len(recalculated_opensim_id_gt))
        final_len = min(final_len_candidates)

        if final_len != min_len:
            pos, vel, acc = pos[:final_len], vel[:final_len], acc[:final_len]
            if pos_gt is not None:
                pos_gt = pos_gt[:final_len]
            if vel_gt is not None:
                vel_gt = vel_gt[:final_len]
            if acc_gt is not None:
                acc_gt = acc_gt[:final_len]
            cop = cop[:final_len]
            cop_length_units = cop_length_units[:final_len]
            gt_cop_ground_aligned_full = gt_cop_ground_aligned_full[:final_len]
            gt_cop_target_full = gt_cop_target_full[:final_len]
            grf, moments = grf[:final_len], moments[:final_len]
            grf_recon, cop_recon, moment_recon = grf_recon[:final_len], cop_recon[:final_len], moment_recon[:final_len]
            jacp, jacr = jacp[:final_len], jacr[:final_len]
            ankle_heights, pelvis_rot, com_accel = ankle_heights[:final_len], pelvis_rot[:final_len], com_accel[:final_len]
            contactBoolean = contactBoolean[:final_len]
            com_l, com_r = com_l[:final_len], com_r[:final_len]
            rot_w_to_ga = rot_w_to_ga[:final_len]
            gt_rot_w_to_ga_physics = gt_rot_w_to_ga_physics[:final_len]
            gt_rot_w_to_ga = gt_rot_w_to_ga[:final_len]
            gt_jacp, gt_jacr = gt_jacp[:final_len], gt_jacr[:final_len]
            ankle_pos = ankle_pos[:final_len]
            knee_pos = knee_pos[:final_len]
            gt_ankle_pos = gt_ankle_pos[:final_len]
            gt_knee_pos = gt_knee_pos[:final_len]
            foot_progression_angle = foot_progression_angle[:final_len]
            calcn_to_floor_angle = calcn_to_floor_angle[:final_len]
            min_len = final_len

        qfrc_grf_contribution = qfrc_grf_contribution[:min_len]
        if id_gt_mjx is not None:
            id_gt_mjx = id_gt_mjx[:min_len]
        if qfrc_inverse is not None:
            qfrc_inverse = qfrc_inverse[:min_len]
        if qfrc_inverse_processed is not None:
            qfrc_inverse_processed = qfrc_inverse_processed[:min_len]
        if qfrc_inverse_mocap is not None:
            qfrc_inverse_mocap = qfrc_inverse_mocap[:min_len]
        if qfrc_inverse_gt_clean is not None:
            qfrc_inverse_gt_clean = qfrc_inverse_gt_clean[:min_len]
        if recalculated_opensim_id_gt is not None:
            recalculated_opensim_id_gt = np.asarray(recalculated_opensim_id_gt, dtype=np.float32)[:min_len]

        torque_target_source = "qfrc_grf_contribution_from_grf_cop_jacobian"
        full_id_gt_source = "MoCap/ProcessedData/ID_GT_MJX.npy" if id_gt_mjx is not None else "computed_from_qfrc_inverse_minus_tau"
        if opensim_id_bundle is not None:
            full_id_gt_source = str(opensim_id_bundle["source_path"])
        if use_recalculated_opensim_id_gt:
            if qfrc_inverse is None or recalculated_opensim_id_gt is None:
                raise FileNotFoundError(
                    "Cannot build recalculated OpenSim torque target without qfrc_inverse and OpenSim_ID_recalculated.npy."
                )
            id_gt_mjx = np.asarray(recalculated_opensim_id_gt, dtype=np.float32)
            full_id_gt_source = "MoCap/ProcessedData/OpenSim_ID_recalculated.npy"
            if force_gt_grf_contribution:
                # Keep the MoCap analytic grf_contribution (gt_jacp/gt_rot/gt_cop · F,
                # computed above) as the torque target instead of overwriting it with
                # qfrc_inverse_processed - recalc_ID. The full-ID GT still uses recalc.
                torque_target_source = "MoCap/ProcessedData/qfrc_grf_contribution (forced GT, recalc full-ID)"
            else:
                target_width = min(qfrc_inverse.shape[1], recalculated_opensim_id_gt.shape[1])
                qfrc_grf_recalc = np.zeros_like(qfrc_grf_contribution, dtype=np.float32)
                qfrc_grf_recalc[:, :target_width] = (
                    np.asarray(qfrc_inverse[:, :target_width], dtype=np.float32)
                    - np.asarray(recalculated_opensim_id_gt[:, :target_width], dtype=np.float32)
                )
                qfrc_grf_contribution = qfrc_grf_recalc
                torque_target_source = "Video/ProcessedData/qfrc_inverse_minus_MoCap/ProcessedData/OpenSim_ID_recalculated"

        if deviation_learning:
            grf_recon = grf_recon[:min_len]
            cop_recon = cop_recon[:min_len]
            moment_recon = moment_recon[:min_len]
        else:
            grf_recon = np.zeros_like(grf, dtype=np.float32)
            cop_recon = np.zeros_like(cop, dtype=np.float32)
            moment_recon = np.zeros_like(moments, dtype=np.float32)

        qfrc_inverse_norm_factor = compute_bodyweight_height_norm_factor(
            mass[:min_len],
            height[:min_len],
            xp=np,
        ).astype(np.float32, copy=False)
        qfrc_inverse_scaled = (
            normalize_qfrc_inverse_by_bw_height(
                qfrc_inverse[:min_len],
                mass[:min_len],
                height[:min_len],
                xp=np,
            ).astype(np.float32, copy=False)
            if qfrc_inverse is not None
            else None
        )
        qfrc_inverse_gt_scaled = (
            normalize_qfrc_inverse_by_bw_height(
                qfrc_inverse_gt_clean[:min_len],
                mass[:min_len],
                height[:min_len],
                xp=np,
            ).astype(np.float32, copy=False)
            if qfrc_inverse_gt_clean is not None
            else None
        )
        # The KAM moment arm (knee->COP vectors) is a target-side kinematic quantity.
        # Prefer a saved MoCap vector; when that export is absent, fall back to the
        # selected input-source ProcessedData vector (the video path for an OpenCap
        # LOSO), keeping the vector source explicit through knee_to_cop_base.
        kam_bases = [mocap_dir] if mocap_dir.exists() else []
        if load_path not in kam_bases:
            kam_bases.append(load_path)
        knee_to_cop_vectors = None
        knee_to_cop_base = None
        for kam_base in kam_bases:
            knee_to_cop_vectors = load_gt_file(
                "KneeToCOP_Vectors_Mocap.npy",
                base_dir=kam_base,
                use_noised_variant=True,
            )
            if knee_to_cop_vectors is None:
                knee_to_cop_vectors = load_gt_file(
                    "KneeToCOP_Vectors.npy",
                    base_dir=kam_base,
                    use_noised_variant=True,
                )
            if knee_to_cop_vectors is not None:
                knee_to_cop_base = kam_base
                break
        if knee_to_cop_vectors is not None:
            knee_to_cop_vectors = np.asarray(knee_to_cop_vectors, dtype=np.float32)[:min_len]
            if subtract_ankle_height_knee_vecs:
                if knee_to_cop_vectors.ndim != 2 or knee_to_cop_vectors.shape[1] < 6:
                    raise ValueError(
                        "--subtractAnkleHeightKneeVecs requires KneeToCOP vectors with "
                        f"at least 6 columns, got {knee_to_cop_vectors.shape}: {trial_id}"
                    )
                # Legacy ProcessData placed COP at ankle Z, so the saved Z components
                # are ankle_z-knee_z. Correct them in memory to floor_z-knee_z by
                # subtracting the ankle height from the same source as the vector.
                # No files change.
                knee_to_cop_vectors = knee_to_cop_vectors.copy()
                correction_ankle_pos = (
                    ankle_pos if knee_to_cop_base == load_path else gt_ankle_pos
                )
                knee_to_cop_vectors[:, 2] -= correction_ankle_pos[:min_len, 0, 2]
                knee_to_cop_vectors[:, 5] -= correction_ankle_pos[:min_len, 1, 2]

        subject_model_xml = _resolve_subject_model_xml_path(trial_root)
        subject_name = trial_root.parent.name
        trial_name = trial_root.name

        return {
            "pos": pos, "vel": vel, "acc": acc,
            "pos_gt": pos_gt[:min_len] if pos_gt is not None else None,
            "vel_gt": vel_gt[:min_len] if vel_gt is not None else None,
            "acc_gt": acc_gt[:min_len] if acc_gt is not None else None,
            "height": height[:min_len], "mass": mass[:min_len], "gender": gender_val,
            "com_l": com_l[:min_len], "com_r": com_r[:min_len], "pelvis_rot": pelvis_rot,
            "cop": (cop if use_grf_norm_cop else cop / height[:min_len]), "grf": grf / (mass[:min_len] * 9.8067), "moments": moments / (mass[:min_len] * 9.8067 * height[:min_len]),
            "cop_recon": cop_recon / height[:min_len], "grf_recon": grf_recon / (mass[:min_len] * 9.8067), "moment_recon": moment_recon / (mass[:min_len]*9.8067 * height[:min_len]),
            "cop_raw": cop_length_units, "grf_raw": grf, "moments_raw": moments, # Keep raw units for inference/plotting
            "cop_gt_raw": cop_length_units, "grf_gt_raw": grf, "moments_gt_raw": moments,
            "grf_source_file": str(grf_source_path),
            "grf_source_filename": grf_source_filename,
            "cop_training_target": cop[:min_len],
            "cop_target_is_grf_norm": bool(use_grf_norm_cop),
            "cop_target_filename": cop_target_filename,
            "qfrc_grf_contribution": qfrc_grf_contribution,
            "tau_grf_gt": qfrc_grf_contribution,
            "jacp": jacp, "jacr": jacr, "body_ids": body_ids,
            "gt_jacp": gt_jacp, "gt_jacr": gt_jacr,
            "ankle_heights": ankle_heights,
            "rot_w_to_ga": rot_w_to_ga,
            "ankle_pos": ankle_pos,
            "knee_pos": knee_pos,
            "gt_ankle_pos": gt_ankle_pos,
            "gt_knee_pos": gt_knee_pos,
            "contactBoolean": contactBoolean, "com_accel": com_accel,
            "patient_size": patient_size_vec,
            "forward_vel": np.float32(forward_vel_scalar),
            "foot_progression_angle": foot_progression_angle,
            "calcn_to_floor_angle": calcn_to_floor_angle,
            "ground_truth_source": ground_truth_source_label,
            "ground_truth_processed_dir": str(ground_truth_base_dir),
            "full_id_gt_source": full_id_gt_source,
            "use_opensim_id_gt": bool(use_opensim_id_gt),
            "opensim_id_source_path": (
                str(opensim_id_bundle["source_path"]) if opensim_id_bundle is not None else None
            ),
            "opensim_id_alignment": (
                str(opensim_id_bundle["alignment"]) if opensim_id_bundle is not None else None
            ),
            "torque_target_source": torque_target_source,
            "input_source_folder": input_source_label,
            "input_source": input_source_norm,
            "input_processed_dir": str(load_path),
            "video_processed_dir": str(processed_dir),
            "mocap_processed_dir": str(mocap_dir),
            "input_kinematics_source": (
                "Pos_noised"
                if input_use_noised
                else ("MoCap" if input_source_label == "MoCap" else "Pos")
            ),
            "use_noised_inputs": bool(input_use_noised),
            "id_gt_mjx": id_gt_mjx[:min_len] if id_gt_mjx is not None else None,
            "qfrc_inverse_norm_factor": qfrc_inverse_norm_factor,
            "qfrc_inverse": qfrc_inverse_scaled,
            "qfrc_inverse_raw": qfrc_inverse[:min_len] if qfrc_inverse is not None else None,
            "qfrc_inverse_gt": qfrc_inverse_gt_scaled,
            "qfrc_inverse_gt_raw": qfrc_inverse_gt_clean[:min_len] if qfrc_inverse_gt_clean is not None else None,
            "qfrc_inverse_processed": (
                qfrc_inverse_processed[:min_len] if qfrc_inverse_processed is not None else None
            ),
            "knee_to_cop_vectors": knee_to_cop_vectors,
            "qfrc_inverse_mocap": (
                qfrc_inverse_mocap[:min_len] if qfrc_inverse_mocap is not None else None
            ),
            "qpos_mjx_input": qpos_mjx_input[:min_len] if qpos_mjx_input is not None else None,
            "qvel_mjx_input": qvel_mjx_input[:min_len] if qvel_mjx_input is not None else None,
            "qacc_mjx_input": qacc_mjx_input[:min_len] if qacc_mjx_input is not None else None,
            "qpos_mjx_gt": qpos_mjx_gt[:min_len] if qpos_mjx_gt is not None else None,
            "qvel_mjx_gt": qvel_mjx_gt[:min_len] if qvel_mjx_gt is not None else None,
            "qacc_mjx_gt": qacc_mjx_gt[:min_len] if qacc_mjx_gt is not None else None,
            "gt_rot_w_to_ga": gt_rot_w_to_ga[:min_len] if gt_rot_w_to_ga is not None else None,
            "subject": subject_name,
            "subject_group": subject_group_id(subject_name),
            "trial_name": trial_name,
            "subject_model_xml": str(subject_model_xml) if subject_model_xml is not None else None,
            "trial_dir": str(trial_root),
        "id_no_grf": (
                id_gt_mjx[:min_len]
                if id_gt_mjx is not None
                else (qfrc_inverse[:min_len] if qfrc_inverse is not None else None)
            ),
        }
    except Exception as e:
        print(f"Error loading trial at {path}: {e}")
        return None

class TrialDataLoader:
    """Loads trial data in batches without loading entire dataset into memory."""
    
    def __init__(self, trials: List[Dict], window_size: int = 64, stride: int = 16,
                 batch_size: int = 32, shuffle: bool = True, trim_cop: bool = False,
                 deviation_learning: bool = False, use_noised: bool = False,
                 noised_gt: bool = False,
                 predict_jacobian: bool = False,
                 opencap_val: bool = False,
                 input_source: str = "processed",
                 include_pelvis_euler: bool = True,
                 include_ankle_heights: bool = True,
                 include_jacobian_input: bool = True,
                 include_auxiliary_denoising_inputs: bool = True,
                 prediction_margin_frames: int = 20,
                 require_qprime_state: bool = False,
                 use_grf_norm_cop: bool = False,
                 use_grf_nofilt: Optional[bool] = None,
                 use_os_filtering: bool = False,
                 use_opensim_id_gt: bool = False,
                 use_recalculated_opensim_id_gt: bool = False,
                 force_gt_grf_contribution: bool = False,
                 grf_grm_from_processed: bool = False,
                 subtract_ankle_height_knee_vecs: bool = False,
                 allow_missing_noised: bool = False,
                 edge_mode: str = "legacy",
                 edge_trim_frames: int = 0,
                 drop_last: bool = True,
                 balance_speed_gender: bool = False,
                 gender_balance: bool = True,
                 speed_bin_width: float = 0.05,
                 speed_weight_power: float = 0.5,
                 weight_clip_ratio: float = 3.0,
                 window_split_role: Optional[str] = None,
                 window_sample_frac: float = 1.0,
                 window_train_frac: float = 0.7,
                 window_split_seed: int = 42):
        self.trials = trials
        # Window-level random split (opt-in fast-subset mode). When window_split_role is
        # 'train' or 'val', both loaders receive the SAME pool of trials; each window
        # (trial, start_idx) is deterministically hashed to a uniform pair (u1, u2):
        # keep it only if u1 < window_sample_frac, then assign it to 'train' when
        # u2 < window_train_frac else 'val'. The two loaders therefore draw disjoint,
        # reproducible window subsets that mix all subjects.
        self.window_split_role = window_split_role
        self.window_sample_frac = float(window_sample_frac)
        self.window_train_frac = float(window_train_frac)
        self.window_split_seed = int(window_split_seed)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.trim_cop = trim_cop
        self.deviation_learning = deviation_learning
        self.use_noised = use_noised
        self.noised_gt = noised_gt
        self.allow_missing_noised = allow_missing_noised
        edge_mode = str(edge_mode or "legacy").strip().lower()
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {EDGE_MODES}, got {edge_mode!r}")
        self.edge_mode = edge_mode
        self.edge_trim_frames = max(0, int(edge_trim_frames))
        if self.edge_mode == "train" and self.edge_trim_frames * 2 >= self.window_size:
            raise ValueError(
                "edge_trim_frames is too large for the window: need "
                f"2 * {self.edge_trim_frames} < window_size {self.window_size}"
            )
        self.n_dropped_short_trials = 0
        self.predict_jacobian = predict_jacobian
        self.opencap_val = opencap_val
        self.input_source = str(input_source)
        self.include_pelvis_euler = include_pelvis_euler
        self.include_ankle_heights = bool(include_ankle_heights)
        self.include_jacobian_input = bool(include_jacobian_input)
        self.include_auxiliary_denoising_inputs = bool(include_auxiliary_denoising_inputs)
        self.prediction_margin_frames = prediction_margin_frames
        self.require_qprime_state = bool(require_qprime_state)
        self.use_grf_norm_cop = bool(use_grf_norm_cop)
        self.use_grf_nofilt = use_grf_nofilt
        self.use_os_filtering = bool(use_os_filtering)
        self.use_opensim_id_gt = bool(use_opensim_id_gt)
        self.use_recalculated_opensim_id_gt = bool(use_recalculated_opensim_id_gt)
        self.force_gt_grf_contribution = bool(force_gt_grf_contribution)
        self.grf_grm_from_processed = bool(grf_grm_from_processed)
        self.subtract_ankle_height_knee_vecs = bool(subtract_ankle_height_knee_vecs)
        self.drop_last = bool(drop_last)
        self.balance_speed_gender = bool(balance_speed_gender)
        self.gender_balance = bool(gender_balance)
        self.speed_bin_width = float(speed_bin_width)
        self.speed_weight_power = float(speed_weight_power)
        self.weight_clip_ratio = float(weight_clip_ratio)
        validate_prediction_margin(self.window_size, self.prediction_margin_frames)

        self.trial_window_counts = []
        self.total_windows = 0
        # Per-trial speed/gender collected during the count pre-pass (parallel to
        # trial_window_counts). Used to build the balancing weights below.
        self._trial_speed = []
        self._trial_gender = []
        # Per-trial scalar loss weight (1.0 when balancing disabled). Indexed by the
        # same trial_idx used in trial_window_counts / window_index.
        self.trial_weights = []
        self.balance_stats = None
        self._md_cache: Dict[str, float] = {}
        skipped_count = 0

        # Pre-compute window counts per trial (lightweight)
        for trial_info in trials:
            # Quick check of trial length
            try:
                trial_root = resolve_trial_root_from_path(
                    Path(str(trial_info.get("trial_root", trial_info.get("training_data_path"))))
                )
                input_source_norm = normalize_input_source_name(self.input_source)
                td_path = Path(
                    str(
                        trial_info.get(
                            "mocap_processed_path" if input_source_norm == "mocap" else "video_processed_path",
                            trial_info.get("training_data_path", source_processed_dir(trial_root, input_source_norm)),
                        )
                    )
                )
                if td_path.name != "ProcessedData":
                    td_path = source_processed_dir(trial_root, input_source_norm)
                video_td = Path(str(trial_info.get("video_processed_path", video_processed_dir(trial_root))))
                mocap_td = Path(str(trial_info.get("mocap_processed_path", mocap_processed_dir(trial_root))))
                gt_path = mocap_td if self.opencap_val else td_path

                # Mirror load_single_trial's all-or-nothing fallback: a trial whose
                # ProcessedData carries no `_noised.npy` files at all is served from
                # its clean files rather than being dropped here. Kept in sync with
                # that check on purpose - this pre-scan decides whether the trial is
                # ever handed to the loader, so the two must agree or the trial is
                # skipped before load_single_trial can rescue it.
                scan_use_noised = bool(self.use_noised)
                scan_noised_gt = bool(self.noised_gt)
                if self.allow_missing_noised:
                    if scan_use_noised and not any(td_path.glob(f"*{NOISED_FILE_SUFFIX}.npy")):
                        scan_use_noised = False
                    if scan_noised_gt and not any(gt_path.glob(f"*{NOISED_FILE_SUFFIX}.npy")):
                        scan_noised_gt = False

                required_paths = [
                    td_path / "pelvis_rot_matrix.npy",
                ]
                if self.use_grf_norm_cop:
                    required_paths.extend([
                        gt_path / "COP_CalcFrame_GroundAligned.npy",
                        gt_path / "GRF_Cleaned.npy",
                        gt_path / "Mass_kg.npy",
                        gt_path / "Height_m.npy",
                    ])
                else:
                    required_paths.append(gt_path / _cop_target_filename(False))
                if self.use_grf_nofilt is True:
                    required_paths.append(gt_path / "GRF_NoFilt_Trimmed.npy")
                elif self.use_grf_nofilt is False:
                    required_paths.append(gt_path / "GRF_Cleaned.npy")
                elif not ((gt_path / "GRF_NoFilt_Trimmed.npy").exists() or (gt_path / "GRF_Cleaned.npy").exists()):
                    skipped_count += 1
                    continue
                if self.require_qprime_state:
                    required_paths.extend([
                        td_path / "pos_mjx.npy",
                        td_path / "qvel_mjx.npy",
                        td_path / "qacc_mjx.npy",
                        td_path / "qfrc_inverse.npy",
                    ])
                    if scan_use_noised:
                        required_paths.extend([
                            td_path / _with_file_suffix("pos_mjx.npy"),
                            td_path / _with_file_suffix("qvel_mjx.npy"),
                            td_path / _with_file_suffix("qacc_mjx.npy"),
                        ])
                if scan_noised_gt and not self.opencap_val:
                    if self.use_grf_norm_cop:
                        required_paths.append(gt_path / _with_file_suffix("COP_CalcFrame_GroundAligned.npy"))
                    else:
                        required_paths.append(gt_path / _with_file_suffix(_cop_target_filename(False)))
                if self.use_recalculated_opensim_id_gt and self.opencap_val:
                    required_paths.append(mocap_td / "OpenSim_ID_recalculated.npy")
                if scan_use_noised:
                    required_paths.extend([
                        td_path / _with_file_suffix("pos_inputs.npy"),
                        td_path / _with_file_suffix("vel_inputs.npy"),
                        td_path / _with_file_suffix("acc_inputs.npy"),
                        td_path / _with_file_suffix("pelvis_rot_matrix.npy"),
                        td_path / _with_file_suffix("ankle_heights.npy"),
                        td_path / _with_file_suffix("COM_r.npy"),
                        td_path / _with_file_suffix("COM_l.npy"),
                        td_path / _with_file_suffix("COM_Acc_Global.npy"),
                        td_path / _with_file_suffix("forwardVel.npy"),
                        td_path / _with_file_suffix("Foot_ProgressionAngle.npy"),
                        td_path / _with_file_suffix("CalcnToFloor_AngleDeg.npy"),
                    ])
                if not all(p.exists() for p in required_paths):
                    skipped_count += 1
                    continue

                # Use cached length if available to avoid NAS disk hits
                trial_len = trial_info.get("length")
                
                if trial_len is None:
                    # Standard ProcessedData path
                    pos_name = _with_file_suffix("pos_inputs.npy") if scan_use_noised else "pos_inputs.npy"
                    pos_path = td_path / pos_name
                    
                    if not pos_path.exists():
                        skipped_count += 1
                        continue

                    shape = np.load(pos_path, mmap_mode='r').shape
                    trial_len = shape[0]
                
                if trial_len > 0:
                    if self.edge_mode == "train" and self.edge_trim_frames > 0:
                        # Trim the trial edges BEFORE windowing, so no window ever
                        # sees them. Windows are then offset back into absolute
                        # trial coordinates for slicing.
                        trim = self.edge_trim_frames
                        usable = int(trial_len) - 2 * trim
                        if usable < self.window_size:
                            self.n_dropped_short_trials += 1
                            continue
                        window_starts = [
                            s + trim
                            for s in build_window_start_indices(
                                seq_len=usable,
                                window_size=self.window_size,
                                stride=self.stride,
                            )
                        ]
                    else:
                        window_starts = build_window_start_indices(
                            seq_len=int(trial_len),
                            window_size=self.window_size,
                            stride=self.stride,
                        )
                    if self.window_split_role is not None:
                        window_starts = self._filter_windows_for_split(trial_info, window_starts)
                    self.trial_window_counts.append((trial_info, window_starts))
                    self.total_windows += len(window_starts)
                if self.balance_speed_gender:
                    self._trial_speed.append(self._read_trial_speed(td_path))
                    self._trial_gender.append(self._read_trial_gender(td_path))
            except Exception as e:
                print(f"Warning: Skipping trial {trial_info['trial_name']}: {e}")
                skipped_count += 1

        if skipped_count > 0:
            print(f"Note: Skipped {skipped_count} trials (missing required files or errors)")
        if self.n_dropped_short_trials:
            print(
                f"Note: Dropped {self.n_dropped_short_trials} trials shorter than "
                f"window_size={self.window_size} after a {self.edge_trim_frames}-frame "
                f"edge trim at each end (edge_mode='train')."
            )

        _loader_log(
            f"DataLoader initialized: {len(self.trial_window_counts)} trials, "
            f"{self.total_windows} total windows"
        )

        # Compute per-trial loss weights for speed/gender balancing (training only).
        self._build_trial_weights()

        # Create window index mapping
        self._build_window_index()

    def _read_trial_speed(self, td_path: Path) -> float:
        """Read trial-mean walking speed (m/s) from forwardVel.npy, mirroring load_single_trial."""
        name = _with_file_suffix("forwardVel.npy") if self.use_noised else "forwardVel.npy"
        p = td_path / name
        if not p.exists():
            p = td_path / "forwardVel.npy"
        try:
            arr = np.load(p).astype(np.float32).reshape(-1)
            if arr.size == 0:
                return 0.0
            v = float(np.nanmean(arr))
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    def _read_trial_gender(self, td_path: Path) -> float:
        """Resolve gender (1.0 male / 0.0 female / 0.5 unknown) from Patient_MD.json, cached per directory."""
        current = td_path
        for _ in range(5):
            key = str(current)
            if key in self._md_cache:
                return self._md_cache[key]
            md_path = current / "Patient_MD.json"
            if md_path.exists():
                gender_val = 0.5
                try:
                    with open(md_path, "r") as f:
                        sex = str(json.load(f).get("BiologicalSex", "")).lower()
                    if sex == "male":
                        gender_val = 1.0
                    elif sex == "female":
                        gender_val = 0.0
                except Exception:
                    gender_val = 0.5
                self._md_cache[key] = gender_val
                return gender_val
            current = current.parent
        return 0.5

    def _build_trial_weights(self):
        """Populate self.trial_weights (one scalar per kept trial) and self.balance_stats."""
        n_trials = len(self.trial_window_counts)
        if not self.balance_speed_gender:
            self.trial_weights = [1.0] * n_trials
            return

        window_counts = [len(ws) for (_info, ws) in self.trial_window_counts]
        self.trial_weights, self.balance_stats = compute_balance_weights(
            speeds=self._trial_speed,
            genders=self._trial_gender,
            window_counts=window_counts,
            bin_width=self.speed_bin_width,
            speed_power=self.speed_weight_power,
            clip_ratio=self.weight_clip_ratio,
            gender_balance=self.gender_balance,
        )
    
    def _build_window_index(self):
        """Build a flat index of (trial_idx, window_idx) for each window."""
        self.window_index = []
        for trial_idx, (trial_info, window_starts) in enumerate(self.trial_window_counts):
            for window_idx in range(len(window_starts)):
                self.window_index.append((trial_idx, window_idx))
    
    def __len__(self):
        return max(1, self.total_windows // self.batch_size)
    
    def _trial_split_key(self, trial_info: Dict) -> str:
        """Stable per-trial identifier for deterministic window hashing.

        Must be identical for the train and val loaders so their window subsets
        partition the same pool. Uses the trial's on-disk path (stable across the
        two loaders since both are given the same trial list).
        """
        return str(
            trial_info.get("trial_root")
            or trial_info.get("training_data_path")
            or trial_info.get("trial_name")
            or ""
        )

    def _window_split_uniforms(self, trial_key: str, start_idx: int):
        """Two independent deterministic uniforms in [0,1) for one window.

        Uses md5 (not Python's salted hash) so the assignment is reproducible across
        processes and runs for a given seed.
        """
        digest = hashlib.md5(f"{trial_key}|{int(start_idx)}|{self.window_split_seed}".encode("utf-8")).digest()
        u1 = int.from_bytes(digest[0:8], "big") / float(1 << 64)
        u2 = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        return u1, u2

    def _filter_windows_for_split(self, trial_info: Dict, window_starts):
        """Keep only the windows assigned to this loader's role (train/val).

        A window is kept in the sample when u1 < window_sample_frac, then assigned to
        'train' when u2 < window_train_frac else 'val'; we return only the starts whose
        assigned role matches self.window_split_role.
        """
        key = self._trial_split_key(trial_info)
        kept = []
        for start_idx in window_starts:
            u1, u2 = self._window_split_uniforms(key, start_idx)
            if u1 >= self.window_sample_frac:
                continue  # not in the sampled subset
            role = "train" if u2 < self.window_train_frac else "val"
            if role == self.window_split_role:
                kept.append(start_idx)
        return kept

    def _load_trial(self, trial_info: Dict) -> Optional[Dict]:
        """Load a single trial's data using consolidated loader."""
        tp = Path(str(trial_info.get("trial_root", trial_info["training_data_path"])))
        trial_root = resolve_trial_root_from_path(tp)
        return load_single_trial(
            trial_root,
            trim_cop=self.trim_cop,
            deviation_learning=self.deviation_learning,
            opencap_val=self.opencap_val,
            input_source=self.input_source,
            use_noised=self.use_noised,
            noised_gt=self.noised_gt,
            use_grf_norm_cop=self.use_grf_norm_cop,
            use_grf_nofilt=self.use_grf_nofilt,
            use_os_filtering=self.use_os_filtering,
            use_opensim_id_gt=self.use_opensim_id_gt,
            use_recalculated_opensim_id_gt=self.use_recalculated_opensim_id_gt,
            force_gt_grf_contribution=self.force_gt_grf_contribution,
            grf_grm_from_processed=self.grf_grm_from_processed,
            subtract_ankle_height_knee_vecs=self.subtract_ankle_height_knee_vecs,
            allow_missing_noised=self.allow_missing_noised,
        )

    def _load_trial_by_idx(self, trial_idx: int) -> Optional[Dict]:
        """Load a trial by its index in trial_window_counts. Used for threaded loading."""
        trial_info, window_starts = self.trial_window_counts[trial_idx]
        trial_data = self._load_trial(trial_info)
        return (trial_idx, window_starts, trial_data)

    def _extract_windows_from_trial(self, trial_data, window_starts, trial_idx: int):
        """Extract all windows from a loaded trial."""
        windows = []
        sample_weight = np.float32(
            self.trial_weights[trial_idx] if trial_idx < len(self.trial_weights) else 1.0
        )
        trial_length = len(trial_data["pos"])
        padded_trial_data = trial_data
        # In 'train' mode short trials were already dropped, so this pad path only
        # ever runs for legacy/infer.
        if 0 < trial_length < self.window_size:
            padded_trial_data = dict(trial_data)
            pad_len = self.window_size - trial_length
            for key, value in trial_data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == trial_length:
                    pad_width = [(0, pad_len)] + [(0, 0)] * (value.ndim - 1)
                    padded_trial_data[key] = np.pad(value, pad_width, mode="edge")

        for start_idx in window_starts:
            end_idx = start_idx + self.window_size

            if end_idx > len(padded_trial_data["pos"]):
                continue

            if self.edge_mode == "legacy":
                supervision_mask = build_window_supervision_mask(
                    window_size=self.window_size,
                    window_start_idx=start_idx,
                    trial_length=trial_length,
                    prediction_margin_frames=self.prediction_margin_frames,
                )
            else:
                # Every frame of the window is supervised. The bounds only exclude
                # frames outside the usable region: the pre-windowing trim in
                # 'train' mode, or edge padding in 'infer' mode.
                trim = self.edge_trim_frames if self.edge_mode == "train" else 0
                supervision_mask = build_full_window_supervision_mask(
                    window_size=self.window_size,
                    window_start_idx=start_idx,
                    valid_lo=trim,
                    valid_hi=trial_length - trim,
                )

            # Temporal model input. Jacobian and rotation arrays also stay in the
            # batch for torque/metrics bookkeeping; auxiliary qfrc/rotation inputs
            # are optionally appended when training denoising checkpoints.
            pos_input = select_pos_input_columns(
                padded_trial_data["pos"][start_idx:end_idx],
                include_pelvis_euler=self.include_pelvis_euler,
            )
            input_parts = [
                padded_trial_data["pelvis_rot"][start_idx:end_idx],
                pos_input,
                padded_trial_data["vel"][start_idx:end_idx],
                padded_trial_data["com_r"][start_idx:end_idx],
                padded_trial_data["com_l"][start_idx:end_idx],
                padded_trial_data["com_accel"][start_idx:end_idx],
                # contactBoolean removed from inputs: model now predicts it as output dims 12-13
            ]
            if self.include_ankle_heights:
                input_parts.append(padded_trial_data["ankle_heights"][start_idx:end_idx])
            if self.include_jacobian_input:
                input_parts.append(
                    flatten_jacobian_components(
                        padded_trial_data["jacp"][start_idx:end_idx],
                        padded_trial_data["jacr"][start_idx:end_idx],
                    )
                )

            # Include reconstructed curves as direct temporal inputs only when
            # DeviationLearning is enabled.
            if self.deviation_learning:
                moment_recon_full = padded_trial_data["moment_recon"][start_idx:end_idx]
                if moment_recon_full.shape[-1] >= 6:
                    moment_recon_input = moment_recon_full[:, [2, 5]]
                else:
                    moment_recon_input = moment_recon_full
                input_parts.extend([
                    padded_trial_data["cop_recon"][start_idx:end_idx],
                    padded_trial_data["grf_recon"][start_idx:end_idx],
                    moment_recon_input,
                ])

            # Always include non-auxiliary geometric temporal context.
            input_parts.extend([
                padded_trial_data["foot_progression_angle"][start_idx:end_idx],
                padded_trial_data["calcn_to_floor_angle"][start_idx:end_idx],
            ])
            if self.include_auxiliary_denoising_inputs:
                input_parts.append(padded_trial_data["qfrc_inverse"][start_idx:end_idx])
                input_parts.append(
                    flatten_rotation_matrices(
                        padded_trial_data["rot_w_to_ga"][start_idx:end_idx]
                    )
                )

            input_window = np.concatenate(input_parts, axis=1)

            moment_recon_window_full = padded_trial_data["moment_recon"][start_idx:end_idx]
            if moment_recon_window_full.shape[-1] >= 6:
                moment_recon_window = moment_recon_window_full[:, [2, 5]]
            else:
                moment_recon_window = moment_recon_window_full

            static_context = np.array([
                trial_data["height"][0, 0],
                trial_data["mass"][0, 0],
                trial_data["gender"],
                trial_data["patient_size"][0],
                trial_data["patient_size"][1],
                trial_data["patient_size"][2],
                trial_data["patient_size"][3],
                trial_data["forward_vel"],
            ], dtype=np.float32)

            window_entry = {
                "input": input_window,
                "static_context": static_context,
                "sample_weight": sample_weight,
                "trial_idx": np.int32(trial_idx),
                "window_start_idx": np.int32(start_idx),
                "trial_length": np.int32(trial_length),
                "supervision_mask": supervision_mask,
                "cop": padded_trial_data["cop"][start_idx:end_idx],
                "grf": padded_trial_data["grf"][start_idx:end_idx],
                "moments": padded_trial_data["moments"][start_idx:end_idx, [2, 5]],
                "cop_recon": padded_trial_data["cop_recon"][start_idx:end_idx] if self.deviation_learning else np.zeros_like(padded_trial_data["cop"][start_idx:end_idx]),
                "grf_recon": padded_trial_data["grf_recon"][start_idx:end_idx] if self.deviation_learning else np.zeros_like(padded_trial_data["grf"][start_idx:end_idx]),
                "moment_recon": moment_recon_window if self.deviation_learning else np.zeros_like(padded_trial_data["moments"][start_idx:end_idx, [2, 5]]),
                "qfrc_grf_contribution": padded_trial_data["qfrc_grf_contribution"][start_idx:end_idx],
                "qfrc_inverse_norm_factor": padded_trial_data["qfrc_inverse_norm_factor"][start_idx:end_idx],
                "qfrc_inverse_input": padded_trial_data["qfrc_inverse"][start_idx:end_idx],
                "qfrc_inverse_input_raw": padded_trial_data["qfrc_inverse_raw"][start_idx:end_idx] if padded_trial_data.get("qfrc_inverse_raw") is not None else None,
                "qfrc_inverse_gt": padded_trial_data["qfrc_inverse_gt"][start_idx:end_idx] if padded_trial_data.get("qfrc_inverse_gt") is not None else None,
                "qfrc_inverse_gt_raw": padded_trial_data["qfrc_inverse_gt_raw"][start_idx:end_idx] if padded_trial_data.get("qfrc_inverse_gt_raw") is not None else None,
                "id_gt_mjx": padded_trial_data["id_gt_mjx"][start_idx:end_idx] if padded_trial_data.get("id_gt_mjx") is not None else None,
                "knee_to_cop_vectors": padded_trial_data["knee_to_cop_vectors"][start_idx:end_idx] if padded_trial_data.get("knee_to_cop_vectors") is not None else None,
                "qpos_mjx_input": padded_trial_data.get("qpos_mjx_input")[start_idx:end_idx] if padded_trial_data.get("qpos_mjx_input") is not None else None,
                "qvel_mjx_input": padded_trial_data.get("qvel_mjx_input")[start_idx:end_idx] if padded_trial_data.get("qvel_mjx_input") is not None else None,
                "qacc_mjx_input": padded_trial_data.get("qacc_mjx_input")[start_idx:end_idx] if padded_trial_data.get("qacc_mjx_input") is not None else None,
                "qpos_mjx_gt": padded_trial_data.get("qpos_mjx_gt")[start_idx:end_idx] if padded_trial_data.get("qpos_mjx_gt") is not None else None,
                "qvel_mjx_gt": padded_trial_data.get("qvel_mjx_gt")[start_idx:end_idx] if padded_trial_data.get("qvel_mjx_gt") is not None else None,
                "qacc_mjx_gt": padded_trial_data.get("qacc_mjx_gt")[start_idx:end_idx] if padded_trial_data.get("qacc_mjx_gt") is not None else None,
                "jacp": padded_trial_data["jacp"][start_idx:end_idx],
                "jacr": padded_trial_data["jacr"][start_idx:end_idx],
                "gt_jacp": padded_trial_data.get("gt_jacp")[start_idx:end_idx] if padded_trial_data.get("gt_jacp") is not None else None,
                "gt_jacr": padded_trial_data.get("gt_jacr")[start_idx:end_idx] if padded_trial_data.get("gt_jacr") is not None else None,
                "ankle_heights": padded_trial_data["ankle_heights"][start_idx:end_idx],
                "rot_w_to_ga": padded_trial_data["rot_w_to_ga"][start_idx:end_idx],
                "gt_rot_w_to_ga": padded_trial_data.get("gt_rot_w_to_ga")[start_idx:end_idx] if padded_trial_data.get("gt_rot_w_to_ga") is not None else None,
                "ankle_pos": padded_trial_data["ankle_pos"][start_idx:end_idx],
                "knee_pos": padded_trial_data["knee_pos"][start_idx:end_idx],
                "gt_ankle_pos": padded_trial_data.get("gt_ankle_pos")[start_idx:end_idx] if padded_trial_data.get("gt_ankle_pos") is not None else None,
                "gt_knee_pos": padded_trial_data.get("gt_knee_pos")[start_idx:end_idx] if padded_trial_data.get("gt_knee_pos") is not None else None,
                "contactBoolean": padded_trial_data["contactBoolean"][start_idx:end_idx],
                "com_accel": padded_trial_data["com_accel"][start_idx:end_idx],
                "body_ids": padded_trial_data["body_ids"],
                "subject": padded_trial_data.get("subject"),
                "trial_name": padded_trial_data.get("trial_name"),
                "subject_model_xml": padded_trial_data.get("subject_model_xml"),
                "cop_target_is_grf_norm": np.float32(1.0 if padded_trial_data.get("cop_target_is_grf_norm") else 0.0),
            }
            windows.append(window_entry)
        return windows

    def __iter__(self):
        """Iterator that yields batches of windows, optimized for NAS access with prefetching."""
        # 1. Shuffle trials
        trial_indices = list(range(len(self.trial_window_counts)))
        if self.shuffle:
            random.shuffle(trial_indices)
        
        # 2. Local buffer to hold windows from multiple trials for variety
        # We load a few trials, mix their windows, and yield them in batches.
        # This maximizes NAS read efficiency while maintaining data variety.
        buffer_trial_size = 10
        num_prefetch_workers = 6  # Number of parallel NAS read threads
        
        batch_count = 0
        total_windows_yielded = 0
        
        # Maintain a persistent pool across trials to ensure only FULL batches are yielded
        # This prevents JAX recompilation triggered by partial batches.
        persistent_window_pool = []
        
        # Split trial indices into chunks
        chunks = [trial_indices[i:i + buffer_trial_size] for i in range(0, len(trial_indices), buffer_trial_size)]
        
        # Use a ThreadPoolExecutor to prefetch the NEXT chunk while processing the current one
        with ThreadPoolExecutor(max_workers=num_prefetch_workers) as executor:
            # Submit first chunk once; each loop consumes one submitted chunk and prefetches the next.
            pending_futures = [executor.submit(self._load_trial_by_idx, tidx) for tidx in chunks[0]] if chunks else None

            for chunk_idx in range(len(chunks)):
                current_futures = pending_futures
                next_chunk_idx = chunk_idx + 1
                pending_futures = (
                    [executor.submit(self._load_trial_by_idx, tidx) for tidx in chunks[next_chunk_idx]]
                    if next_chunk_idx < len(chunks) else None
                )

                # Collect current chunk results (no duplicate processing of chunks)
                for future in as_completed(current_futures):
                    trial_idx, window_starts, trial_data = future.result()
                    if trial_data is not None:
                        windows = self._extract_windows_from_trial(trial_data, window_starts, trial_idx)
                        persistent_window_pool.extend(windows)
                
                # Shuffle the accumulated pool intermittently to maintain variety
                if self.shuffle:
                    random.shuffle(persistent_window_pool)
                
                # Yield as many FULL batches as possible from the current pool
                while len(persistent_window_pool) >= self.batch_size:
                    batch_list = persistent_window_pool[:self.batch_size]
                    persistent_window_pool = persistent_window_pool[self.batch_size:]
                    
                    batch_data = self._collate_batch(batch_list)
                    
                    batch_count += 1
                    total_windows_yielded += len(batch_list)
                    yield batch_data
            
        # Yield any remaining full batches in pool after all chunks are processed
        if self.shuffle:
            random.shuffle(persistent_window_pool)

        while len(persistent_window_pool) >= self.batch_size:
            batch_list = persistent_window_pool[:self.batch_size]
            persistent_window_pool = persistent_window_pool[self.batch_size:]

            batch_data = self._collate_batch(batch_list)

            batch_count += 1
            total_windows_yielded += len(batch_list)
            yield batch_data

        if persistent_window_pool and not self.drop_last:
            batch_list = persistent_window_pool
            batch_data = self._collate_batch(batch_list)
            batch_count += 1
            total_windows_yielded += len(batch_list)
            yield batch_data

        _loader_log(
            f"   DataLoader Epoch Summary: Yielded {batch_count} batches ({total_windows_yielded} windows total)"
        )

    def _collate_batch(self, batch_list: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """Collate only numeric fields into JAX arrays for jitted train/eval steps."""
        batch_data: Dict[str, jnp.ndarray] = {}
        for key in batch_list[0].keys():
            values = [item[key] for item in batch_list]

            if all(v is None for v in values):
                continue
            if any(v is None for v in values):
                continue

            try:
                if key == "body_ids":
                    stacked = np.asarray(values)
                else:
                    stacked = np.stack(values, axis=0)
            except Exception:
                continue

            if not hasattr(stacked, "dtype"):
                continue
            if stacked.dtype.kind not in {"b", "i", "u", "f", "c"}:
                continue

            batch_data[key] = jnp.asarray(stacked)

        return batch_data
