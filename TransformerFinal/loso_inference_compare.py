"""Paired LOSO inference on identical held-out windows.

This module deliberately does not discover data or construct loaders.  A LOSO
runner supplies a model and a ``loader_factory`` so the exact loader settings
used during compatibility checking/fine tuning are also used for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import numpy as np

from data_loader import coerce_independent_dof_width
from direct_torque_utils import DIRECT_TORQUE_NAMES, direct_torque_percent_to_nm
from train import (
    compute_full_external_moments,
    compute_tau_grf_from_predictions,
    decode_cop_signal_to_length,
    normalize_batch,
)
from train_directTorque import normalize_direct_batch


ArrayDict = Dict[str, np.ndarray]


def _params(bundle: Mapping[str, Any]) -> Any:
    """Accept a checkpoint bundle or a bare Flax parameter tree."""
    if "params" in bundle:
        return bundle["params"]
    if "model_params" in bundle:
        return bundle["model_params"]
    if isinstance(bundle.get("variables"), Mapping) and "params" in bundle["variables"]:
        return bundle["variables"]["params"]
    return bundle


def _normalizers(bundle: Mapping[str, Any], fallback: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    value = bundle.get("normalizers", fallback)
    if value is None:
        raise KeyError("Inference requires checkpoint normalizers")
    return value


def _make_loader(factory: Callable[..., Any], trial: Mapping[str, Any]) -> Any:
    """Support factories accepting either a trial or a one-item trial list."""
    try:
        return factory([trial])
    except (TypeError, AttributeError):
        return factory(trial)


def _make_predict_fn(model: Any) -> Callable[[Any, Any, Any], Any]:
    """Compile one apply function and reuse it for both parameter trees."""
    @jax.jit
    def predict(params: Any, x: Any, static_context: Any) -> Any:
        return model.apply({"params": params}, x, static_context, train=False)

    return predict


def _predict(predict_fn: Callable[..., Any], params: Any, batch: Mapping[str, Any]) -> np.ndarray:
    output = predict_fn(params, batch["input"], batch["static_context"])
    if isinstance(output, (tuple, list)):
        output = output[0]
    return np.asarray(jax.device_get(output))


def _frame_mask(batch: Mapping[str, Any], width: int) -> np.ndarray:
    mask = np.asarray(batch.get("supervision_mask", np.ones(width, dtype=bool)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim == 1:
        mask = mask[None, :]
    return mask.astype(bool)


class _Stitcher:
    def __init__(self) -> None:
        self.sums: Dict[str, np.ndarray] = {}
        self.count: Optional[np.ndarray] = None

    def add(self, starts: np.ndarray, lengths: np.ndarray, mask: np.ndarray, arrays: Mapping[str, np.ndarray]) -> None:
        total = int(np.max(lengths))
        if self.count is None:
            self.count = np.zeros(total, dtype=np.float64)
        elif total > len(self.count):
            self.count = np.pad(self.count, (0, total - len(self.count)))
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if name not in self.sums:
                self.sums[name] = np.zeros((len(self.count),) + arr.shape[2:], dtype=np.float64)
            elif total > self.sums[name].shape[0]:
                self.sums[name] = np.pad(
                    self.sums[name], ((0, total - self.sums[name].shape[0]),) + ((0, 0),) * (arr.ndim - 2)
                )
        for b, start in enumerate(np.asarray(starts, dtype=int)):
            for local in np.flatnonzero(mask[b]):
                frame = int(start) + int(local)
                if frame < 0 or frame >= len(self.count):
                    continue
                if not all(np.all(np.isfinite(np.asarray(a)[b, local])) for a in arrays.values()):
                    continue
                self.count[frame] += 1.0
                for name, arr in arrays.items():
                    self.sums[name][frame] += np.asarray(arr)[b, local]

    def finish(self) -> Tuple[ArrayDict, np.ndarray]:
        if self.count is None:
            raise RuntimeError("The held-out loader produced no windows")
        valid = self.count > 0
        result: ArrayDict = {}
        for name, summed in self.sums.items():
            out = np.full(summed.shape, np.nan, dtype=np.float64)
            out[valid] = summed[valid] / self.count[valid].reshape((-1,) + (1,) * (summed.ndim - 1))
            result[name] = out
        return result, valid


def _errors(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    err = np.asarray(pred)[valid] - np.asarray(target)[valid]
    if not err.size:
        return {"mae": None, "rmse": None, "per_channel_mae": []}
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "per_channel_mae": np.mean(np.abs(err), axis=0).tolist(),
    }


def _direct_arrays(
    predict_fn: Callable[..., Any],
    raw: Mapping[str, Any],
    original: Mapping[str, Any],
    fine_tuned: Mapping[str, Any],
    normalizers: Mapping[str, Any],
) -> Tuple[ArrayDict, np.ndarray]:
    norm = normalize_direct_batch(raw, normalizers)
    original_z = _predict(predict_fn, _params(original), norm)
    fine_z = _predict(predict_fn, _params(fine_tuned), norm)
    original_pct = np.asarray(normalizers["direct_torque"].unnormalize(original_z))
    fine_pct = np.asarray(normalizers["direct_torque"].unnormalize(fine_z))
    target_pct = np.asarray(norm["direct_torque_target_raw"])
    static_raw = np.asarray(norm["static_context_raw"])
    return {
        "original_torque_percent_bwh": original_pct,
        "fine_tuned_torque_percent_bwh": fine_pct,
        "target_torque_percent_bwh": target_pct,
        "original_torque_nm": np.asarray(direct_torque_percent_to_nm(original_pct, static_raw, xp=np)),
        "fine_tuned_torque_nm": np.asarray(direct_torque_percent_to_nm(fine_pct, static_raw, xp=np)),
        "target_torque_nm": np.asarray(direct_torque_percent_to_nm(target_pct, static_raw, xp=np)),
    }, _frame_mask(raw, original_pct.shape[1])


def _physics_predictions(
    output: np.ndarray,
    raw: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    config: Mapping[str, Any],
) -> ArrayDict:
    output = np.asarray(output)[..., :14]
    if bool(config.get("deviation_learning", False)):
        cop_ratio = np.asarray(raw["cop_recon"]) + output[..., 0:4] * np.asarray(normalizers["cop"].std)
        grf_ratio = np.asarray(raw["grf_recon"]) + output[..., 4:10] * np.asarray(normalizers["grf"].std)
        moment_ratio = np.asarray(raw["moment_recon"]) + output[..., 10:12] * np.asarray(normalizers["moments"].std)
    else:
        cop_ratio = np.asarray(normalizers["cop"].unnormalize(output[..., 0:4]))
        grf_ratio = np.asarray(normalizers["grf"].unnormalize(output[..., 4:10]))
        moment_ratio = np.asarray(normalizers["moments"].unnormalize(output[..., 10:12]))
    contact = output[..., 12:14]
    if bool(config.get("cop_mask", False)):
        right = (contact[..., 0:1] >= 0.5).astype(np.float32)
        left = (contact[..., 1:2] >= 0.5).astype(np.float32)
        cop_ratio *= np.concatenate([right, right, left, left], axis=-1)
        grf_ratio *= np.concatenate([right, right, right, left, left, left], axis=-1)
        moment_ratio *= np.concatenate([right, left], axis=-1)
    static = np.asarray(raw["static_context"])
    height = static[..., 0:1]
    mass = static[..., 1:2]
    use_grf_norm = bool(np.asarray(raw.get("cop_target_is_grf_norm", 0.0)).reshape(-1)[0])
    cop = np.asarray(decode_cop_signal_to_length(
        cop_ratio, grf_ratio, height[:, None, :], use_grf_norm_cop=use_grf_norm, xp=np
    ))
    grf = grf_ratio * mass[:, None, :] * 9.8067
    moments = moment_ratio * mass[:, None, :] * height[:, None, :] * 9.8067
    full_moments = compute_full_external_moments(
        cop, grf, moments, raw["ankle_heights"], raw["rot_w_to_ga"]
    )
    tau = compute_tau_grf_from_predictions(grf, full_moments, raw["jacp"], raw["jacr"])
    return {
        "cop": cop,
        "grf": grf,
        "free_moments": moments,
        "torque_nm": np.asarray(jax.device_get(tau)),
        "contact_probability": contact,
    }


def _physics_arrays(
    predict_fn: Callable[..., Any],
    raw: Mapping[str, Any],
    original: Mapping[str, Any],
    fine_tuned: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[ArrayDict, np.ndarray]:
    norm = normalize_batch(dict(raw), dict(normalizers))
    op = _physics_predictions(_predict(predict_fn, _params(original), norm), raw, normalizers, config)
    fp = _physics_predictions(_predict(predict_fn, _params(fine_tuned), norm), raw, normalizers, config)
    arrays: ArrayDict = {}
    for key in op:
        arrays[f"original_{key}"] = op[key]
        arrays[f"fine_tuned_{key}"] = fp[key]
    static = np.asarray(raw["static_context"])
    height, mass = static[..., 0:1], static[..., 1:2]
    target_grf_ratio = np.asarray(raw["grf"])
    use_grf_norm = bool(np.asarray(raw.get("cop_target_is_grf_norm", 0.0)).reshape(-1)[0])
    arrays.update({
        "target_cop": np.asarray(decode_cop_signal_to_length(
            raw["cop"], target_grf_ratio, height[:, None, :], use_grf_norm_cop=use_grf_norm, xp=np
        )),
        "target_grf": target_grf_ratio * mass[:, None, :] * 9.8067,
        "target_free_moments": np.asarray(raw["moments"]) * mass[:, None, :] * height[:, None, :] * 9.8067,
        "target_torque_nm": np.asarray(raw.get("tau_grf_gt", raw["qfrc_grf_contribution"])),
    })
    return arrays, _frame_mask(raw, op["cop"].shape[1])


def compare_trial(
    *,
    trial: Mapping[str, Any],
    model: Any,
    original_checkpoint: Mapping[str, Any],
    fine_tuned_checkpoint: Mapping[str, Any],
    loader_factory: Callable[..., Any],
    model_structure: str,
    output_dir: Optional[Path | str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare two parameter sets using one shared pass over held-out windows.

    Returns ``{"metrics", "arrays", "evaluation_mask", "trial_dir"}``.  If
    ``output_dir`` is provided, every stitched array plus metrics and the common
    mask are persisted under ``Subject__Trial``.
    """
    normalizers = _normalizers(original_checkpoint, fine_tuned_checkpoint.get("normalizers"))
    config = dict(config or {})
    stitcher = _Stitcher()
    predict_fn = _make_predict_fn(model)
    family = str(model_structure).lower()
    for raw in _make_loader(loader_factory, trial):
        if family == "direct_torque":
            arrays, mask = _direct_arrays(
                predict_fn, raw, original_checkpoint, fine_tuned_checkpoint, normalizers
            )
        elif family in {"cop_grf_moments", "physics", "cop_grf"}:
            arrays, mask = _physics_arrays(
                predict_fn, raw, original_checkpoint, fine_tuned_checkpoint, normalizers, config
            )
        else:
            raise ValueError(f"Unsupported model_structure: {model_structure!r}")
        stitcher.add(
            np.asarray(raw["window_start_idx"]), np.asarray(raw["trial_length"]), mask, arrays
        )
    arrays, valid = stitcher.finish()
    if family == "direct_torque":
        target_key, units, components = "target_torque_percent_bwh", "percent_bw_height", list(DIRECT_TORQUE_NAMES)
        original_key, fine_key = "original_torque_percent_bwh", "fine_tuned_torque_percent_bwh"
    else:
        target_key, units, components = "target_torque_nm", "Nm", []
        original_key, fine_key = "original_torque_nm", "fine_tuned_torque_nm"
    subject = str(trial.get("subject", ""))
    leaf = str(trial.get("trial_name", "" )).strip("/")
    trial_name = leaf if leaf == subject or leaf.startswith(f"{subject}/") else f"{subject}/{leaf}"
    metrics: Dict[str, Any] = {
        "trial": trial_name,
        "subject": subject,
        "model_structure": family,
        "n_eval_frames": int(np.sum(valid)),
        "torque_units": units,
        "torque_components": components[: arrays[target_key].shape[-1]],
        "original": {"torque": _errors(arrays[original_key], arrays[target_key], valid)},
        "fine_tuned": {"torque": _errors(arrays[fine_key], arrays[target_key], valid)},
    }
    if family != "direct_torque":
        for source in ("original", "fine_tuned"):
            for signal in ("cop", "grf", "free_moments"):
                metrics[source][signal] = _errors(arrays[f"{source}_{signal}"], arrays[f"target_{signal}"], valid)
    original_mae = metrics["original"]["torque"]["mae"]
    fine_mae = metrics["fine_tuned"]["torque"]["mae"]
    metrics["torque_mae_change"] = None if original_mae is None else float(fine_mae - original_mae)
    metrics["torque_mae_improvement_percent"] = (
        None if not original_mae else float(100.0 * (original_mae - fine_mae) / original_mae)
    )

    trial_dir: Optional[Path] = None
    if output_dir is not None:
        trial_dir = Path(output_dir) / trial_name.replace("/", "__")
        trial_dir.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            np.save(trial_dir / f"{name}.npy", np.asarray(value, dtype=np.float32))
        np.save(trial_dir / "evaluation_mask.npy", valid)
        with (trial_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
    return {"metrics": metrics, "arrays": arrays, "evaluation_mask": valid, "trial_dir": trial_dir}


def ankle_power_dual_source(
    *,
    original_torque_nm: np.ndarray,
    fine_tuned_torque_nm: np.ndarray,
    ankle_angular_velocity_rad_s: np.ndarray,
    evaluation_mask: Optional[np.ndarray] = None,
    channel_index: int = 0,
    mass_kg: Optional[float] = None,
    original_power_101_w: Optional[np.ndarray] = None,
    fine_tuned_power_101_w: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create backward-compatible original/fine-tuned ankle-power fields.

    Torque may be ``(frames,)`` or ``(frames, channels)``.  The returned legacy
    ``pred_*`` values intentionally alias the original checkpoint prediction.
    """
    def select(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        return value if value.ndim == 1 else value[:, int(channel_index)]

    omega = np.asarray(ankle_angular_velocity_rad_s).reshape(-1)
    selected_original = select(original_torque_nm)
    selected_fine = select(fine_tuned_torque_nm)
    if len(selected_original) != len(omega) or len(selected_fine) != len(omega):
        raise ValueError("Torque and ankle angular velocity must have identical frame counts")
    original = selected_original * omega
    fine = selected_fine * omega
    valid = np.isfinite(original) & np.isfinite(fine) & np.isfinite(omega)
    if evaluation_mask is not None:
        valid &= np.asarray(evaluation_mask, dtype=bool).reshape(-1)

    def curve_101(values: np.ndarray, supplied: Optional[np.ndarray]) -> np.ndarray:
        if supplied is not None:
            curve = np.asarray(supplied, dtype=float).reshape(-1)
            if len(curve) != 101:
                raise ValueError("Supplied percent-stance power curves must contain 101 points")
            return curve
        kept = values[valid]
        if not kept.size:
            return np.full(101, np.nan)
        if kept.size == 1:
            return np.full(101, kept[0])
        return np.interp(np.linspace(0.0, 1.0, 101), np.linspace(0.0, 1.0, kept.size), kept)

    def stats(prefix: str, values: np.ndarray, supplied_101: Optional[np.ndarray]) -> Dict[str, Any]:
        kept = values[valid]
        curve = curve_101(values, supplied_101)
        curve_json = [None if not np.isfinite(v) else float(v) for v in curve]
        result = {
            f"{prefix}_power_valid_w": kept.tolist(),
            f"{prefix}_power_101_w": curve_json,
            f"{prefix}_peak_power_w": None if not kept.size else float(np.max(kept)),
            f"{prefix}_peak_w": None if not kept.size else float(np.max(kept)),
            f"{prefix}_mean_power_w": None if not kept.size else float(np.mean(kept)),
            f"{prefix}_minimum_power_w": None if not kept.size else float(np.min(kept)),
            f"{prefix}_peak_frame": None if not kept.size else int(np.flatnonzero(valid)[np.argmax(kept)]),
        }
        if mass_kg is not None:
            if float(mass_kg) <= 0:
                raise ValueError("mass_kg must be positive")
            result[f"{prefix}_power_valid_w_per_kg"] = (kept / float(mass_kg)).tolist()
            result[f"{prefix}_power_101_w_per_kg"] = [
                None if v is None else v / float(mass_kg) for v in curve_json
            ]
            result[f"{prefix}_peak_power_w_per_kg"] = (
                None if not kept.size else float(np.max(kept) / float(mass_kg))
            )
        return result

    result = {
        **stats("original_pred", original, original_power_101_w),
        **stats("fine_tuned_pred", fine, fine_tuned_power_101_w),
    }
    for key, value in list(result.items()):
        if key.startswith("original_pred_"):
            result[key.replace("original_pred_", "pred_", 1)] = value
    return result


def _complete_stance_intervals(contact: np.ndarray) -> Sequence[Tuple[int, int]]:
    """Return contact intervals that have both a heel-strike and toe-off in the trial."""
    stance = np.asarray(contact).reshape(-1) > 0.5
    if stance.size < 3:
        return []
    changes = np.diff(stance.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if start > 0 and end < stance.size and end - start >= 2
    ]


def write_ankle_power_stance_report(
    *,
    trial: Mapping[str, Any],
    comparison: Mapping[str, Any],
    model_structure: str,
    output_root: Path | str,
) -> Path:
    """Write the dual-source stance JSON consumed by ``AnklePowerAnalysis``.

    Both checkpoint predictions use the same saved MJX angular velocity, stance
    boundaries, and paired inference mask. Legacy ``pred_*`` fields alias the
    original checkpoint prediction.
    """
    trial_root = Path(str(trial.get("trial_root") or Path(str(trial["training_data_path"])).parent))
    processed = Path(str(trial.get("training_data_path") or trial_root / "ProcessedData"))
    arrays = comparison["arrays"]
    eval_mask = np.asarray(comparison["evaluation_mask"], dtype=bool).reshape(-1)
    family = str(model_structure).lower()
    # Torque predictions use the independent 23-DOF schema, whereas older
    # ProcessedData folders commonly retain qvel in the full model schema
    # (31/33/39/43 DOFs).  Coerce qvel before applying independent ankle
    # indices; indexing the full-width array directly selects other joints.
    qvel = np.asarray(
        coerce_independent_dof_width(
            np.load(processed / "qvel_mjx.npy"),
            label="ankle-power qvel_mjx",
            trial_id=str(trial.get("trial_name") or trial.get("trial") or trial_root.name),
        ),
        dtype=np.float64,
    )
    contact = np.asarray(np.load(processed / "contactBoolean.npy"), dtype=np.float64)
    mass = float(np.asarray(np.load(processed / "Mass_kg.npy"), dtype=np.float64).reshape(-1)[0])

    if family == "direct_torque":
        original_joint_torque = np.asarray(arrays["original_torque_nm"], dtype=np.float64)
        fine_joint_torque = np.asarray(arrays["fine_tuned_torque_nm"], dtype=np.float64)
        target_joint_torque = np.asarray(arrays["target_torque_nm"], dtype=np.float64)
        torque_source = "direct_joint_torque"
    else:
        # The COP/GRF model outputs tau_grf: the generalized contribution from
        # the external ground reaction force.  It is not the net joint moment
        # used for joint power. Match infer.py's full-ID reconstruction:
        #     joint moment = qfrc_inverse - tau_grf
        use_noised = bool(trial.get("use_noised", False))
        qfrc_name = "qfrc_inverse_noised.npy" if use_noised else "qfrc_inverse.npy"
        qfrc_path = processed / qfrc_name
        if not qfrc_path.is_file() and use_noised:
            qfrc_path = processed / "qfrc_inverse.npy"
        qfrc_inverse = np.asarray(
            coerce_independent_dof_width(
                np.load(qfrc_path),
                label="ankle-power qfrc_inverse",
                trial_id=str(trial.get("trial_name") or trial.get("trial") or trial_root.name),
            ),
            dtype=np.float64,
        )
        original_joint_torque = qfrc_inverse[: len(arrays["original_torque_nm"])] - np.asarray(
            arrays["original_torque_nm"], dtype=np.float64
        )
        fine_joint_torque = qfrc_inverse[: len(arrays["fine_tuned_torque_nm"])] - np.asarray(
            arrays["fine_tuned_torque_nm"], dtype=np.float64
        )
        id_gt_path = processed / "ID_GT_MJX.npy"
        if id_gt_path.is_file():
            target_joint_torque = np.asarray(
                coerce_independent_dof_width(
                    np.load(id_gt_path),
                    label="ankle-power ID_GT_MJX",
                    trial_id=str(trial.get("trial_name") or trial.get("trial") or trial_root.name),
                ),
                dtype=np.float64,
            )
            target_source = "ID_GT_MJX"
        else:
            target_joint_torque = qfrc_inverse[: len(arrays["target_torque_nm"])] - np.asarray(
                arrays["target_torque_nm"], dtype=np.float64
            )
            target_source = "qfrc_inverse_minus_target_tau_grf"
        torque_source = f"{qfrc_path.name}_minus_tau_grf; target={target_source}"

    n_frames = min(
        len(eval_mask), len(qvel), len(contact), len(original_joint_torque),
        len(fine_joint_torque), len(target_joint_torque),
    )
    eval_mask = eval_mask[:n_frames]
    qvel = qvel[:n_frames]
    contact = contact[:n_frames]

    if family == "direct_torque":
        torque_indices = {"right": 5, "left": 12}
        velocity_indices = {"right": 10, "left": 17}
    else:
        torque_indices = {"right": 10, "left": 17}
        velocity_indices = torque_indices

    subject = str(trial.get("subject", trial_root.parent.name))
    trial_leaf = str(trial.get("trial", trial_root.name))
    sides: Dict[str, Any] = {}
    for side_index, side in enumerate(("right", "left")):
        torque_idx = torque_indices[side]
        velocity_idx = velocity_indices[side]
        original_torque = original_joint_torque[:n_frames, torque_idx]
        fine_torque = fine_joint_torque[:n_frames, torque_idx]
        target_torque = target_joint_torque[:n_frames, torque_idx]
        omega = qvel[:, velocity_idx]
        complete_stances = []
        for start, end in _complete_stance_intervals(contact[:, side_index]):
            local_valid = eval_mask[start:end]
            valid_indices = np.flatnonzero(local_valid)
            if valid_indices.size < 2:
                continue
            # Prediction margins normally leave one contiguous interior run.
            # Select its bounding interval so start_frame + local argmax remains
            # an exact trial-frame index for the legacy analysis scripts.
            first = start + int(valid_indices[0])
            last = start + int(valid_indices[-1]) + 1
            contiguous_valid = eval_mask[first:last]
            if not np.all(contiguous_valid):
                continue
            omega_segment = omega[first:last]
            power_fields = ankle_power_dual_source(
                original_torque_nm=original_torque[first:last],
                fine_tuned_torque_nm=fine_torque[first:last],
                ankle_angular_velocity_rad_s=omega_segment,
            )
            gt_power = target_torque[first:last] * omega_segment
            if not np.all(np.isfinite(gt_power)):
                continue
            gt_101 = np.interp(
                np.linspace(0.0, 1.0, 101),
                np.linspace(0.0, 1.0, len(gt_power)),
                gt_power,
            )
            power_fields.update({
                "stance_percent_valid": np.linspace(0.0, 100.0, len(gt_power)).tolist(),
                "gt_power_valid_w": gt_power.tolist(),
                "gt_power_101_w": gt_101.tolist(),
                "summary": {
                    "gt_peak_w": float(np.max(gt_power)),
                    "original_pred_peak_w": power_fields["original_pred_peak_w"],
                    "fine_tuned_pred_peak_w": power_fields["fine_tuned_pred_peak_w"],
                    "pred_peak_w": power_fields["pred_peak_w"],
                },
            })
            complete_stances.append({
                "start_frame": int(first),
                "end_frame_exclusive": int(last),
                "length_frames": int(last - first),
                "ankle_power": power_fields,
            })
        sides[side] = {"complete_stances": complete_stances, "opensim_id_peaks": {"peaks": []}}

    output_dir = Path(output_root) / f"{subject}_{trial_leaf}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject}_{trial_leaf}_complete_stance_peak_metrics.json"
    payload = {
        "available": True,
        "model_structure": family,
        "torque_source": torque_source,
        "ground_truth_label": str(trial.get("ground_truth_label", "MJX GT")),
        "trial_frame_count": int(n_frames),
        "subject": {"name": subject, "mass_kg": mass},
        "trial": trial_leaf,
        "sides": sides,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return path


__all__ = ["compare_trial", "ankle_power_dual_source", "write_ankle_power_stance_report"]
