"""LOSO checkpoint helpers for checkpoint-compatible fine-tuning.

This module now reuses the exact `train.py` model architectures for LOSO
fine-tuning so a checkpoint can be loaded and trained for a few additional
epochs without inserting any extra adapter layers.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict

from train import (  # type: ignore
    KinematicsToCOPGRFMoments,
    Normalizer,
    SinusoidalPosEmb,
    TransformerBlock,
    compute_full_external_moments,
    compute_tau_grf_from_predictions,
    compute_total_loss,
    normalize_batch,
)

try:
    from direct_torque_utils import is_direct_torque_hparams  # type: ignore
    from train_directTorque import KinematicsToDirectTorque  # type: ignore
except Exception:  # pragma: no cover - direct torque scripts are optional for legacy checkpoints.
    is_direct_torque_hparams = None  # type: ignore
    KinematicsToDirectTorque = None  # type: ignore

try:
    from train import MultiTaskKinematicsToCOPGRFMoments  # type: ignore
except ImportError:
    MultiTaskKinematicsToCOPGRFMoments = KinematicsToCOPGRFMoments  # type: ignore

try:
    from train import TaskHead  # type: ignore
except ImportError:
    class TaskHead(nn.Module):
        head_d_model: int = 128
        head_num_layers: int = 3
        head_num_heads: int = 4
        head_ff_dim: int = 256
        output_dim: int = 4
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
            x = nn.Dense(self.head_d_model)(x)
            x = nn.gelu(x)
            for _ in range(self.head_num_layers):
                x = TransformerBlock(
                    d_model=self.head_d_model,
                    num_heads=self.head_num_heads,
                    ff_dim=self.head_ff_dim,
                    dropout_rate=self.dropout_rate,
                )(x, train=train)
            x = nn.LayerNorm()(x)
            return nn.Dense(self.output_dim)(x)

TRAINABLE_ADAPTER_ROOTS: Tuple[str, str] = (
    "pre_transformer_adapter",
    "post_transformer_adapter",
)

STANDARD_OUTPUT_DIM = 14
PREDICTED_JACOBIAN_BODY_COUNT = 2
PREDICTED_JACOBIAN_COMPONENT_COUNT = 2
PREDICTED_JACOBIAN_SPATIAL_DIMS = 3
PREDICTED_JACOBIAN_DOF_COUNT = 39
PREDICTED_JACOBIAN_FLAT_DIM = (
    PREDICTED_JACOBIAN_BODY_COUNT
    * PREDICTED_JACOBIAN_COMPONENT_COUNT
    * PREDICTED_JACOBIAN_SPATIAL_DIMS
    * PREDICTED_JACOBIAN_DOF_COUNT
)


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _as_tuple_of_ints(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(int(v) for v in value.tolist())
    text = str(value).strip()
    if not text:
        return default
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return tuple(int(v) for v in parsed)
        except Exception:
            pass
    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    try:
        return tuple(int(p) for p in parts)
    except Exception:
        return default


def _path_is_trainable(path: Tuple[str, ...], trainable_roots: Sequence[str]) -> bool:
    for component in path:
        for root in trainable_roots:
            if component == root or component.startswith(f"{root}_"):
                return True
    return False


@dataclass(frozen=True)
class LosoCheckpointConfig:
    """Canonicalized checkpoint hyperparameters used by the LOSO adapter model."""

    input_dim: int
    static_dim: int = 8
    output_dim: int = 14
    qfrc_inverse_output_dim: int = 0
    rotation_output_dim: int = 6
    predict_jacobian: bool = False
    jacobian_output_dim: int = PREDICTED_JACOBIAN_FLAT_DIM
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_film: bool = False
    use_cnn: bool = True
    cnn_num_layers: int = 2
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5)
    use_multitask: bool = False
    trim_cop: bool = False
    deviation_learning: bool = False
    include_pelvis_euler: bool = False
    use_noised: bool = False
    noised_gt: bool = False
    use_grf_norm_cop: bool = False
    cop_mask: bool = True
    use_contact_weighting: bool = False
    mag_on_off: bool = False
    contact_on_off: bool = False
    contact_weight_multiplier: float = 1.5
    mag_weight: float = 3.0
    cop_weight: float = 1.0
    grf_weight: float = 1.0
    moments_weight: float = 0.25
    contact_weight: float = 1.0
    torque_weight: float = 2.0
    jacobian_weight: float = 1.0
    grf_correction_weight: float = 500.0
    output_reg_weight: float = 0.0
    best_model_by_torque: bool = False
    best_model_torque_weighting: Dict[str, float] = field(default_factory=dict)
    adapter_hidden_dim: int = 0
    adapter_dropout_rate: float = 0.0
    model_structure: str = "cop_grf_moments"

    def model_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for one of the LOSO adapter model classes."""
        return {
            "input_dim": self.input_dim,
            "static_dim": self.static_dim,
            "output_dim": self.output_dim,
            "qfrc_inverse_output_dim": self.qfrc_inverse_output_dim,
            "rotation_output_dim": self.rotation_output_dim,
            "predict_jacobian": self.predict_jacobian,
            "jacobian_output_dim": self.jacobian_output_dim,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "use_film": self.use_film,
            "use_cnn": self.use_cnn,
            "cnn_num_layers": self.cnn_num_layers,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "adapter_hidden_dim": self.adapter_hidden_dim or self.ff_dim,
            "adapter_dropout_rate": self.adapter_dropout_rate or self.dropout_rate,
        }

    def loss_weights(self) -> Dict[str, float]:
        return {
            "cop": self.cop_weight,
            "grf": self.grf_weight,
            "moments": self.moments_weight,
            "jacobian": self.jacobian_weight if self.predict_jacobian else 0.0,
            "contact": self.contact_weight,
            "torque": self.torque_weight,
            "grf_correction": self.grf_correction_weight,
            "output_reg": self.output_reg_weight if self.deviation_learning else 0.0,
        }


def normalize_checkpoint_hyperparameters(raw_hparams: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize legacy/current hyperparameter key spellings to a canonical dict."""
    raw = dict(raw_hparams or {})

    canonical: Dict[str, Any] = dict(raw)
    if is_direct_torque_hparams is not None and is_direct_torque_hparams(raw):
        canonical["model_structure"] = "direct_torque"
        canonical["model_type"] = "direct_torque"
    else:
        canonical["model_structure"] = str(_first_present(raw, ("model_structure", "model_type"), "cop_grf_moments"))
    canonical["use_cnn_explicit"] = any(k in raw for k in ("use_cnn", "UseCNN"))
    canonical["use_multitask_explicit"] = any(k in raw for k in ("use_multitask", "MultiTaskLearning"))
    canonical["input_dim"] = _as_int(
        _first_present(raw, ("input_dim", "inputDim", "in_dim"), None),
        _as_int(raw.get("input_dim"), 0),
    )
    canonical["static_dim"] = _as_int(_first_present(raw, ("static_dim", "staticDim"), 8), 8)
    canonical["output_dim"] = _as_int(_first_present(raw, ("output_dim", "outputDim"), 14), 14)
    canonical["qfrc_inverse_output_dim"] = _as_int(
        _first_present(raw, ("qfrc_inverse_output_dim", "qfrcInverseOutputDim"), 0),
        0,
    )
    canonical["rotation_output_dim"] = _as_int(
        _first_present(raw, ("rotation_output_dim", "rotationOutputDim"), 6),
        6,
    )
    canonical["predict_jacobian"] = _as_bool(
        _first_present(
            raw,
            ("predict_jacobian", "PredictJacobian", "predicted_jacobian", "use_predicted_jacobian"),
            False,
        ),
        False,
    )
    canonical["jacobian_output_dim"] = _as_int(
        _first_present(
            raw,
            ("jacobian_output_dim", "PredictedJacobianDim", "predicted_jacobian_dim", "jacobian_flat_dim"),
            PREDICTED_JACOBIAN_FLAT_DIM,
        ),
        PREDICTED_JACOBIAN_FLAT_DIM,
    )
    if canonical["predict_jacobian"] and canonical["output_dim"] <= STANDARD_OUTPUT_DIM:
        canonical["output_dim"] = STANDARD_OUTPUT_DIM + int(canonical["jacobian_output_dim"])
    elif (
        not canonical["predict_jacobian"]
        and canonical["output_dim"] > STANDARD_OUTPUT_DIM
        and canonical["output_dim"] - STANDARD_OUTPUT_DIM == int(canonical["jacobian_output_dim"])
    ):
        canonical["predict_jacobian"] = True
    canonical["d_model"] = _as_int(_first_present(raw, ("d_model", "dm"), 256), 256)
    canonical["num_heads"] = _as_int(_first_present(raw, ("num_heads", "heads"), 4), 4)
    canonical["num_layers"] = _as_int(_first_present(raw, ("num_layers", "nl"), 4), 4)
    canonical["ff_dim"] = _as_int(_first_present(raw, ("ff_dim", "ff"), 1024), 1024)
    canonical["dropout_rate"] = _as_float(_first_present(raw, ("dropout_rate", "dr"), 0.1), 0.1)
    canonical["use_film"] = _as_bool(_first_present(raw, ("use_film", "subject_film"), False), False)
    canonical["use_cnn"] = _as_bool(_first_present(raw, ("use_cnn", "UseCNN"), True), True)
    canonical["cnn_num_layers"] = _as_int(_first_present(raw, ("cnn_num_layers", "CNN_Num_Layers", "cnl"), 2), 2)
    canonical["cnn_kernel_sizes"] = _as_tuple_of_ints(
        _first_present(raw, ("cnn_kernel_sizes", "CNN_Kernel_Sizes", "cnn_kernels"), (3, 5)),
        (3, 5),
    )
    canonical["use_multitask"] = _as_bool(_first_present(raw, ("use_multitask", "MultiTaskLearning"), False), False)
    canonical["trim_cop"] = _as_bool(_first_present(raw, ("trim_cop", "TrimCOP"), False), False)
    canonical["deviation_learning"] = _as_bool(
        _first_present(raw, ("DeviationLearning", "deviation_learning"), False),
        False,
    )
    canonical["include_pelvis_euler"] = _as_bool(
        _first_present(raw, ("includePelvisEuler", "include_pelvis_euler"), False),
        False,
    )
    canonical["use_noised"] = _as_bool(_first_present(raw, ("UseNoised", "use_noised"), False), False)
    canonical["noised_gt"] = _as_bool(_first_present(raw, ("NoisedGT", "noised_gt"), False), False)
    canonical["use_grf_norm_cop"] = _as_bool(
        _first_present(raw, ("UseGRFNormCOP", "use_grf_norm_cop"), False),
        False,
    )
    canonical["cop_mask"] = _as_bool(_first_present(raw, ("cop_mask", "CopMask"), True), True)
    canonical["torque_grad_through_jacob"] = _as_bool(
        _first_present(raw, ("torque_grad_through_jacob", "torqueGradThroughJacob"), True),
        True,
    )
    canonical["use_contact_weighting"] = _as_bool(
        _first_present(raw, ("use_contact_weighting", "useContactWeighting"), False),
        False,
    )
    canonical["mag_on_off"] = _as_bool(_first_present(raw, ("magOnOff", "mag_on_off"), False), False)
    canonical["contact_on_off"] = _as_bool(
        _first_present(raw, ("contactOnOff", "contact_on_off"), False),
        False,
    )
    canonical["contact_weight_multiplier"] = _as_float(
        _first_present(raw, ("contact_weight_multiplier", "contactWeightMultiplier"), 1.5),
        1.5,
    )
    canonical["mag_weight"] = _as_float(_first_present(raw, ("magWeight", "mag_weight"), 3.0), 3.0)
    canonical["cop_weight"] = _as_float(_first_present(raw, ("cop_weight", "copWeight"), 1.0), 1.0)
    canonical["grf_weight"] = _as_float(_first_present(raw, ("grf_weight", "grfWeight"), 1.0), 1.0)
    canonical["moments_weight"] = _as_float(
        _first_present(raw, ("moments_weight", "momentsWeight"), 0.25),
        0.25,
    )
    canonical["contact_weight"] = _as_float(
        _first_present(raw, ("contact_weight", "contactWeight"), 1.0),
        1.0,
    )
    canonical["torque_weight"] = _as_float(_first_present(raw, ("torque_weight", "torqueWeight"), 2.0), 2.0)
    canonical["qfrc_inverse_weight"] = _as_float(
        _first_present(raw, ("qfrc_inverse_weight", "qfrcInverseWeight"), 1.0),
        1.0,
    )
    canonical["qfrc_inverse_input_reg_weight"] = _as_float(
        _first_present(
            raw,
            ("qfrc_inverse_input_reg_weight", "qfrcInverseInputRegWeight"),
            canonical["qfrc_inverse_weight"],
        ),
        canonical["qfrc_inverse_weight"],
    )
    canonical["rotation_weight"] = _as_float(
        _first_present(raw, ("rotation_weight", "rotationWeight"), 1.0),
        1.0,
    )
    canonical["rotation_input_reg_weight"] = _as_float(
        _first_present(
            raw,
            ("rotation_input_reg_weight", "rotationInputRegWeight"),
            canonical["rotation_weight"],
        ),
        canonical["rotation_weight"],
    )
    canonical["jacobian_weight"] = _as_float(
        _first_present(raw, ("jacobian_weight", "jacobianWeight"), 1.0),
        1.0,
    )
    canonical["jacobian_input_reg_weight"] = _as_float(
        _first_present(
            raw,
            ("jacobian_input_reg_weight", "jacobianInputRegWeight"),
            canonical["jacobian_weight"],
        ),
        canonical["jacobian_weight"],
    )
    canonical["grf_correction_weight"] = _as_float(
        _first_present(raw, ("grf_correction_weight", "grfCorrectionWeight"), 500.0),
        500.0,
    )
    canonical["output_reg_weight"] = _as_float(
        _first_present(raw, ("output_reg_weight", "outputRegWeight"), 0.0),
        0.0,
    )
    canonical["best_model_by_torque"] = _as_bool(
        _first_present(raw, ("BestModelByTorque", "best_model_by_torque"), False),
        False,
    )
    torque_weighting = _first_present(raw, ("BestModel_TorqueWeighting", "best_model_torque_weighting"), {})
    if isinstance(torque_weighting, str):
        try:
            torque_weighting = json.loads(torque_weighting)
        except Exception:
            torque_weighting = {}
    if not isinstance(torque_weighting, Mapping):
        torque_weighting = {}
    canonical["best_model_torque_weighting"] = {
        str(k): _as_float(v, 1.0) for k, v in dict(torque_weighting).items()
    }
    canonical["adapter_hidden_dim"] = _as_int(
        _first_present(raw, ("adapter_hidden_dim", "adapterHiddenDim"), 0),
        0,
    )
    canonical["adapter_dropout_rate"] = _as_float(
        _first_present(raw, ("adapter_dropout_rate", "adapterDropoutRate"), 0.0),
        0.0,
    )
    canonical["window_size"] = _as_int(_first_present(raw, ("window_size", "windowSize"), 0), 0)
    canonical["stride"] = _as_int(_first_present(raw, ("stride",), 16), 16)
    canonical["prediction_margin_frames"] = _as_int(
        _first_present(raw, ("prediction_margin_frames", "predictionMarginFrames"), 20),
        20,
    )
    canonical["batch_size"] = _as_int(_first_present(raw, ("batch_size", "batchSize"), 64), 64)
    canonical["epochs"] = _as_int(_first_present(raw, ("epochs",), 40), 40)
    canonical["learning_rate"] = _as_float(_first_present(raw, ("learning_rate", "lr"), 1.84e-4), 1.84e-4)
    canonical["weight_decay"] = _as_float(_first_present(raw, ("weight_decay", "wd"), 1e-2), 1e-2)
    return canonical


def infer_checkpoint_family(
    checkpoint_params: Optional[Mapping[str, Any]],
    canonical_hparams: Optional[Mapping[str, Any]] = None,
) -> Dict[str, bool]:
    """Best-effort inference for legacy checkpoints when flags are missing."""
    canonical = dict(canonical_hparams or {})
    use_multitask = canonical.get("use_multitask")
    use_cnn = canonical.get("use_cnn")
    multitask_explicit = bool(canonical.get("use_multitask_explicit", False))
    use_cnn_explicit = bool(canonical.get("use_cnn_explicit", False))

    flat_keys = set()
    if checkpoint_params is not None:
        flat_keys = {"/".join(path) for path in flatten_dict(unfreeze(checkpoint_params)).keys()}

    if not multitask_explicit:
        use_multitask = any(
            key.startswith(("cop_head/", "grf_head/", "moments_head/", "contact_head/"))
            for key in flat_keys
        )
    if not use_cnn_explicit:
        use_cnn = any(
            key.startswith(("cnn_conv_0/", "cnn_conv_1/", "cnn_conv_2/"))
            for key in flat_keys
        )

    return {"use_multitask": bool(use_multitask), "use_cnn": bool(use_cnn)}


def resolve_checkpoint_config(
    raw_hparams: Mapping[str, Any],
    *,
    checkpoint_params: Optional[Mapping[str, Any]] = None,
    adapter_hidden_dim: Optional[int] = None,
    adapter_dropout_rate: Optional[float] = None,
) -> LosoCheckpointConfig:
    """Create a canonical config object from legacy/current checkpoint hyperparameters."""
    canonical = normalize_checkpoint_hyperparameters(raw_hparams)
    family = infer_checkpoint_family(checkpoint_params, canonical)

    input_dim = canonical.get("input_dim", 0)
    if int(input_dim) <= 0:
        raise ValueError(
            "Checkpoint hyperparameters do not contain a valid input_dim. "
            "Expected a sibling hyperparameters.json from train.py."
        )

    hidden_dim = adapter_hidden_dim
    if hidden_dim is None or int(hidden_dim) <= 0:
        hidden_dim = canonical.get("adapter_hidden_dim", 0)
    if hidden_dim is None or int(hidden_dim) <= 0:
        hidden_dim = canonical["ff_dim"]

    adapter_dr = adapter_dropout_rate
    if adapter_dr is None:
        adapter_dr = canonical.get("adapter_dropout_rate", 0.0)
    if adapter_dr is None or float(adapter_dr) <= 0.0:
        adapter_dr = canonical["dropout_rate"]

    return LosoCheckpointConfig(
        input_dim=int(input_dim),
        static_dim=int(canonical.get("static_dim", 8)),
        output_dim=int(canonical.get("output_dim", 14)),
        qfrc_inverse_output_dim=int(canonical.get("qfrc_inverse_output_dim", 0)),
        rotation_output_dim=int(canonical.get("rotation_output_dim", 6)),
        predict_jacobian=bool(canonical.get("predict_jacobian", False)),
        jacobian_output_dim=int(canonical.get("jacobian_output_dim", PREDICTED_JACOBIAN_FLAT_DIM)),
        d_model=int(canonical["d_model"]),
        num_heads=int(canonical["num_heads"]),
        num_layers=int(canonical["num_layers"]),
        ff_dim=int(canonical["ff_dim"]),
        dropout_rate=float(canonical["dropout_rate"]),
        use_film=bool(canonical.get("use_film", False)),
        use_cnn=bool(canonical["use_cnn"]) if canonical.get("use_cnn_explicit") else bool(family["use_cnn"]),
        cnn_num_layers=int(canonical["cnn_num_layers"]),
        cnn_kernel_sizes=tuple(int(v) for v in canonical["cnn_kernel_sizes"]) or (3, 5),
        use_multitask=bool(canonical["use_multitask"]) if canonical.get("use_multitask_explicit") else bool(family["use_multitask"]),
        trim_cop=bool(canonical["trim_cop"]),
        deviation_learning=bool(canonical["deviation_learning"]),
        include_pelvis_euler=bool(canonical["include_pelvis_euler"]),
        use_noised=bool(canonical["use_noised"]),
        noised_gt=bool(canonical["noised_gt"]),
        use_grf_norm_cop=bool(canonical["use_grf_norm_cop"]),
        cop_mask=bool(canonical["cop_mask"]),
        use_contact_weighting=bool(canonical["use_contact_weighting"]),
        mag_on_off=bool(canonical["mag_on_off"]),
        contact_on_off=bool(canonical["contact_on_off"]),
        contact_weight_multiplier=float(canonical["contact_weight_multiplier"]),
        mag_weight=float(canonical["mag_weight"]),
        cop_weight=float(canonical["cop_weight"]),
        grf_weight=float(canonical["grf_weight"]),
        moments_weight=float(canonical["moments_weight"]),
        contact_weight=float(canonical["contact_weight"]),
        torque_weight=float(canonical["torque_weight"]),
        jacobian_weight=float(canonical["jacobian_weight"]),
        grf_correction_weight=float(canonical["grf_correction_weight"]),
        output_reg_weight=float(canonical["output_reg_weight"]),
        best_model_by_torque=bool(canonical["best_model_by_torque"]),
        best_model_torque_weighting=dict(canonical["best_model_torque_weighting"]),
        adapter_hidden_dim=int(hidden_dim),
        adapter_dropout_rate=float(adapter_dr),
        model_structure=str(canonical.get("model_structure", "cop_grf_moments")),
    )


def load_sibling_hyperparameters(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the `hyperparameters.json` that should live next to a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    hyper_path = checkpoint_path.with_name("hyperparameters.json")
    if not hyper_path.exists():
        raise FileNotFoundError(
            f"Missing sibling hyperparameters.json for checkpoint: {checkpoint_path}"
        )
    try:
        with open(hyper_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read hyperparameters.json at {hyper_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a JSON object in {hyper_path}, got {type(data).__name__}")
    return dict(data)


def load_checkpoint_bundle(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the pickled checkpoint bundle saved by `train.py`."""
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, Mapping):
        return dict(bundle)
    return {"params": bundle}


def extract_checkpoint_params(checkpoint_bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the parameter pytree from a loaded checkpoint bundle."""
    if "params" in checkpoint_bundle:
        return checkpoint_bundle["params"]
    if "model_params" in checkpoint_bundle:
        return checkpoint_bundle["model_params"]
    variables = checkpoint_bundle.get("variables")
    if isinstance(variables, Mapping) and "params" in variables:
        return variables["params"]
    if any(
        key in checkpoint_bundle
        for key in (
            "Dense_0",
            "Dense_1",
            "TransformerBlock_0",
            "pre_transformer_adapter",
            "post_transformer_adapter",
            "cop_head",
            "predicted_jacobian_head",
        )
    ):
        flat = flatten_dict(unfreeze(checkpoint_bundle))
        frozen_only = {
            path: value
            for path, value in flat.items()
            if not _path_is_trainable(path, TRAINABLE_ADAPTER_ROOTS)
        }
        return freeze(unflatten_dict(frozen_only))
    raise KeyError(
        "Checkpoint bundle does not contain a params tree. Expected one of: "
        "'params', 'model_params', or variables['params']."
    )


def build_loso_model(
    config: Union[LosoCheckpointConfig, Mapping[str, Any]],
    checkpoint_params: Optional[Mapping[str, Any]] = None,
    *,
    adapter_hidden_dim: Optional[int] = None,
    adapter_dropout_rate: Optional[float] = None,
) -> nn.Module:
    """Construct the original checkpoint-compatible model for LOSO fine-tuning."""
    resolved = (
        config
        if isinstance(config, LosoCheckpointConfig)
        else _model_config_from_mapping(
            config,
            checkpoint_params,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
        )
    )
    kwargs = resolved.model_kwargs()
    # FiLM safety net: if the checkpoint params carry a top-level "film_mlp", the model
    # was trained with FiLM regardless of what the config says, so force it on to match.
    use_film = bool(kwargs.get("use_film", False))
    try:
        if isinstance(checkpoint_params, dict) and "film_mlp" in checkpoint_params:
            use_film = True
    except Exception:
        pass
    if str(getattr(resolved, "model_structure", "")).lower() == "direct_torque":
        if KinematicsToDirectTorque is None:
            raise ImportError(
                "Direct-torque checkpoint detected, but train_directTorque.KinematicsToDirectTorque "
                "could not be imported."
            )
        return KinematicsToDirectTorque(
            input_dim=kwargs["input_dim"],
            static_dim=kwargs["static_dim"],
            output_dim=kwargs["output_dim"],
            d_model=kwargs["d_model"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            ff_dim=kwargs["ff_dim"],
            dropout_rate=kwargs["dropout_rate"],
            use_film=use_film,
        )
    if resolved.use_multitask and MultiTaskKinematicsToCOPGRFMoments is not KinematicsToCOPGRFMoments:
        # Multitask model does not support FiLM; keep prior behavior.
        return MultiTaskKinematicsToCOPGRFMoments(
            input_dim=kwargs["input_dim"],
            static_dim=kwargs["static_dim"],
            output_dim=kwargs["output_dim"],
            d_model=kwargs["d_model"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            ff_dim=kwargs["ff_dim"],
            dropout_rate=kwargs["dropout_rate"],
        )
    return KinematicsToCOPGRFMoments(
        input_dim=kwargs["input_dim"],
        static_dim=kwargs["static_dim"],
        output_dim=kwargs["output_dim"],
        d_model=kwargs["d_model"],
        num_heads=kwargs["num_heads"],
        num_layers=kwargs["num_layers"],
        ff_dim=kwargs["ff_dim"],
        dropout_rate=kwargs["dropout_rate"],
        use_film=use_film,
    )


def prepare_loso_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    adapter_hidden_dim: Optional[int] = None,
    adapter_dropout_rate: Optional[float] = None,
) -> Tuple[LosoCheckpointConfig, nn.Module, Dict[str, Any], Mapping[str, Any]]:
    """Convenience loader for future LOSO entrypoints."""
    bundle = load_checkpoint_bundle(checkpoint_path)
    raw_hparams = load_sibling_hyperparameters(checkpoint_path)
    checkpoint_params = extract_checkpoint_params(bundle)
    config = resolve_checkpoint_config(
        raw_hparams,
        checkpoint_params=checkpoint_params,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_dropout_rate=adapter_dropout_rate,
    )
    model = build_loso_model(config)
    return config, model, bundle, checkpoint_params


def build_trainable_label_tree(
    params: Mapping[str, Any],
    *,
    trainable_roots: Sequence[str] = TRAINABLE_ADAPTER_ROOTS,
) -> Mapping[str, Any]:
    """Label only adapter leaves as trainable for Optax multi_transform."""
    flat_params = flatten_dict(unfreeze(params))
    labels_flat: Dict[Tuple[str, ...], str] = {}
    for path in flat_params.keys():
        labels_flat[path] = "train" if _path_is_trainable(path, trainable_roots) else "frozen"
    return freeze(unflatten_dict(labels_flat))


def describe_checkpoint_alignment(
    model_params: Mapping[str, Any],
    checkpoint_params: Mapping[str, Any],
    *,
    trainable_roots: Sequence[str] = TRAINABLE_ADAPTER_ROOTS,
) -> Dict[str, Sequence[str]]:
    """Return a compact diff between model params and checkpoint params."""
    model_flat = flatten_dict(unfreeze(model_params))
    checkpoint_flat = flatten_dict(unfreeze(checkpoint_params))

    expected_base_paths = {
        path for path in model_flat.keys() if not _path_is_trainable(path, trainable_roots)
    }
    checkpoint_paths = set(checkpoint_flat.keys())

    missing_base = sorted("/".join(path) for path in expected_base_paths - checkpoint_paths)
    unexpected_checkpoint = sorted("/".join(path) for path in checkpoint_paths - expected_base_paths)
    return {
        "missing_base_params": missing_base,
        "unexpected_checkpoint_params": unexpected_checkpoint,
    }


def merge_pretrained_params(
    init_params: Mapping[str, Any],
    checkpoint_params: Mapping[str, Any],
    *,
    strict: bool = True,
    trainable_roots: Sequence[str] = TRAINABLE_ADAPTER_ROOTS,
) -> Mapping[str, Any]:
    """Overlay checkpoint weights onto freshly initialized adapter parameters."""
    diff = describe_checkpoint_alignment(
        init_params,
        checkpoint_params,
        trainable_roots=trainable_roots,
    )
    if strict and (diff["missing_base_params"] or diff["unexpected_checkpoint_params"]):
        raise ValueError(
            "Checkpoint parameters do not align with the LOSO adapter model. "
            f"Missing base params: {diff['missing_base_params'][:6]} "
            f"Unexpected checkpoint params: {diff['unexpected_checkpoint_params'][:6]}"
        )

    def _merge(base: Any, source: Any) -> Any:
        if isinstance(base, Mapping) and isinstance(source, Mapping):
            merged: Dict[str, Any] = dict(base)
            for key, value in source.items():
                if key in merged:
                    merged[key] = _merge(merged[key], value)
            return merged
        return source

    merged = _merge(unfreeze(init_params), unfreeze(checkpoint_params))
    return freeze(merged)


def _checkpoint_family_name(use_multitask: bool) -> str:
    return "multitask" if use_multitask else "single_head"


class TransformerAdapterBlock(nn.Module):
    """Single trainable transformer block used as a frozen-backbone adapter."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        return TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=max(1, int(self.ff_dim)),
            dropout_rate=self.dropout_rate,
            name="TransformerBlock_0",
        )(x, train=train)


class _LosoAdapterBackboneMixin:
    """Shared stem/backbone helpers for the single-head and multitask LOSO models."""

    input_dim: int
    static_dim: int
    output_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    ff_dim: int
    dropout_rate: float
    use_cnn: bool
    cnn_num_layers: int
    cnn_kernel_sizes: Tuple[int, ...]
    adapter_hidden_dim: int
    adapter_dropout_rate: float

    def _stem_and_backbone(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Run the pretrained stem, adapters, and shared transformer backbone."""
        x = nn.Dense(self.d_model, name="Dense_0")(x)

        if self.use_cnn:
            residual = nn.gelu(x)
            kernels = list(self.cnn_kernel_sizes) if self.cnn_kernel_sizes else [3, 5]
            for i in range(self.cnn_num_layers):
                kernel = kernels[i] if i < len(kernels) else kernels[-1]
                x = nn.Conv(
                    features=self.d_model,
                    kernel_size=(kernel,),
                    strides=(1,),
                    padding="same",
                    name=f"cnn_conv_{i}",
                )(x)
                x = nn.gelu(x)

            alpha = self.param("cnn_gate", nn.initializers.ones, (1, 1, self.d_model))
            beta = self.param("res_gate", nn.initializers.ones, (1, 1, self.d_model))
            x = residual * beta + alpha * x
            x = nn.gelu(x)
            static_norm_name = "LayerNorm_0"
            final_norm_name = "LayerNorm_1"
        else:
            x = nn.LayerNorm(name="LayerNorm_0")(x)
            x = nn.gelu(x)
            static_norm_name = "LayerNorm_1"
            final_norm_name = "LayerNorm_2"

        x = TransformerAdapterBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.adapter_hidden_dim,
            dropout_rate=self.adapter_dropout_rate,
            name="pre_transformer_adapter",
        )(x, train=train)

        x = SinusoidalPosEmb(dim=self.d_model, name="SinusoidalPosEmb_0")(x)

        static_token = nn.Dense(self.d_model, name="Dense_1")(static_context)
        static_token = nn.gelu(static_token)
        static_token = nn.LayerNorm(name=static_norm_name)(static_token)

        static_token = jnp.expand_dims(static_token, axis=1)
        x = jnp.concatenate([static_token, x], axis=1)

        for layer_idx in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"TransformerBlock_{layer_idx}",
            )(x, train=train)

        x = x[:, 1:, :]
        x = nn.LayerNorm(name=final_norm_name)(x)
        x = TransformerAdapterBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.adapter_hidden_dim,
            dropout_rate=self.adapter_dropout_rate,
            name="post_transformer_adapter",
        )(x, train=train)
        return x


class LOSOAdapterKinematicsToCOPGRFMoments(_LosoAdapterBackboneMixin, nn.Module):
    """Single-head checkpoint-compatible LOSO model with trainable adapters only."""

    input_dim: int = 54
    static_dim: int = 8
    output_dim: int = 14
    predict_jacobian: bool = False
    jacobian_output_dim: int = PREDICTED_JACOBIAN_FLAT_DIM
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_cnn: bool = True
    cnn_num_layers: int = 2
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5)
    adapter_hidden_dim: int = 1024
    adapter_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self._stem_and_backbone(x, static_context, train=train)

        raw_out = nn.Dense(self.output_dim, name="Dense_2")(x)
        contact_prob = nn.sigmoid(raw_out[..., 12:14])
        output_parts = [
            raw_out[..., 0:4],
            raw_out[..., 4:10],
            raw_out[..., 10:12],
            contact_prob,
        ]
        if self.predict_jacobian:
            tail_dim = int(self.jacobian_output_dim or max(0, self.output_dim - STANDARD_OUTPUT_DIM))
            if tail_dim > 0:
                output_parts.append(raw_out[..., STANDARD_OUTPUT_DIM:STANDARD_OUTPUT_DIM + tail_dim])
        out = jnp.concatenate(output_parts, axis=-1)
        return out


class LOSOAdapterMultiTaskKinematicsToCOPGRFMoments(_LosoAdapterBackboneMixin, nn.Module):
    """Multitask checkpoint-compatible LOSO model with trainable adapters only."""

    input_dim: int = 54
    static_dim: int = 8
    output_dim: int = 14
    predict_jacobian: bool = False
    jacobian_output_dim: int = PREDICTED_JACOBIAN_FLAT_DIM
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_cnn: bool = True
    cnn_num_layers: int = 2
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5)
    adapter_hidden_dim: int = 1024
    adapter_dropout_rate: float = 0.1
    head_d_model: int = 128
    head_num_layers: int = 3
    head_num_heads: int = 4
    head_ff_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self._stem_and_backbone(x, static_context, train=train)

        cop_raw = TaskHead(
            head_d_model=self.head_d_model,
            head_num_layers=self.head_num_layers,
            head_num_heads=self.head_num_heads,
            head_ff_dim=self.head_ff_dim,
            output_dim=4,
            dropout_rate=self.dropout_rate,
            name="cop_head",
        )(x, train=train)
        grf_raw = TaskHead(
            head_d_model=self.head_d_model,
            head_num_layers=self.head_num_layers,
            head_num_heads=self.head_num_heads,
            head_ff_dim=self.head_ff_dim,
            output_dim=6,
            dropout_rate=self.dropout_rate,
            name="grf_head",
        )(x, train=train)
        moments_raw = TaskHead(
            head_d_model=self.head_d_model,
            head_num_layers=self.head_num_layers,
            head_num_heads=self.head_num_heads,
            head_ff_dim=self.head_ff_dim,
            output_dim=2,
            dropout_rate=self.dropout_rate,
            name="moments_head",
        )(x, train=train)
        contact_logits = TaskHead(
            head_d_model=self.head_d_model,
            head_num_layers=self.head_num_layers,
            head_num_heads=self.head_num_heads,
            head_ff_dim=self.head_ff_dim,
            output_dim=2,
            dropout_rate=self.dropout_rate,
            name="contact_head",
        )(x, train=train)
        predicted_jacobian_raw = None
        if self.predict_jacobian:
            tail_dim = int(self.jacobian_output_dim or max(0, self.output_dim - STANDARD_OUTPUT_DIM))
            if tail_dim > 0:
                predicted_jacobian_raw = TaskHead(
                    head_d_model=self.head_d_model,
                    head_num_layers=self.head_num_layers,
                    head_num_heads=self.head_num_heads,
                    head_ff_dim=self.head_ff_dim,
                    output_dim=tail_dim,
                    dropout_rate=self.dropout_rate,
                    name="predicted_jacobian_head",
                )(x, train=train)

        contact_prob = nn.sigmoid(contact_logits)
        output_parts = [cop_raw, grf_raw, moments_raw, contact_prob]
        if predicted_jacobian_raw is not None:
            output_parts.append(predicted_jacobian_raw)
        out = jnp.concatenate(output_parts, axis=-1)
        return out


def create_frozen_adapter_train_state(
    rng: jax.Array,
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    static_shape: Tuple[int, int],
    checkpoint_params: Mapping[str, Any],
    *,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    trainable_roots: Sequence[str] = TRAINABLE_ADAPTER_ROOTS,
    strict_checkpoint_match: bool = True,
) -> Tuple[train_state.TrainState, Mapping[str, Any]]:
    """Create a TrainState where only the adapter leaves are trainable."""
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    dummy_static = jnp.ones(static_shape, dtype=jnp.float32)

    variables = model.init(rng, dummy_input, dummy_static, train=False)
    init_params = variables["params"]
    merged_params = merge_pretrained_params(
        init_params,
        checkpoint_params,
        strict=strict_checkpoint_match,
        trainable_roots=trainable_roots,
    )
    label_tree = build_trainable_label_tree(merged_params, trainable_roots=trainable_roots)

    tx = optax.multi_transform(
        {
            "train": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate, weight_decay=weight_decay),
            ),
            "frozen": optax.set_to_zero(),
        },
        label_tree,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=merged_params,
        tx=tx,
    )
    return state, label_tree


def normalize_hyperparameters(
    raw_hparams: Mapping[str, Any],
    checkpoint_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper used by the LOSO CLI.

    Keeps missing `input_dim` / `static_dim` as `None` so the caller can infer
    them from a sampled loader batch when older hyperparameter files do not
    contain those fields.
    """
    canonical = normalize_checkpoint_hyperparameters(raw_hparams)
    checkpoint_meta = dict(checkpoint_metadata or {})
    raw_keys = set(dict(raw_hparams or {}).keys())
    if "use_cnn" in checkpoint_meta and "use_cnn" not in canonical:
        canonical["use_cnn"] = bool(checkpoint_meta["use_cnn"])
    if "use_multitask" in checkpoint_meta and "use_multitask" not in canonical:
        canonical["use_multitask"] = bool(checkpoint_meta["use_multitask"])
    checkpoint_params = checkpoint_meta.get("params")
    checkpoint_normalizers = checkpoint_meta.get("normalizers")
    if canonical.get("output_dim", 0) <= 0 and isinstance(checkpoint_params, Mapping):
        try:
            dense2_kernel = checkpoint_params.get("Dense_2", {}).get("kernel")
        except Exception:
            dense2_kernel = None
        if dense2_kernel is not None:
            canonical["output_dim"] = int(np.asarray(dense2_kernel).shape[-1])
    elif isinstance(checkpoint_params, Mapping):
        try:
            dense2_kernel = checkpoint_params.get("Dense_2", {}).get("kernel")
        except Exception:
            dense2_kernel = None
        if dense2_kernel is not None:
            checkpoint_output_dim = int(np.asarray(dense2_kernel).shape[-1])
            if checkpoint_output_dim != int(canonical.get("output_dim", checkpoint_output_dim)):
                canonical["output_dim"] = checkpoint_output_dim
    if isinstance(checkpoint_normalizers, Mapping):
        if not ({"qfrc_inverse_output_dim", "qfrcInverseOutputDim"} & raw_keys):
            try:
                qinv_norm = checkpoint_normalizers.get("qfrc_inverse")
                qinv_mean = getattr(qinv_norm, "mean", None)
                if qinv_mean is not None:
                    canonical["qfrc_inverse_output_dim"] = int(np.asarray(qinv_mean).shape[-1])
            except Exception:
                pass
        if not ({"rotation_output_dim", "rotationOutputDim"} & raw_keys):
            try:
                rot_norm = checkpoint_normalizers.get("rotation")
                rot_mean = getattr(rot_norm, "mean", None)
                if rot_mean is not None:
                    canonical["rotation_output_dim"] = int(np.asarray(rot_mean).shape[-1])
            except Exception:
                pass
        if not ({"jacobian_output_dim", "PredictedJacobianDim", "predicted_jacobian_dim", "jacobian_flat_dim"} & raw_keys):
            try:
                jac_norm = checkpoint_normalizers.get("jacobian")
                jac_mean = getattr(jac_norm, "mean", None)
                if jac_mean is not None:
                    canonical["jacobian_output_dim"] = int(np.asarray(jac_mean).shape[-1])
            except Exception:
                pass
    auxiliary_output_dim = (
        STANDARD_OUTPUT_DIM
        + int(canonical.get("qfrc_inverse_output_dim", 0))
        + int(canonical.get("rotation_output_dim", 0))
    )
    if bool(canonical.get("predict_jacobian", False)):
        checkpoint_output_dim = int(canonical.get("output_dim", auxiliary_output_dim))
        if checkpoint_output_dim >= auxiliary_output_dim:
            canonical["jacobian_output_dim"] = int(checkpoint_output_dim - auxiliary_output_dim)
    if canonical.get("input_dim", 0) <= 0:
        canonical["input_dim"] = None
    if canonical.get("static_dim", 0) <= 0:
        canonical["static_dim"] = None
    return canonical


def _model_config_from_mapping(
    config: Mapping[str, Any],
    checkpoint_params: Optional[Mapping[str, Any]] = None,
    *,
    adapter_hidden_dim: Optional[int] = None,
    adapter_dropout_rate: Optional[float] = None,
) -> LosoCheckpointConfig:
    canonical = dict(normalize_checkpoint_hyperparameters(config))
    dense0 = None
    dense1 = None
    if checkpoint_params is not None:
        try:
            dense0 = checkpoint_params.get("Dense_0", {}).get("kernel")
        except Exception:
            dense0 = None
        try:
            dense1 = checkpoint_params.get("Dense_1", {}).get("kernel")
        except Exception:
            dense1 = None
    if canonical.get("input_dim", 0) <= 0 and dense0 is not None:
        canonical["input_dim"] = int(np.asarray(dense0).shape[0])
    if canonical.get("static_dim", 0) <= 0 and dense1 is not None:
        canonical["static_dim"] = int(np.asarray(dense1).shape[0])
    return resolve_checkpoint_config(
        canonical,
        checkpoint_params=checkpoint_params,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_dropout_rate=adapter_dropout_rate,
    )


def create_loso_train_state(
    rng: jax.Array,
    model: nn.Module,
    checkpoint_params: Mapping[str, Any],
    *,
    input_shape: Tuple[int, int, int],
    static_shape: Tuple[int, int],
    learning_rate: float,
    weight_decay: float,
):
    """Create a full fine-tuning TrainState initialized from checkpoint params."""
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    dummy_static = jnp.ones(static_shape, dtype=jnp.float32)
    init_params = model.init(rng, dummy_input, dummy_static, train=False)["params"]
    merged_params = merge_pretrained_params(
        init_params,
        checkpoint_params,
        strict=True,
        trainable_roots=(),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=merged_params,
        tx=tx,
    )


def extract_adapter_params(params_or_state: Mapping[str, Any]) -> Mapping[str, Any]:
    params = params_or_state.get("params", params_or_state)
    flat = flatten_dict(unfreeze(params))
    adapter_only = {
        path: value
        for path, value in flat.items()
        if _path_is_trainable(path, TRAINABLE_ADAPTER_ROOTS)
    }
    return freeze(unflatten_dict(adapter_only))


def checkpoint_params_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    before_flat = flatten_dict(unfreeze(before))
    after_flat = flatten_dict(unfreeze(after))
    if set(before_flat.keys()) != set(after_flat.keys()):
        return False
    for path in before_flat.keys():
        if not np.array_equal(np.asarray(before_flat[path]), np.asarray(after_flat[path])):
            return False
    return True


def count_parameters(params: Mapping[str, Any]) -> int:
    """Count scalar parameters in a pytree."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(unfreeze(params)):
        try:
            total += int(np.prod(np.asarray(leaf).shape))
        except Exception:
            continue
    return total


__all__ = [
    "TRAINABLE_ADAPTER_ROOTS",
    "LosoCheckpointConfig",
    "LOSOAdapterKinematicsToCOPGRFMoments",
    "LOSOAdapterMultiTaskKinematicsToCOPGRFMoments",
    "TransformerAdapterBlock",
    "build_loso_model",
    "build_trainable_label_tree",
    "count_parameters",
    "create_loso_train_state",
    "create_frozen_adapter_train_state",
    "checkpoint_params_unchanged",
    "describe_checkpoint_alignment",
    "extract_adapter_params",
    "extract_checkpoint_params",
    "infer_checkpoint_family",
    "load_checkpoint_bundle",
    "load_sibling_hyperparameters",
    "merge_pretrained_params",
    "normalize_hyperparameters",
    "normalize_checkpoint_hyperparameters",
    "prepare_loso_model_from_checkpoint",
    "resolve_checkpoint_config",
    # Re-exported train.py helpers for LOSO CLI reuse.
    "Normalizer",
    "SinusoidalPosEmb",
    "TaskHead",
    "TransformerBlock",
    "compute_full_external_moments",
    "compute_tau_grf_from_predictions",
    "compute_total_loss",
    "normalize_batch",
]
