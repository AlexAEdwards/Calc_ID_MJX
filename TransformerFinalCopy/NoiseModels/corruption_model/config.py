from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    paired_path: str = "Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects"
    mocap_only_path: str = "TrustedDatasetNoisedFromModel"
    output_path: str = "outputs/corruption_model"
    subject_metadata_filename: str = "Patient_MD.json"


@dataclass
class MetadataConfig:
    height_source: str = "patient_md"
    mass_source: str = "patient_md"


@dataclass
class RepresentationConfig:
    sample_rate_hz: float = 100.0
    dof_source: str = "patient_md"
    joint_names: List[str] = field(default_factory=list)
    normalize_anthropometrics: bool = False


@dataclass
class ModelConfig:
    activity: str = "walking"
    use_phase_conditioning: bool = True
    phase_window_frames: int = 50
    use_phase_residual: bool = True
    phase_residual_sample_scale: float = 0.2
    phase_residual_gain_std: float = 0.05
    use_lowrank: bool = True
    pca_components: int = 6
    explained_variance_threshold: float = 0.9
    use_ar1: bool = True
    use_dropout: bool = False
    use_smoothing: bool = True
    use_lag: bool = True
    lag_max_frames: int = 10
    smoothing_filter_order: int = 4
    smoothing_cutoff_hz_default: float = 6.0
    minimum_variance: float = 1e-6
    lowrank_sample_scale: float = 0.25
    lowrank_template_mix: float = 0.9
    lowrank_template_gain_std: float = 0.05
    lowrank_template_jitter_scale: float = 0.1
    noise_sample_scale: float = 0.06
    lag_std_scale: float = 0.05
    smoothing_std_scale: float = 0.1


@dataclass
class GenerationConfig:
    samples_per_trial: int = 8
    random_seed: int = 42


@dataclass
class ExportConfig:
    mode: str = "processeddata_subdir"
    shard_size: int = 64
    output_subdir_name: str = "ProcessedData_1"


@dataclass
class EvaluationConfig:
    split: str = "loso"
    save_plots: bool = True
    plots_max_trials: int = 5


@dataclass
class CorruptionConfig:
    data: DataConfig = field(default_factory=DataConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dataclass(instance: Any, values: Optional[Dict[str, Any]]) -> Any:
    if not values:
        return instance
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> CorruptionConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configs.")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = CorruptionConfig()
    config.data = _merge_dataclass(config.data, raw.get("data"))
    config.metadata = _merge_dataclass(config.metadata, raw.get("metadata"))
    config.representation = _merge_dataclass(config.representation, raw.get("representation"))
    config.model = _merge_dataclass(config.model, raw.get("model"))
    config.generation = _merge_dataclass(config.generation, raw.get("generation"))
    config.export = _merge_dataclass(config.export, raw.get("export"))
    config.evaluation = _merge_dataclass(config.evaluation, raw.get("evaluation"))
    return config
