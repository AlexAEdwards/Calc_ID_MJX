"""Data-backed checks against the staged fixture. Skipped if it is absent."""
import numpy as np
import pytest


def test_fixture_spans_all_three_mjx_widths(fixture_trials):
    widths = set()
    for t in fixture_trials:
        p = f"{t['training_data_path']}/pos_mjx.npy"
        widths.add(int(np.load(p, mmap_mode="r").shape[1]))
    assert widths >= {23, 33, 43}, f"fixture lost width coverage: {sorted(widths)}"


def test_fixture_spans_many_cohorts(fixture_trials):
    exps = {t.get("experiment", "") for t in fixture_trials}
    assert len(exps) >= 7, f"expected broad cohort coverage, got {sorted(exps)}"


def test_fixture_spans_two_orders_of_trial_length(fixture_trials):
    lengths = sorted(int(t["length"]) for t in fixture_trials)
    assert lengths[-1] / max(lengths[0], 1) > 20, f"length spread too narrow: {lengths}"


def test_undersized_trials_are_not_discovered(fixture_root, fixture_trials):
    """The 24-frame and 2-frame trials are staged but must stay below MIN_TRIAL_LENGTH."""
    staged = {p.parent.parent.name + "/" + p.parent.name
              for p in fixture_root.glob("*/*/Trial_*/ProcessedData")}
    found = {t["subject"] + "/" + t["trial"] for t in fixture_trials}
    assert "SUBJ72/Trial_3" not in found
    assert "PD_SUB10_off/Trial_7" not in found
    assert len(staged) > len(found), "fixture should contain trials discovery rejects"


def test_every_discovered_trial_builds_finite_targets(fixture_trials):
    from TransformerFinal.data_loader import TrialDataLoader
    from TransformerFinal.direct_torque_utils import build_direct_torque_targets
    cfg = dict(window_size=70, stride=16, batch_size=4, shuffle=False,
               prediction_margin_frames=20, drop_last=False, use_noised=True,
               noised_gt=True, allow_missing_noised=True, edge_mode="train",
               edge_trim_frames=20, include_pelvis_euler=False)
    checked = 0
    for t in fixture_trials:
        dl = TrialDataLoader([t], **cfg)
        b = next(iter(dl), None)
        if b is None:
            continue
        tgt = np.asarray(build_direct_torque_targets(b, xp_name="numpy"))
        assert tgt.shape[-1] == 14
        assert np.isfinite(tgt).all(), f"non-finite target in {t['subject']}/{t['trial']}"
        checked += 1
    assert checked >= 5, "expected most fixture trials to produce a batch"


def test_input_width_is_uniform_across_cohorts(fixture_trials):
    """A model trained on one cohort must be applicable to all of them."""
    from TransformerFinal.data_loader import TrialDataLoader
    cfg = dict(window_size=70, stride=16, batch_size=4, shuffle=False,
               prediction_margin_frames=20, drop_last=False, use_noised=True,
               noised_gt=True, allow_missing_noised=True, edge_mode="train",
               edge_trim_frames=20, include_pelvis_euler=False,
               include_ankle_heights=True, include_jacobian_input=True,
               include_auxiliary_denoising_inputs=True)
    widths = set()
    for t in fixture_trials:
        b = next(iter(TrialDataLoader([t], **cfg)), None)
        if b is not None:
            widths.add(int(np.asarray(b["input"]).shape[-1]))
    assert len(widths) == 1, f"input width varies across cohorts: {sorted(widths)}"
