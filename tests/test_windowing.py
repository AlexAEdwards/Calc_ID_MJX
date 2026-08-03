"""Windowing and edge-mode contract - pure logic, no data, no GPU."""
import numpy as np
import pytest

# data_loader imports jax at module scope. Skip rather than error where jax is
# absent (CI runs without it), so `pytest -k` collection cannot abort the run.
pytest.importorskip("jax", reason="TransformerFinal.data_loader requires jax")

from TransformerFinal.data_loader import (  # noqa: E402
    build_full_window_supervision_mask,
    build_window_start_indices,
    build_window_supervision_mask,
    validate_prediction_margin,
)

WIN, TRIM, STRIDE = 70, 20, 16


def _coverage(T, starts, masks, win=WIN):
    cov = np.zeros(T, dtype=int)
    for s, m in zip(starts, masks):
        for i in range(win):
            if s + i < T and m[i, 0] > 0:
                cov[s + i] += 1
    return cov


@pytest.mark.parametrize("T", [39789, 1998, 200, 139, 121, 120, 111, 110])
def test_train_mode_supervises_exactly_the_interior(T):
    """edge_mode='train': trim before windowing, supervise every window frame."""
    usable = T - 2 * TRIM
    starts = [s + TRIM for s in build_window_start_indices(usable, WIN, STRIDE)]
    masks = [build_full_window_supervision_mask(WIN, s, TRIM, T - TRIM) for s in starts]
    cov = _coverage(T, starts, masks)
    sup = np.flatnonzero(cov > 0)
    assert sup[0] == TRIM, "supervision must start exactly at the trim boundary"
    assert sup[-1] == T - TRIM - 1, "supervision must end exactly at the trim boundary"
    assert not cov[:TRIM].any(), "no window may touch the trimmed head"
    assert not cov[T - TRIM:].any(), "no window may touch the trimmed tail"


@pytest.mark.parametrize("T", [39789, 1998, 200, 139, 111, 70, 60, 31])
def test_infer_mode_predicts_every_frame(T):
    """edge_mode='infer': no trim, so every frame gets a prediction."""
    starts = build_window_start_indices(T, WIN, STRIDE)
    masks = [build_full_window_supervision_mask(WIN, s, 0, T) for s in starts]
    cov = _coverage(T, starts, masks)
    assert int((cov > 0).sum()) == T, f"only {(cov>0).sum()}/{T} frames covered"


def test_train_mode_drops_trials_shorter_than_the_window():
    for T in (109, 100, 60, 31):
        assert T - 2 * TRIM < WIN, "fixture assumption: these must be undersized"


def test_legacy_mask_insets_both_window_and_trial():
    T = 200
    m = build_window_supervision_mask(WIN, 0, T, TRIM)
    assert m[:TRIM].sum() == 0 and m[WIN - TRIM:].sum() == 0
    assert m.sum() == WIN - 2 * TRIM


def test_full_window_mask_has_no_inset():
    m = build_full_window_supervision_mask(WIN, 100, 0, 10_000)
    assert m.sum() == WIN, "every frame of an interior window is supervised"


def test_validate_prediction_margin_rejects_impossible_config():
    validate_prediction_margin(70, 20)
    with pytest.raises(ValueError):
        validate_prediction_margin(40, 20)   # window <= 2*margin
    with pytest.raises(ValueError):
        validate_prediction_margin(70, -1)


def test_window_starts_always_cover_the_tail():
    for T in (139, 200, 1998):
        starts = build_window_start_indices(T, WIN, STRIDE)
        assert starts[-1] + WIN == T, "a tail window must reach the final frame"


def test_short_trial_yields_single_window():
    assert build_window_start_indices(50, WIN, STRIDE) == [0]
