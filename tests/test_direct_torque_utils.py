"""Direct-torque target construction and unit conversions."""
import numpy as np
import pytest

from TransformerFinal.direct_torque_utils import (
    DIRECT_TORQUE_NAMES,
    DIRECT_TORQUE_OUTPUT_DIM,
    ID_GT_COLUMN_BY_TARGET,
    bodyweight_height_norm_factor_from_static,
    direct_torque_percent_to_nm,
    finite_direct_torque_mask,
    is_direct_torque_hparams,
)

GRAVITY = 9.8067


def test_channel_set_is_fourteen_and_bilateral():
    assert DIRECT_TORQUE_OUTPUT_DIM == 14 == len(DIRECT_TORQUE_NAMES)
    right = [n for n in DIRECT_TORQUE_NAMES if n.endswith("_r")]
    left = [n for n in DIRECT_TORQUE_NAMES if n.endswith("_l")]
    assert len(right) == len(left) == 7
    assert [n[:-2] for n in right] == [n[:-2] for n in left], "channels must pair L/R"


def test_id_columns_reference_the_23dof_schema():
    """Knee adduction is computed from GRF, so it has no ID column."""
    assert set(ID_GT_COLUMN_BY_TARGET) == {
        n for n in DIRECT_TORQUE_NAMES if not n.startswith("knee_adduction")
    }
    assert all(0 <= c < 23 for c in ID_GT_COLUMN_BY_TARGET.values())


def test_percent_to_nm_roundtrip():
    static = np.array([[1.75, 70.0] + [0.0] * 6])
    pct = np.linspace(-5, 5, 14).reshape(1, 14)
    nm = np.asarray(direct_torque_percent_to_nm(pct, static, xp=np))
    factor = 70.0 * 1.75 * GRAVITY
    np.testing.assert_allclose(nm, pct / 100.0 * factor, rtol=1e-12)


def test_norm_factor_is_mass_times_height_times_g():
    static = np.array([[1.8, 80.0, 0, 0, 0, 0, 0, 0]])
    f = np.asarray(bodyweight_height_norm_factor_from_static(static, xp=np))
    np.testing.assert_allclose(f.ravel()[0], 80.0 * 1.8 * GRAVITY, rtol=1e-12)


def test_finite_mask_is_all_or_nothing_per_frame():
    a = np.ones((4, 14)); a[1, 3] = np.nan; a[2, 0] = np.inf
    m = np.asarray(finite_direct_torque_mask(a, xp=np))
    assert m.shape == (4, 1)
    assert m[0, 0] and m[3, 0]
    assert not m[1, 0] and not m[2, 0], "one bad channel invalidates the frame"


@pytest.mark.parametrize("h,expected", [
    ({"model_structure": "direct_torque"}, True),
    ({"model_type": "DirectTorque"}, True),
    ({"direct_torque_model": True}, True),
    ({"model_structure": "cop_grf"}, False),
    ({}, False),
])
def test_hparams_detection(h, expected):
    assert is_direct_torque_hparams(h) is expected
