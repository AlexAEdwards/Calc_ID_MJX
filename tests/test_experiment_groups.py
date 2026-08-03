"""Cohort assignment rules - the ordering here is load-bearing."""
import pytest

from TransformerFinal.experiment_groups import (
    DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS,
    NON_EXPERIMENT_DIR_NAMES,
    experiment_of_subject,
)


@pytest.mark.parametrize("subject,expected", [
    ("02", "Numeric"), ("20", "Numeric"),
    ("SUBJ01", "Stroke"), ("SUBJ105", "Stroke"),
    ("GaitRetraining_Subject103", "GaitRetraining"),
    ("GaitRetraining_SubjectR892", "GaitRetraining"),
    ("PD_SUB01_on", "PD"), ("PD_SUB26_off", "PD"),
    ("HOA059_M0", "Hip_OA"), ("HEA121_Marche", "Hip_OA"),
    ("OA1", "OA_Y"), ("Y21", "OA_Y"),
    ("S1", "S_GAH"), ("S_GAH_8", "S_GAH"),
])
def test_subject_maps_to_expected_cohort(subject, expected):
    assert experiment_of_subject(subject) == expected


def test_prefix_order_stroke_before_bare_s():
    """SUBJ* must be tested before S*, or every stroke subject lands in S_GAH."""
    assert experiment_of_subject("SUBJ40") == "Stroke"
    assert experiment_of_subject("S4") == "S_GAH"


def test_oa_means_older_adult_not_osteoarthritis():
    """OA* is Silder-2008 older adults; the hip-OA cohort is HOA*/HEA*."""
    assert experiment_of_subject("OA10") == "OA_Y"
    assert experiment_of_subject("HOA010_M0") == "Hip_OA"


def test_unknown_subject_is_unassigned():
    assert experiment_of_subject("ZZZ_unknown") is None
    assert experiment_of_subject("") is None


def test_quarantine_dirs_are_never_experiments():
    assert "UnwantedSubjects" in NON_EXPERIMENT_DIR_NAMES


def test_hip_oa_is_excluded_from_training_by_default():
    assert "Hip_OA" in DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS
