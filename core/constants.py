"""Output-vector layout shared by the model, the losses and the decoders.

Single source of truth: duplicating STANDARD_OUTPUT_DIM or a slice across
modules would let them drift silently, which is a correctness bug rather than
an untidiness. Copied verbatim from train.py in REFACTOR_PLAN.md Stage 5,
including the constants these are derived from.
"""

STANDARD_OUTPUT_DIM = 14
COP_SLICE = slice(0, 4)
GRF_SLICE = slice(4, 10)
MOMENTS_SLICE = slice(10, 12)
CONTACT_SLICE = slice(12, 14)
ROTATION_RESIDUAL_FEET = 2
ROTATION_RESIDUAL_AXIS_DIM = 3
ROTATION_OUTPUT_DIM = ROTATION_RESIDUAL_FEET * ROTATION_RESIDUAL_AXIS_DIM
