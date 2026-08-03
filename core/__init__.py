"""Shared building blocks extracted from train.py (REFACTOR_PLAN.md Stage 5).

These modules hold code that more than one entry point needs. train.py still
re-exports every name it used to define, so nothing that imported from train
has to change; migrating those call sites is a later, separate step.
"""
