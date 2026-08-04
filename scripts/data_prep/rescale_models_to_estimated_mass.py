#!/usr/bin/env python3
"""Rescale each subject's MuJoCo (MyoSuite) XML models so their total mass equals
the GRF-estimated mass in Patient_MD.json.

Physics: changing total body mass while keeping segment geometry fixed = uniformly
scaling density by k. Then per body:  mass -> mass*k  and  inertia -> inertia*k
(inertia = integral r^2 dm, so it scales linearly with mass when geometry is fixed).
Center-of-mass positions (`pos`) are unchanged.

Per file, k = Mass_kg(estimated) / sum(current <inertial mass>). Computing the
denominator from each file's own inertial sum makes the output total EXACTLY the
estimated mass regardless of the file's starting total (e.g. the patella-bearing
MyosuiteModel.xml vs the patella-free _FIXED/_Runtime variants).

Only the <inertial> tags are rewritten (via targeted regex); all other bytes --
comments, keyframes, actuators, mesh refs, formatting -- are preserved exactly.
The original Patient_MD.Mass_kg_reported is what the models were built to; this
overwrites the XML in place. A report is written to outputs/mass_estimation/.
"""
from __future__ import annotations

import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "TrustedDataSetNoised12Distributed_AllPatients_EstimatedWeights",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "mass_estimation",
)
MODEL_FILES = ["MyosuiteModel.xml", "MyosuiteModel_FIXED.xml", "MyosuiteModel_Runtime.xml"]

INERTIAL_RE = re.compile(r"<inertial\b[^>]*?/>")
MASS_RE = re.compile(r'(\bmass=")([^"]+)(")')
FULLINERTIA_RE = re.compile(r'(\bfullinertia=")([^"]+)(")')
DIAGINERTIA_RE = re.compile(r'(\bdiaginertia=")([^"]+)(")')


def fmt(x: float) -> str:
    """Compact, high-precision float matching MuJoCo's tolerance; handles tiny inertias."""
    return f"{x:.10g}"


def sum_inertial_mass(text: str) -> float:
    total = 0.0
    for tag in INERTIAL_RE.findall(text):
        m = MASS_RE.search(tag)
        if m:
            total += float(m.group(2))
    return total


def scale_inertia_attr(tag: str, regex: re.Pattern, k: float) -> str:
    def repl(m):
        vals = [float(v) * k for v in m.group(2).split()]
        return m.group(1) + " ".join(fmt(v) for v in vals) + m.group(3)
    return regex.sub(repl, tag)


def rescale_file(path: str, target_mass: float) -> dict:
    text = open(path).read()
    m_old = sum_inertial_mass(text)
    if m_old <= 0:
        raise ValueError(f"{path}: no positive inertial mass found")
    k = target_mass / m_old

    def repl_inertial(match):
        tag = match.group(0)
        tag = MASS_RE.sub(lambda m: m.group(1) + fmt(float(m.group(2)) * k) + m.group(3), tag)
        tag = scale_inertia_attr(tag, FULLINERTIA_RE, k)
        tag = scale_inertia_attr(tag, DIAGINERTIA_RE, k)
        return tag

    new_text = INERTIAL_RE.sub(repl_inertial, text)
    with open(path, "w") as f:
        f.write(new_text)

    m_new = sum_inertial_mass(new_text)
    return dict(m_old=m_old, k=k, m_new=m_new)


def main() -> None:
    subjects = sorted(
        d for d in os.listdir(DATASET)
        if os.path.isdir(os.path.join(DATASET, d))
        and os.path.isfile(os.path.join(DATASET, d, "Patient_MD.json"))
    )

    rows = []
    n_files = 0
    for subj in subjects:
        sdir = os.path.join(DATASET, subj)
        md = json.load(open(os.path.join(sdir, "Patient_MD.json")))
        target = md.get("Mass_kg")
        if md.get("Mass_kg_est_source") != "GRF_estimated" or target is None:
            print(f"  SKIP (no estimated mass)  {subj}")
            continue
        for fn in MODEL_FILES:
            path = os.path.join(sdir, fn)
            if not os.path.isfile(path):
                continue
            r = rescale_file(path, float(target))
            rows.append(dict(subject=subj, file=fn, target_mass=float(target), **r))
            n_files += 1

    df = pd.DataFrame(rows)
    # verify every output totals the target mass
    df["abs_err_kg"] = (df.m_new - df.target_mass).abs()
    max_err = float(df.abs_err_kg.max())
    os.makedirs(OUT_DIR, exist_ok=True)
    report = os.path.join(OUT_DIR, "model_rescale_report.csv")
    df.to_csv(report, index=False)

    print(f"\nRescaled {n_files} XML files across {df.subject.nunique()} subjects.")
    print(f"Scale factor k: median {df.k.median():.4f}  range [{df.k.min():.4f}, {df.k.max():.4f}]")
    print(f"Max |output total - target| over all files: {max_err:.2e} kg  (should be ~0)")
    print("Per model variant:")
    print(df.groupby("file").agg(n=("k", "size"), median_k=("k", "median"),
                                 max_abs_err_kg=("abs_err_kg", "max")).round(6).to_string())
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
