# Implementation Plan: Knee-Coupling Fix + Name-Based Mapping for ProcessData.py

> **Audience:** an LLM/engineer implementing this change. Self-contained — you should not
> need to re-derive the analysis below. Read the whole doc before editing.
> **Environment:** run everything in the `myoconverter` conda env, which is the only env with
> `jax` + `mujoco` + `mjx` + `opensim`:
> `/home/mobl/miniconda3/envs/myoconverter/bin/python3`
> `ProcessData.py` imports `jax` at module top, so it will not even import in other envs.
> **Repo root:** `/home/mobl/Documents/Classwork/BioSimClass/ClonedRepo/Calc_ID_MJX`

---

## 1. Objective

Change how `ProcessData.py` builds and uses the MuJoCo `*_FIXED.xml` models for the **Trusted
datasets** (`TrustedDataSetNoised12Distributed*`), and how it maps kinematics to `qpos`.

Five concrete requirements (from the project owner):

1. **`*_FIXED.xml` is always used**, regardless of which mode/argument is passed (including the
   legacy `--OC_Mocap` path), **unless** a new config flag `--DontUseFixed` is passed (then use
   the raw `MyosuiteModel.xml`).
2. The **only** things `_FIXED` does now are: **(a) remove the patella bodies**, and
   **(b) replace the coupled knee generalized coordinates** for models that are missing the
   third knee translation coupling (the "39-DOF" cohort). **Stop stripping arms. Stop removing
   `walker_knee_*_translation1`.** (Both were done by the old `fix_xml_masses` to normalize
   everything to 31 DOF — no longer wanted.)
3. **Name-based `qpos` mapping only.** Remove the hardcoded index table `NP_TO_QPOS` /
   `map_patient_to_qpos(..., qpos_size=31)`. Map each kinematics column to its `qpos` slot by
   **joint name** (`model.jnt_qposadr`).
4. **`--OpenCapVal` stays separate and untouched.** It is only run on
   `OpenCapWalkingTrunkSwaySubjects/` and datasets like it (51-DOF myoconverter models), never on
   the Trusted datasets. Its existing code path in `ProcessData.py` must keep working.
5. **Saved physics arrays use a canonical, name-ordered "lumbar-down" 33-DOF slice** (see §5),
   uniform across all subjects. Arms remain **in** the 43-DOF cohort models (their mass must
   contribute to ID; they are frozen at neutral because nothing drives them), but arm DOFs are
   **excluded from the saved arrays**.

**Deliverable end state:** every 39-DOF Trusted model gets a `*_FIXED.xml` whose knee coupled
coordinates have been replaced with GaitRetraining-derived, femur-scaled coupling (adding the
missing translation axis), and with the patella removed. All Trusted models process through a
single name-based mapping and produce uniform-width (33-DOF) saved physics arrays.

---

## 2. Background & key findings (why this is the plan)

### 2.1 Three model structures exist
Across all `TrustedDataSet*` folders there are exactly **2 raw model types**, plus the OpenCap
type (handled separately):

| Type | Count (all Trusted variants) | Arms? | Patella? | Knee translations present |
|---|---|---|---|---|
| **39-DOF** (e.g. `Y11`, `OA*`, `S*`, `S_GAH*`, numeric) | 322 | **No** | Yes (`patellofemoral_*`) | translation1(**axis B**), translation2(**axis C**) — **missing axis A** |
| **43-DOF** (e.g. `GaitRetraining_*`) | 854 | Yes (10) | No | translation1(**A**), translation2(**B**), translation3(**C**) — **complete** |
| **51-DOF** (OpenCap, separate) | 40 | Yes | Yes | all three (A,B,C) |

In the specific variant `TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded`: **82 subjects are
39-DOF, 188 are 43-DOF.** (The fix must be dataset-variant-agnostic — detect by structure, not by
folder name.)

The 39-DOF subject folder names in that variant are:
```
02 04 05 06 07 08 09 11 12 13 14 15 17 18 19 20
OA1 OA2 OA4 OA5 OA7 OA8 OA9 OA10 OA11 OA12 OA13 OA14 OA15 OA17 OA18 OA19 OA20 OA22 OA23 OA24 OA25
S1 S4 S6 S8 S10 S11 S13 S14 S15 S16 S19 S20 S21 S22
S_GAH_1 S_GAH_2 S_GAH_3 S_GAH_4 S_GAH_5 S_GAH_6 S_GAH_7 S_GAH_8 S_GAH_9 S_GAH_10
Y1 Y2 Y4 Y5 Y6 Y7 Y8 Y9 Y10 Y11 Y12 Y13 Y14 Y15 Y16 Y17 Y18 Y19 Y20 Y21 Y22
```
(Do **not** hardcode this list — detect structurally per §3.1.)

### 2.2 The missing DOF is real but genuinely zero at the source
The 39-DOF cohort is missing the knee translation along local OpenSim axis `1 0 0` (call it
**axis A**, which in MuJoCo global coords is `-4.566e-07 -0.07071 -0.9975`, identical to the
`knee_angle` hinge axis). In the **original OpenSim `.osim`** for the 39-DOF cohort, this axis is
defined as `Constant(0)` with an empty `<coordinates>` tag — i.e., it was authored as identically
zero, not lost by conversion. In the 43-DOF/51-DOF `.osim`, the same axis is a real non-zero
`PolynomialFunction` of `knee_angle`. So the two cohorts were built from **different generic knee
definitions**; this is a population/model-source difference, not a bug.

### 2.3 Decision: replace all 3 knee translations with femur-scaled GaitRetraining coupling
The project owner chose to **replace the 39-DOF cohort's knee translation coupling entirely** with
the GaitRetraining coupling, scaled per subject. Rationale and validation:
- The GaitRetraining translation coupling is a single universal shape per axis, scaled per subject
  by one geometric factor. That factor is predicted extremely well by **femur length alone**
  (R²=0.977; adding tibia/height barely improves it — keep it simple, use femur only).
- Per-coefficient linear fit of the MuJoCo `polycoef` vs femur (see §4) gives axis B R²=1.000,
  axis A R²=0.971, axis C R²=0.908.
- Left-knee polycoefs are the exact negation of right-knee polycoefs.

### 2.4 Name-based mapping is required (and safer)
Adding the third translation shifts every downstream `qpos` index (e.g. `knee_angle_r` moves from
index 11 → 12), and keeping arms changes DOF counts per cohort. The current hardcoded `NP_TO_QPOS`
table would then **silently place joint angles into the wrong DOFs**. Name-based mapping fixes this
for any model, and fails loudly (see §6 safeguards) if a joint name is missing rather than
mis-mapping silently.

---

## 3. Detection & data-gathering

### 3.1 Detect a "needs knee fix" (39-DOF-type) model — structural, not name-based
A model needs the knee-coupling replacement iff its `walker_knee_r` joint set does **not** already
contain a translation joint whose MuJoCo global axis matches **axis A**
(`-4.566e-07 -0.07071 -0.9975`, tolerance ~1e-3). Equivalently: it has only 2 walker-knee
translations instead of 3. Implement this as a helper that loads the model (or parses the XML) and
checks the axis vectors of `walker_knee_r_translation*` joints.

### 3.2 Femur length (the scale predictor)
Femur length = Euclidean norm of the `tibia_r` body `pos` attribute in the MuJoCo model (the knee
joint origin expressed in the femur frame):
```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(model_xml)
i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
femur_len = float(np.linalg.norm(m.body_pos[i]))
```
39-DOF cohort femur range is **0.337–0.503 m**; the GaitRetraining fit range was **0.347–0.461 m**,
so **12 of 82 subjects extrapolate mildly** (~3–9% beyond an edge). Acceptable given the strong
linear fit, but the batch report (§7) must flag which subjects extrapolated.

### 3.3 Axis vectors & joint attributes (universal across all subjects)
MuJoCo global axis vectors (right knee), confirmed identical across all sampled subjects/cohorts:
- **A** = `-4.566e-07 -0.07071 -0.9975`  (== `knee_angle` hinge axis)
- **B** = `-0.1243 0.9898 -0.07016`
- **C** = `0.9922 0.124 -0.008789`

**Left-knee axis vectors are NOT captured in this doc — extract them from a reference
GaitRetraining model at implementation time** (they may be mirrored). Recommended robust approach:
transplant the `walker_knee_l_translation1/2/3` joint elements (names, axes, `type="slide"`,
`ref`, `limited`) from a reference GaitRetraining `MyosuiteModel.xml`
(e.g. `GaitRetraining_Subject123`), then overwrite only the equality `polycoef`. The joint `range`
on these coupled slide joints is non-constraining (equality forces the value), so copying
GaitRetraining's `range="0 2.443"` verbatim is fine, or set it to the target's own
`knee_angle_r` range for cleanliness.

Reference right-knee joint block to replicate (from `GaitRetraining_Subject123/MyosuiteModel.xml`):
```xml
<joint name="walker_knee_r_translation1" range="0 2.443" limited="true" user="0.0" ref="0" axis="-4.566e-07 -0.07071 -0.9975" type="slide"/>
<joint name="walker_knee_r_translation2" range="0 2.443" limited="true" user="0.0" ref="0" axis="-0.1243 0.9898 -0.07016"    type="slide"/>
<joint name="walker_knee_r_translation3" range="0 2.443" limited="true" user="0.0" ref="0" axis="0.9922 0.124 -0.008789"     type="slide"/>
<joint name="knee_angle_r"               range="0 2.443" limited="true" user="0.0" ref="0" axis="-4.566e-07 -0.07071 -0.9975" type="hinge"/>
<joint name="walker_knee_r_rotation2"    range="0 2.443" limited="true" user="-1.47e-08" ref="0" axis="0.9922 0.124 -0.008789"     type="hinge"/>
<joint name="walker_knee_r_rotation3"    range="0 2.443" limited="true" user="-4.43e-08" ref="0" axis="-0.1243 0.9898 -0.07016"    type="hinge"/>
```
**Target naming convention after the fix:** the rebuilt 39-DOF knee must use
`translation1=axis A, translation2=axis B, translation3=axis C` (i.e., match GaitRetraining's
name→axis mapping) so the canonical save slice (§5) is name-consistent across cohorts.
The existing 39-DOF model names its two translations `translation1=B, translation2=C`; after the
fix they must become `translation1=A, translation2=B, translation3=C`. Simplest: remove the 39-DOF
model's existing walker-knee translation joints + their equality entries, then insert the 3 fresh
joints (in this order, before `knee_angle`) and 3 fresh equality entries.

`knee_angle`, `rotation2`, `rotation3` and their equalities are **left unchanged** — the two
cohorts' rotation couplings are already numerically near-identical (differ < 0.001 rad), and they
carry no per-subject scale. (You may optionally also overwrite the rotations for full consistency,
but it is a near-no-op and higher-risk XML surgery; leaving them is recommended.)

---

## 4. Scaling constants (femur → MuJoCo polycoef)

The MuJoCo equality constraint reads `q_slave = c0 + c1·θ + c2·θ² + c3·θ³ + c4·θ⁴`, where θ is
`knee_angle` (radians) and `polycoef="c0 c1 c2 c3 c4"`. See
`calculate_coupled_coordinates_automated` in `ProcessData.py` for the exact evaluation. Each
coefficient is a **linear function of femur length** (right knee). Left knee = negate all.

```
# right-knee polycoef[i] = SLOPE[axis][i] * femur_len + INTERCEPT[axis][i]
# (c0 is always exactly 0 for all axes)

AXIS A (global -4.566e-07 -0.07071 -0.9975):   # the MISSING one; R^2 ~ 0.971
  c0: slope= 0.0            intercept= 0.0
  c1: slope=+6.26442e-05    intercept=+9.27305e-07
  c2: slope=+4.31087e-03    intercept=+6.36608e-05
  c3: slope=-2.40897e-03    intercept=-3.50869e-05
  c4: slope=+3.77169e-04    intercept=+5.59542e-06

AXIS B (global -0.1243 0.9898 -0.07016):        # R^2 = 1.000 (pure proportional)
  c0: slope= 0.0            intercept= 0.0
  c1: slope=+9.70695e-03    intercept=-2.75591e-06
  c2: slope=-2.82235e-02    intercept=+1.20722e-05
  c3: slope=+1.25264e-02    intercept=-4.21431e-06
  c4: slope=-1.42972e-03    intercept=+5.82397e-07

AXIS C (global 0.9922 0.124 -0.008789):         # R^2 = 0.908
  c0: slope= 0.0            intercept= 0.0
  c1: slope=+1.42490e-02    intercept=+4.93216e-04
  c2: slope=+1.39093e-03    intercept=+4.81872e-05
  c3: slope=-1.01289e-02    intercept=-3.50370e-04
  c4: slope=+2.74516e-03    intercept=+9.60242e-05
```

**These constants are reproducible** — regenerate/verify with the script below (run in the
`myoconverter` env). Store the constants in code with a provenance comment (source dataset variant,
n=83 GaitRetraining subjects, date, R² values).

<details><summary>Reproduction script for the §4 constants</summary>

```python
import mujoco, re, glob, os
import numpy as np
AXES = {'A': (-4.566e-07, -0.07071, -0.9975),
        'B': (-0.1243, 0.9898, -0.07016),
        'C': (0.9922, 0.124, -0.008789)}
def axis_label(vec):
    v = np.array([float(x) for x in vec.split()])
    for k, a in AXES.items():
        if np.allclose(v, a, atol=1e-3): return k
    return 'OTHER'
def parse_knee(xmlpath, side='r'):
    xml = open(xmlpath).read()
    body = re.search(rf'<body name="tibia_{side}".*?(?=<body name="talus_{side}")', xml, re.S).group(0)
    jaxis = {m.group(1): m.group(2) for m in
             re.finditer(r'<joint name="(walker_knee_%s_\w+)"[^>]*axis="([^"]+)"[^>]*/>' % side, body)}
    eq = re.search(r'<equality>.*?</equality>', xml, re.S).group(0)
    poly = {m.group(1): [float(x) for x in m.group(2).split()] for m in
            re.finditer(rf'joint1="(walker_knee_{side}_\w+)" joint2="knee_angle_{side}" polycoef="([^"]+)"', eq)}
    return {axis_label(ax): poly[j] for j, ax in jaxis.items() if j in poly and 'translation' in j}
def femur(mx):
    m = mujoco.MjModel.from_xml_path(mx)
    return float(np.linalg.norm(m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'tibia_r')]))
root = 'TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded'
gr = sorted(set(p.split('/')[1] for p in glob.glob(f'{root}/GaitRetraining_*/MyosuiteModel.xml')))
femurs, coeffs = [], {'A': [], 'B': [], 'C': []}
for s in gr:
    mx = f'{root}/{s}/MyosuiteModel.xml'; kn = parse_knee(mx)
    if not all(k in kn for k in 'ABC'): continue
    femurs.append(femur(mx))
    for k in 'ABC': coeffs[k].append(kn[k])
femurs = np.array(femurs)
for k in 'ABC':
    C = np.array(coeffs[k])
    for i in range(5):
        A = np.vstack([femurs, np.ones_like(femurs)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, C[:, i], rcond=None)
        print(k, i, slope, intercept)
```
</details>

---

## 5. Canonical 33-DOF save slice (name-ordered)

After computing full-model-width physics arrays, extract these **33 DOFs by joint name** (via
`model.jnt_dofadr`; for these models `nq == nv`, so `jnt_qposadr == jnt_dofadr`) before saving. This
is identical across the fixed 39-DOF (33 total) and 43-DOF (arms dropped) cohorts, giving uniform
width. Order = leg+lumbar `qpos` order of the models:

```
 0 pelvis_tx           11 walker_knee_r_translation3   22 walker_knee_l_translation2
 1 pelvis_ty           12 knee_angle_r                 23 walker_knee_l_translation3
 2 pelvis_tz           13 walker_knee_r_rotation2       24 knee_angle_l
 3 pelvis_tilt         14 walker_knee_r_rotation3       25 walker_knee_l_rotation2
 4 pelvis_list         15 ankle_angle_r                26 walker_knee_l_rotation3
 5 pelvis_rotation     16 subtalar_angle_r             27 ankle_angle_l
 6 hip_flexion_r       17 mtp_angle_r                  28 subtalar_angle_l
 7 hip_adduction_r     18 hip_flexion_l                29 mtp_angle_l
 8 hip_rotation_r      19 hip_adduction_l              30 lumbar_extension
 9 walker_knee_r_translation1  20 hip_rotation_l       31 lumbar_bending
10 walker_knee_r_translation2  21 walker_knee_l_translation1  32 lumbar_rotation
```

Apply this slice (by name) to every saved array indexed by DOF: `pos_mjx`, `qvel_mjx`, `qacc_mjx`,
`ID_GT_MJX`, `qfrc_inverse`, `qfrc_grf_contribution`, and the `nv` axis of the `Jacobian` (jacp/jacr
columns). COP/GRF/Moment arrays are not DOF-indexed and are unaffected.

**Implementation note:** the arm DOFs (present only in the 43-DOF cohort) are left in the model so
their mass contributes to `mj_inverse`; they are driven to neutral (0 pos/vel/acc) by the
name-based mapping because nothing maps to them (the "frozen above lumbar" behavior). They are then
dropped by this slice. The fixed 39-DOF cohort has no arms, so nothing is dropped for them.

---

## 6. Implementation steps

### Step A — `ProcessAddbiomechnics/updateModel.py` : `fix_xml_masses`
(Resolved in Stage 7: the repo-root near-duplicate has been deleted, and this file — the one
`ProcessData.py` actually imports — is now tracked. There is one copy.)

1. **Add module-level constants**: the §4 regression tables, the §3.3 axis vectors, and a
   `femur_length_from_model_xml(path)` helper.
2. **Add `rebuild_knee_coupling(root, femur_len)`** (operates on the parsed `ElementTree` root):
   - For each side (`r`, `l`): locate the `tibia_{side}` body; remove its existing
     `walker_knee_{side}_translation*` joints; insert 3 new translation joints
     (`translation1`=A, `translation2`=B, `translation3`=C) immediately **before**
     `knee_angle_{side}`, using the reference joint attributes (§3.3) and side-correct axes.
   - In `<equality>`: remove existing `walker_knee_{side}_translation*` constraints; add 3 new
     `<joint joint1="walker_knee_{side}_translationN" joint2="knee_angle_{side}"
     polycoef="c0 c1 c2 c3 c4" .../>` where the polycoef is computed from femur (§4), negated for
     the left side. Copy `solimp`/`active` attributes from an existing knee equality entry.
   - Do **not** touch `knee_angle`, `rotation2`, `rotation3` or their equalities.
3. **Modify the body of `fix_xml_masses`**:
   - **Keep** the patella-removal block (§ "7. Removing Patella Bodies") and its equality/contact/
     tendon/keyframe cleanup.
   - **Delete/skip** the arm-removal block ("7.5 Removing Arm Joints") — arms stay.
   - **Delete/skip** the `if current_qpos_idx == 43: ... translation1` removal — knee translations
     stay.
   - After patella removal, **if the model needs the knee fix (§3.1)**, compute `femur_len` and call
     `rebuild_knee_coupling`. (Compute femur from the model geometry; the tibia body `pos` is not
     affected by patella removal.)
   - The keyframe-fixup logic must still run and now must account for the **added** translation
     joints as well as removed patella joints. Simplest robust approach: after all structural edits,
     if a `<keyframe>` exists and its `qpos` length no longer equals the new model `nq`,
     **drop the `<keyframe>` entirely** (it is only a default pose and is not needed for
     processing; MuJoCo will default to zeros). This avoids fragile index bookkeeping. (During this
     session we already confirmed dropping the keyframe is safe for loading.)
   - `DoNotFixMassArmatureInertia` stays `True` (mass/inertia/armature fixing remains disabled).

### Step B — `ProcessData.py` : name-based qpos mapping (remove hardcoded table)
- Delete `NP_TO_QPOS` (~line 267) and rewrite `map_patient_to_qpos` (~line 841) to be name-based.
  New signature e.g. `map_patient_to_qpos(pos_row, model, pos_columns)` or, better, build the qpos
  **matrix** in the core with a precomputed name→index map. The 23 kinematics columns (the
  `POS_COLUMNS` order used to build `Pos.npy`) are:
  ```
  pelvis_tilt pelvis_list pelvis_rotation pelvis_tx pelvis_ty pelvis_tz
  hip_flexion_r hip_adduction_r hip_rotation_r knee_angle_r ankle_angle_r subtalar_angle_r mtp_angle_r
  hip_flexion_l hip_adduction_l hip_rotation_l knee_angle_l ankle_angle_l subtalar_angle_l mtp_angle_l
  lumbar_extension lumbar_bending lumbar_rotation
  ```
  Map each to `model.jnt_qposadr[mj_name2id(model, JOINT, name)]`. Allocate `qpos` of width
  `model.nq`, leave all non-mapped DOFs (arms, coupled knee) at 0. Coupled DOFs are then filled by
  the existing `calculate_coupled_coordinates_automated` (already name/equality-based — no change).
- **Safeguards (fail loud, do not silently mis-map):** before mapping, assert every one of the 23
  names exists in the model and is a 1-DOF hinge/slide joint; assert the 23 map to 23 distinct
  indices. Raise a clear error naming any missing/duplicate joint. (Empirically all 1,216 Trusted +
  OpenCap models pass this, so this is a tripwire, not an expected failure.)
- Update the **call sites** (core at ~line 3787: `map_patient_to_qpos(pos[t])`; the OpenSim-filter
  path at ~line 5113 which already passes `nq`). Both must use the name-based path and pass the
  model.

### Step C — `ProcessData.py` : canonical 33-DOF save slice (§5)
- Add a module constant `CANONICAL_SAVE_DOF_NAMES` = the 33 names in §5 order.
- Add a helper that, given a model, returns the integer column indices
  `[jnt_dofadr[name] for name in CANONICAL_SAVE_DOF_NAMES]` (assert all present).
- Immediately before saving DOF-indexed arrays in the core (`pos_mjx`, `qvel_mjx`, `qacc_mjx`,
  `ID_GT_MJX`, `qfrc_inverse`, `qfrc_grf_contribution`, and the `nv` axis of `Jacobian`), slice to
  these 33 columns. All saved arrays become uniformly 33-wide.

### Step D — `ProcessData.py` : `_FIXED` always, add `--DontUseFixed`
- Add argparse flag `--DontUseFixed` (store_true) → `cfg["DontUseFixed"]`.
- In `resolve_subject_model_xml` (~line 2528): for the **non-OpenCapVal** path, always resolve to
  `MyosuiteModel_FIXED.xml`, building it from the raw model via `fix_xml_masses` if missing (or if
  a `--rebuild-fixed-models`-style flag is set — keep that behavior). If `cfg["DontUseFixed"]` is
  set, return the raw `MyosuiteModel.xml` instead. This must hold **regardless of mode**, including
  the legacy `--OC_Mocap` path (it currently has separate model handling — route it through the
  same `_FIXED` resolution). Leave the existing early `--OpenCapVal` branch (which returns the raw
  `MyosuiteModel_{MoCap,Video}.xml`) exactly as-is.
- The current `UsedFIXEDModels` cfg key can be repurposed: `DontUseFixed` ⇒ `UsedFIXEDModels=False`.

### Step E — leave `--OpenCapVal` alone
The `--OpenCapVal` code (dual MoCap/Video pass, lowercase-`trial_*` discovery, raw-model use,
vel/accel derivation, its own dir routing) was added earlier this session and must keep working
unchanged. It never touches the Trusted datasets. Do not route OpenCapVal through the new `_FIXED`
logic. (OpenCapVal has its own known open issues tracked separately — not in scope here.)

---

## 7. Validation (do before batch)

Run in the `myoconverter` env.

1. **Model build**: pick one 39-DOF subject (e.g. `Y11`). Delete any stale `_FIXED.xml`. Run
   `fix_xml_masses`. Assert the new `_FIXED.xml` **loads in MuJoCo**, has patella removed, has 3
   walker-knee translations (axes A,B,C) per side, and `nq == 33`.
2. **Coupling magnitude sanity**: drive `knee_angle` across a real gait range (0 → ~1.3 rad) and
   evaluate the 3 new translation polynomials; peak displacements should be on the order of a few mm
   (translations 2 & 3 the largest, ~3–6 mm; translation-A ~1–1.5 mm) and continuous. Compare the
   femur-scaled result to a real GaitRetraining subject of similar femur — should be close.
3. **Name-based mapping**: assert `knee_angle_r` lands in the correct `qpos` slot (index 12 in the
   33-DOF model), not a translation DOF.
4. **End-to-end**: process one full trial for a 39-DOF subject through `ProcessData.py`. Confirm
   saved arrays are 33-wide, `ID_GT_MJX` is finite and physiologically plausible (peak knee
   flexion/extension moment ~tens of N·m, **not** hundreds — hundreds means the ID harness is wrong,
   e.g. noisy finite-difference derivatives; the pipeline uses `gcv_derivatives` + chain-rule
   coupled derivatives for a reason). Compare qualitatively to a pre-change run if available.
5. **Regression on a 43-DOF subject**: confirm a `GaitRetraining_*` subject is detected as NOT
   needing the knee fix, keeps its arms, produces a 43-DOF `_FIXED` model, and still yields a
   33-wide canonical save slice.

## 8. Batch run & report
- Regenerate `_FIXED.xml` for all Trusted subjects (39-DOF get the knee fix + patella removal;
  43-DOF get patella-noop + arms kept). Note: all `_FIXED.xml` in
  `TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded` were **already deleted** this session
  (raw models intact), so they will regenerate on next run.
- Write a JSON/CSV report: per subject → detected type, femur length, whether it extrapolated the
  femur fit range (0.347–0.461), success/failure, resulting `nq`.
- **Reprocessing:** existing `ProcessedData/*.npy` were generated with the old 31-DOF layout and are
  now stale (different DOF layout/width). A full reprocess of the Trusted datasets is required after
  this change.

---

## 9. Downstream consequences (OUT OF SCOPE here — flag to owner)

The saved physics arrays move from **31-wide** to the new **33-wide canonical layout**, and joint
indices shift (`knee_angle_r` 11→12, lumbar 28–30→30–32, etc.). Downstream code that hardcodes the
old indices must be updated **separately**:

- `TransformerFinal/train.py` `dof_weights_dict` default (~line 2116) is keyed by integer DOF index
  in the **old 31-DOF** layout: `{6,7,11,14,15,17,18,22,25,26,28,29,30}`. In the new 33-DOF layout
  the same joints are `{6,7,12,15,16,18,19,24,27,28,30,31,32}`
  (hip_flex_r, hip_add_r, knee_r, ankle_r, subtalar_r, hip_flex_l, hip_add_l, knee_l, ankle_l,
  subtalar_l, lumbar_ext, lumbar_bend, lumbar_rot).
- `TransformerFinal/data_loader.py` loads `ID_GT_MJX.npy`/`qfrc_inverse.npy` and assumes a fixed
  width — verify it tolerates 33 (it should, but the per-DOF weighting must match the new layout).
- Any other consumer of `pos_mjx`/`qvel_mjx`/`qacc_mjx`/`Jacobian` widths.

---

## 10. Current repo state / gotchas
- Env: `myoconverter` conda env only (`/home/mobl/miniconda3/envs/myoconverter/bin/python3`).
- `TrustedDataSetNoised12Distributed*` dirs are **gitignored** (untouched by version control); raw
  `MyosuiteModel.xml` files are never mutated by the pipeline — only `*_FIXED.xml` are generated, so
  deleting/regenerating `_FIXED` is always safe.
- All `_FIXED.xml` in `TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded` were deleted this
  session; other variants may still have stale `_FIXED.xml` (regenerate them).
- `map_patient_to_qpos`, `NP_TO_QPOS`, `calculate_coupled_coordinates_automated`, `gcv_derivatives`,
  `_process_single_trial_processed_core`, `resolve_subject_model_xml` are all in `ProcessData.py`
  (line numbers approximate — search by name, the file is ~6.9k lines and has been edited this
  session).
- `calculate_coupled_coordinates_automated` is already generic (parses `<equality>` polynomials from
  whichever XML is used) — it will correctly fill the new translation3 with **no changes**.
- Keep changes additive/guarded; do not regress the default, `--OC_Mocap`, or `--OpenCapVal` paths
  beyond the intended behavior changes (always-`_FIXED`, no arm stripping, name-based mapping).
```
