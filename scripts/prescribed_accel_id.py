#!/usr/bin/env python3
"""OpenSim inverse dynamics with MJX-prescribed velocities & accelerations.

The standard OpenSim ID pipeline re-derives joint velocity/acceleration from the
coordinates via a GCVSpline. This module instead hands OpenSim the EXACT MuJoCo/MJX
generalized velocities (``qvel_mjx``) and accelerations (``qacc_mjx``), so the only
remaining difference vs the MJX ID ground truth is the rigid-body dynamics engine
itself (inertia representation, etc.) -- "as close as possible".

It is computed with ``SimbodyMatterSubsystem.calcResidualForceIgnoringConstraints``,
which solves ``f_residual = M*udot + f_inertial(q,u) - f_applied``. ``f_applied`` is
built by hand (gravity on every body + measured GRF on the feet) because that operator
ignores forces otherwise present in the model.

Two subtleties that are easy to get wrong (and are handled here):
  * The CoordinateSet order is NOT the multibody mobility (u-vector) order, so each
    coordinate is mapped to its true u-index empirically (set one speed, see which U
    slot moves). Skipping this silently scrambles per-DOF results.
  * Applied body forces use the SimTK convention: SpatialVec(moment-about-body-origin,
    force), both expressed in Ground.

Returns moments mapped onto the 31-channel MJX layout (same convention as
``batch_opensim_inverse_dynamics.load_opensim_31ch``) so downstream metric code is shared.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import opensim as osim

from batch_opensim_inverse_dynamics import _OPENSIM_TO_MJX_IDX, read_storage_file

# OpenSim coordinate name -> MJX channel index. Positions, velocities and accelerations
# share indexing because the pelvis is modeled as six 1-DOF joints (no quaternion).
NAME_TO_MJX: dict[str, int] = dict(_OPENSIM_TO_MJX_IDX)
NAME_TO_MJX.update({
    "pelvis_tilt": 3, "pelvis_list": 4, "pelvis_rotation": 5,
    "pelvis_tx": 0, "pelvis_ty": 1, "pelvis_tz": 2,
})


def _vec3(a) -> "osim.Vec3":
    return osim.Vec3(float(a[0]), float(a[1]), float(a[2]))


def _grf_force_point_torque(res_dir: Path):
    """Read (force, point, free-torque) in Ground for each foot from ground_reaction.mot."""
    cols, rows = read_storage_file(res_dir / "ground_reaction.mot")
    g = np.asarray(rows, dtype=np.float64)
    gci = {c: i for i, c in enumerate(cols)}

    def side(tag):
        v, p, t = f"{tag}_ground_force_v", f"{tag}_ground_force_p", f"{tag}_ground_torque_"
        F = np.column_stack([g[:, gci[v + "x"]], g[:, gci[v + "y"]], g[:, gci[v + "z"]]])
        P = np.column_stack([g[:, gci[p + "x"]], g[:, gci[p + "y"]], g[:, gci[p + "z"]]])
        T = np.column_stack([g[:, gci[t + "x"]], g[:, gci[t + "y"]], g[:, gci[t + "z"]]])
        return F, P, T

    return {"R": side("R"), "L": side("L")}


def compute_prescribed_id_31ch(
    paths,
    right_body: str = "calcn_r",
    left_body: str = "calcn_l",
) -> tuple[np.ndarray, int]:
    """Return (T, 31) prescribed-acceleration ID moments in MJX channel layout, and T.

    Requires the regular ID pipeline to have produced OpenSimModel_NoPatel.osim and
    ground_reaction.mot in paths.output_dir, and pos_mjx/qvel_mjx/qacc_mjx in ProcessedData.
    """
    proc, res = paths.processed_dir, paths.output_dir
    pos = np.load(proc / "pos_mjx.npy").astype(np.float64)
    vel = np.load(proc / "qvel_mjx.npy").astype(np.float64)
    acc = np.load(proc / "qacc_mjx.npy").astype(np.float64)
    grf = _grf_force_point_torque(res)
    n = min(pos.shape[0], vel.shape[0], acc.shape[0], grf["R"][0].shape[0])

    model = osim.Model(str(res / "OpenSimModel_NoPatel.osim"))
    state = model.initSystem()
    coords = model.getCoordinateSet()
    ncoord = coords.getSize()
    names = [coords.get(i).getName() for i in range(ncoord)]
    matter = model.getMatterSubsystem()
    nu = state.getNU()
    gravity = np.array([model.getGravity().get(k) for k in range(3)])

    # bodies: (Body, mobilized-body index, mass, mass_center)
    bodyset = model.getBodySet()
    bodies = []
    for i in range(bodyset.getSize()):
        b = bodyset.get(i)
        mc = b.get_mass_center()
        bodies.append((b, int(b.getMobilizedBodyIndex()), b.getMass(),
                       np.array([mc.get(k) for k in range(3)])))
    nmb = max(mbi for _, mbi, _, _ in bodies) + 1
    feet = {"R": bodyset.get(right_body), "L": bodyset.get(left_body)}
    feet_mbi = {s: int(feet[s].getMobilizedBodyIndex()) for s in "RL"}

    # coordinate index -> mobility (u) index (empirical; CoordinateSet order != u order)
    umap = []
    for i in range(ncoord):
        for j in range(ncoord):
            coords.get(j).setSpeedValue(state, 0.0)
        coords.get(i).setSpeedValue(state, 1.0)
        U = state.getU()
        umap.append(next(k for k in range(nu) if abs(U.get(k) - 1.0) < 1e-9))
    for j in range(ncoord):
        coords.get(j).setSpeedValue(state, 0.0)

    mjx_idx = [NAME_TO_MJX[nm] for nm in names]
    zero_sv = osim.SpatialVec(osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0))
    applied_mob = osim.Vector(nu, 0.0)
    out = np.full((n, 31), np.nan, dtype=np.float64)

    for f in range(n):
        for i in range(ncoord):
            c = coords.get(i)
            c.setValue(state, float(pos[f, mjx_idx[i]]), False)
            c.setSpeedValue(state, float(vel[f, mjx_idx[i]]))
        model.realizeVelocity(state)

        # applied body forces: gravity (every body) + GRF (feet), moment about body origin, in Ground
        M = np.zeros((nmb, 3)); Fv = np.zeros((nmb, 3))
        for b, mbi, mass, mc in bodies:
            Op = b.getTransformInGround(state).p()
            O = np.array([Op.get(k) for k in range(3)])
            com = b.findStationLocationInGround(state, _vec3(mc))
            com = np.array([com.get(k) for k in range(3)])
            Fg = mass * gravity
            M[mbi] += np.cross(com - O, Fg); Fv[mbi] += Fg
        for s in "RL":
            F, P, Tq = grf[s]
            mbi = feet_mbi[s]
            Op = feet[s].getTransformInGround(state).p()
            O = np.array([Op.get(k) for k in range(3)])
            M[mbi] += np.cross(P[f] - O, F[f]) + Tq[f]; Fv[mbi] += F[f]
        applied_body = osim.VectorOfSpatialVec(nmb, zero_sv)
        for k in range(nmb):
            applied_body.set(k, osim.SpatialVec(_vec3(M[k]), _vec3(Fv[k])))

        udot = osim.Vector(nu, 0.0)
        for i in range(ncoord):
            udot.set(umap[i], float(acc[f, mjx_idx[i]]))
        resid = osim.Vector(nu, 0.0)
        matter.calcResidualForceIgnoringConstraints(state, applied_mob, applied_body, udot, resid)
        for i in range(ncoord):
            out[f, mjx_idx[i]] = resid.get(umap[i])

    return out, n
