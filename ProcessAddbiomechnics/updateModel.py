import xml.etree.ElementTree as ET
import mujoco
import mujoco.mjx as mjx
import jax.numpy as jnp
import numpy as np

# Hardcoded global flags
DoNotFixMassArmatureInertia = True

AXIS_A_R = np.array([-4.566e-07, -0.07071, -0.9975], dtype=float)
AXIS_B_R = np.array([-0.1243, 0.9898, -0.07016], dtype=float)
AXIS_C_R = np.array([0.9922, 0.124, -0.008789], dtype=float)
AXIS_A_L = np.array([-4.566e-07, -0.07071, 0.9975], dtype=float)
AXIS_B_L = np.array([-0.1243, 0.9898, 0.07016], dtype=float)
AXIS_C_L = np.array([-0.9922, -0.124, -0.008789], dtype=float)
KNEE_AXIS_R = AXIS_A_R
KNEE_AXIS_L = np.array([4.566e-07, 0.07071, -0.9975], dtype=float)
CANONICAL_KNEE_RANGE = "0 2.443"
CANONICAL_KNEE_SOLIMP = "0.9999 0.9999 0.001 0.5 2"
AXIS_LABELS_R = {"A": AXIS_A_R, "B": AXIS_B_R, "C": AXIS_C_R}
CANONICAL_KNEE_AXES = {
    "r": {
        "translation1": AXIS_A_R,
        "translation2": AXIS_B_R,
        "translation3": AXIS_C_R,
        "knee_angle": KNEE_AXIS_R,
        "rotation2": AXIS_C_R,
        "rotation3": AXIS_B_R,
    },
    "l": {
        "translation1": AXIS_A_L,
        "translation2": AXIS_B_L,
        "translation3": AXIS_C_L,
        "knee_angle": KNEE_AXIS_L,
        "rotation2": AXIS_C_L,
        "rotation3": AXIS_B_L,
    },
}
CANONICAL_ROTATION_POLYCOEFS = {
    ("r", "rotation2"): np.array([-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083], dtype=float),
    ("l", "rotation2"): np.array([-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083], dtype=float),
    ("r", "rotation3"): np.array([-4.43e-08, 0.3695, -0.1695, 0.02517, 0.0], dtype=float),
    ("l", "rotation3"): np.array([4.43e-08, -0.3695, 0.1695, -0.02517, 0.0], dtype=float),
}
CANONICAL_KNEE_JOINT_SUFFIXES = (
    "translation1", "translation2", "translation3", "knee_angle", "rotation2", "rotation3"
)

# Femur-length regressions from TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded
# GaitRetraining cohort (n=83), regenerated 2026-07-02. R2: A=0.971, B=1.000, C=0.908.
KNEE_POLYCOEF_SLOPES = {
    "A": np.array([0.0, 6.26442e-05, 4.31087e-03, -2.40897e-03, 3.77169e-04]),
    "B": np.array([0.0, 9.70695e-03, -2.82235e-02, 1.25264e-02, -1.42972e-03]),
    "C": np.array([0.0, 1.42490e-02, 1.39093e-03, -1.01289e-02, 2.74516e-03]),
}
KNEE_POLYCOEF_INTERCEPTS = {
    "A": np.array([0.0, 9.27305e-07, 6.36608e-05, -3.50869e-05, 5.59542e-06]),
    "B": np.array([0.0, -2.75591e-06, 1.20722e-05, -4.21431e-06, 5.82397e-07]),
    "C": np.array([0.0, 4.93216e-04, 4.81872e-05, -3.50370e-04, 9.60242e-05]),
}


def _parse_vec(text):
    return np.array([float(x) for x in str(text).split()], dtype=float)


def _fmt_vec(vec):
    return " ".join(f"{float(x):.8g}" for x in vec)


def _fmt_poly(vec):
    return " ".join(f"{float(x):.8g}" for x in vec)


def femur_length_from_model_xml(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    lengths = []
    for body_name in ("tibia_r", "tibia_l"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            length = float(np.linalg.norm(model.body_pos[body_id]))
            if np.isfinite(length) and length > 0.0:
                lengths.append(length)
    if not lengths:
        raise ValueError(f"Could not find tibia_r/tibia_l body offsets in {path}")
    return float(np.mean(lengths))


def _body_named(root, name):
    return root.find(f".//body[@name='{name}']")


def _axis_label_right(axis):
    for label, ref in AXIS_LABELS_R.items():
        if np.allclose(axis, ref, atol=1e-3):
            return label
    return None


def model_needs_knee_coupling_fix(root):
    tibia = _body_named(root, "tibia_r")
    if tibia is None:
        return False
    knee = tibia.find("./joint[@name='knee_angle_r']")
    if knee is None or knee.get("axis") is None:
        return False
    axis_a = _parse_vec(knee.get("axis"))
    translations = [
        j for j in tibia.findall("./joint")
        if (j.get("name") or "").startswith("walker_knee_r_translation")
    ]
    has_axis_a_translation = any(
        j.get("axis") is not None and np.allclose(_parse_vec(j.get("axis")), axis_a, atol=1e-3)
        for j in translations
    )
    return len(translations) < 3 or not has_axis_a_translation


def _infer_side_axes(root, side):
    tibia = _body_named(root, f"tibia_{side}")
    if tibia is None:
        raise ValueError(f"Could not find tibia_{side} body")
    knee = tibia.find(f"./joint[@name='knee_angle_{side}']")
    if knee is None or knee.get("axis") is None:
        raise ValueError(f"Could not find knee_angle_{side} axis")
    axes = {"A": _parse_vec(knee.get("axis"))}
    for joint in tibia.findall("./joint"):
        name = joint.get("name") or ""
        if not name.startswith(f"walker_knee_{side}_translation") or joint.get("axis") is None:
            continue
        if side == "r":
            label = _axis_label_right(_parse_vec(joint.get("axis")))
            if label in {"B", "C"}:
                axes[label] = _parse_vec(joint.get("axis"))
        else:
            # In the 39-DOF cohort the existing translations are ordered B, C.
            if name.endswith("translation1"):
                axes["B"] = _parse_vec(joint.get("axis"))
            elif name.endswith("translation2"):
                axes["C"] = _parse_vec(joint.get("axis"))
    missing = [label for label in ("A", "B", "C") if label not in axes]
    if missing:
        raise ValueError(f"Could not infer {side}-knee axes: missing {missing}")
    return axes


def _knee_polycoef(axis_label, femur_len, side):
    coef = KNEE_POLYCOEF_SLOPES[axis_label] * float(femur_len) + KNEE_POLYCOEF_INTERCEPTS[axis_label]
    if side == "l" and axis_label == "C":
        coef = -coef
    coef[0] = 0.0
    return coef


def _canonical_knee_joint_name(side, suffix):
    if suffix == "knee_angle":
        return f"knee_angle_{side}"
    return f"walker_knee_{side}_{suffix}"


def _canonical_knee_polycoef(side, suffix, femur_len):
    if suffix.startswith("translation"):
        label = {"translation1": "A", "translation2": "B", "translation3": "C"}[suffix]
        return _knee_polycoef(label, femur_len, side)
    return CANONICAL_ROTATION_POLYCOEFS[(side, suffix)].copy()


def _canonical_knee_joint_attrs(side, suffix, femur_len):
    jtype = "slide" if suffix.startswith("translation") else "hinge"
    poly = None if suffix == "knee_angle" else _canonical_knee_polycoef(side, suffix, femur_len)
    user = 0.0 if poly is None else float(poly[0])
    return {
        "name": _canonical_knee_joint_name(side, suffix),
        "range": CANONICAL_KNEE_RANGE,
        "limited": "true",
        "user": f"{user:.16g}",
        "ref": "0",
        "axis": _fmt_vec(CANONICAL_KNEE_AXES[side][suffix]),
        "type": jtype,
    }


def _knee_block_joint_names(side):
    return {_canonical_knee_joint_name(side, suffix) for suffix in CANONICAL_KNEE_JOINT_SUFFIXES}


def _canonical_knee_template_attrs(equality):
    template_attrs = None
    if equality is not None:
        for joint_eq in equality.findall("joint"):
            j1 = joint_eq.get("joint1") or ""
            j2 = joint_eq.get("joint2") or ""
            if "walker_knee_" in j1 and j2.startswith("knee_angle_"):
                template_attrs = {
                    k: v for k, v in joint_eq.attrib.items()
                    if k not in {"joint1", "joint2", "polycoef"}
                }
                break
    if template_attrs is None:
        template_attrs = {"solimp": CANONICAL_KNEE_SOLIMP, "active": "true"}
    else:
        template_attrs.setdefault("solimp", CANONICAL_KNEE_SOLIMP)
        template_attrs.setdefault("active", "true")
    return template_attrs


def _ranges_close(actual, expected=CANONICAL_KNEE_RANGE, atol=1e-6):
    try:
        actual_vals = [float(x) for x in str(actual).split()]
        expected_vals = [float(x) for x in str(expected).split()]
    except Exception:
        return False
    return len(actual_vals) == len(expected_vals) and np.allclose(actual_vals, expected_vals, atol=atol, rtol=0.0)


def _poly_close(actual, expected, atol=5e-6):
    try:
        vals = np.array([float(x) for x in str(actual).split()], dtype=float)
    except Exception:
        return False
    if vals.size < expected.size:
        vals = np.pad(vals, (0, expected.size - vals.size))
    return vals.size >= expected.size and np.allclose(vals[:expected.size], expected, atol=atol, rtol=0.0)


def _joint_qpos_count(joint):
    jtype = joint.get("type", "hinge")
    if jtype == "free":
        return 7
    if jtype == "ball":
        return 4
    return 1


def _worldbody_nq(root):
    worldbody = root.find("worldbody")
    if worldbody is None:
        return 0
    return sum(_joint_qpos_count(joint) for joint in worldbody.findall(".//joint"))


def rebuild_knee_coupling(root, femur_len):
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    template_attrs = _canonical_knee_template_attrs(equality)

    for side in ("r", "l"):
        tibia = _body_named(root, f"tibia_{side}")
        if tibia is None:
            raise ValueError(f"Could not find tibia_{side} body")

        names_to_rebuild = _knee_block_joint_names(side)
        children = list(tibia)
        insertion_idx = next(
            (
                i for i, child in enumerate(children)
                if child.tag == "joint" and (child.get("name") or "") in names_to_rebuild
            ),
            None,
        )
        if insertion_idx is None:
            insertion_idx = next(
                (i for i, child in enumerate(children) if child.tag != "joint"),
                len(children),
            )

        for joint in list(tibia.findall("./joint")):
            if (joint.get("name") or "") in names_to_rebuild:
                tibia.remove(joint)

        for offset, suffix in enumerate(CANONICAL_KNEE_JOINT_SUFFIXES):
            tibia.insert(
                insertion_idx + offset,
                ET.Element("joint", _canonical_knee_joint_attrs(side, suffix, femur_len)),
            )

        for joint_eq in list(equality.findall("joint")):
            j1 = joint_eq.get("joint1") or ""
            j2 = joint_eq.get("joint2") or ""
            if j1 in names_to_rebuild or j2 in names_to_rebuild:
                equality.remove(joint_eq)

        for suffix in ("translation1", "translation2", "translation3", "rotation2", "rotation3"):
            attrs = dict(template_attrs)
            attrs.update({
                "joint1": _canonical_knee_joint_name(side, suffix),
                "joint2": f"knee_angle_{side}",
                "polycoef": _fmt_poly(_canonical_knee_polycoef(side, suffix, femur_len)),
            })
            equality.append(ET.Element("joint", attrs))


def knee_coupling_is_canonical_root(root, femur_len, atol=1e-5):
    equality = root.find("equality")
    if equality is None:
        return False

    eq_by_joint1 = {}
    for eq in equality.findall("joint"):
        j1 = eq.get("joint1")
        if j1:
            eq_by_joint1[j1] = eq

    for side in ("r", "l"):
        tibia = _body_named(root, f"tibia_{side}")
        if tibia is None:
            return False

        joint_order = [
            joint.get("name") for joint in tibia.findall("./joint")
            if (joint.get("name") or "") in _knee_block_joint_names(side)
        ]
        expected_order = [_canonical_knee_joint_name(side, suffix) for suffix in CANONICAL_KNEE_JOINT_SUFFIXES]
        if joint_order != expected_order:
            return False

        for suffix in CANONICAL_KNEE_JOINT_SUFFIXES:
            name = _canonical_knee_joint_name(side, suffix)
            joint = tibia.find(f"./joint[@name='{name}']")
            if joint is None:
                return False
            expected_type = "slide" if suffix.startswith("translation") else "hinge"
            if joint.get("type", "hinge") != expected_type:
                return False
            if not _ranges_close(joint.get("range")):
                return False
            if joint.get("axis") is None or not np.allclose(
                _parse_vec(joint.get("axis")),
                CANONICAL_KNEE_AXES[side][suffix],
                atol=atol,
                rtol=0.0,
            ):
                return False

        for suffix in ("translation1", "translation2", "translation3", "rotation2", "rotation3"):
            name = _canonical_knee_joint_name(side, suffix)
            eq = eq_by_joint1.get(name)
            if eq is None or eq.get("joint2") != f"knee_angle_{side}":
                return False
            expected_poly = _canonical_knee_polycoef(side, suffix, femur_len)
            if not _poly_close(eq.get("polycoef", ""), expected_poly):
                return False
    return True


def knee_coupling_is_canonical_xml(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    femur_len = femur_length_from_model_xml(xml_path)
    return knee_coupling_is_canonical_root(root, femur_len)

def fix_xml_masses(xml_path, output_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Fix zero masses, small inertias, and small armatures directly in XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    print(f"\n{'='*70}")
    print(f"FIXING XML: {xml_path}")
    print(f"{'='*70}")

    # Update compiler to enforce minimum mass and inertia
    if not DoNotFixMassArmatureInertia:
        compiler = root.find('.//compiler')
        if compiler is not None:
            compiler.set('boundmass', str(min_mass))
            compiler.set('boundinertia', str(min_inertia))
            print(f"✓ Updated compiler:")
            print(f"    boundmass={min_mass}")
            print(f"    boundinertia={min_inertia}")
        else:
            print("⚠️  No compiler element found")

    # Mesh file paths should reference subject-specific Geometry/ folder
    print(f"\n1. Checking mesh geometry paths...")
    mesh_fixed = 0
    for mesh in root.findall('.//mesh'):
        file_path = mesh.get('file')
        # Ensure paths start with 'Geometry/' (subject-specific folder)
        if file_path and file_path.startswith('../Geometry/'):
            # Change from '../Geometry/' to 'Geometry/' (subject-specific)
            new_path = file_path.replace('../Geometry/', 'Geometry/')
            mesh.set('file', new_path)
            mesh_fixed += 1
        elif file_path and not file_path.startswith('Geometry/'):
            # If it has no prefix or wrong prefix, add 'Geometry/'
            if '/' in file_path:
                # Extract just the filename
                filename = file_path.split('/')[-1]
                mesh.set('file', f'Geometry/{filename}')
            else:
                mesh.set('file', f'Geometry/{file_path}')
            mesh_fixed += 1

    if mesh_fixed > 0:
        print(f"   ✓ Updated {mesh_fixed} mesh paths to use subject-specific 'Geometry/'")
    else:
        print(f"   ✓ All mesh paths already reference subject-specific 'Geometry/'")

    # Fix default joint armature
    if not DoNotFixMassArmatureInertia:
        print(f"\n2. Fixing default joint armature...")
        default_joint = root.find('.//default/joint')
        if default_joint is not None:
            current_armature = default_joint.get('armature')
            if current_armature is not None:
                current_val = float(current_armature)
                if current_val < min_armature:
                    default_joint.set('armature', str(min_armature))
                    print(f"   ✓ Updated default joint armature: {current_val:.2e} -> {min_armature:.2e}")
            else:
                default_joint.set('armature', str(min_armature))
                print(f"   ✓ Added default joint armature: {min_armature:.2e}")
        else:
            # Create default/joint element if it doesn't exist
            default_elem = root.find('.//default')
            if default_elem is None:
                default_elem = ET.SubElement(root, 'default')
            joint_elem = ET.SubElement(default_elem, 'joint')
            joint_elem.set('armature', str(min_armature))
            print(f"   ✓ Created default joint with armature: {min_armature:.2e}")
    else:
        print(f"\n2. Skipping default joint armature fix (DoNotFixMassArmatureInertia=True)")

    # Fix default geom collision settings (disable collisions and zero margin)
    print(f"\n3. Setting default geom collision properties...")
    default_elem = root.find('.//default')
    if default_elem is None:
        default_elem = ET.SubElement(root, 'default')
        print(f"   Created <default> element")

    default_geom = default_elem.find('geom')
    if default_geom is not None:
        # Update existing geom element
        default_geom.set('contype', '0')
        default_geom.set('conaffinity', '0')
        # Remove margin attribute so MJX doesn't generate mesh-mesh broad-phase pairs
        if 'margin' in default_geom.attrib:
            del default_geom.attrib['margin']
        default_geom.set('margin', '0')
        default_geom.set('gap', '0')
        print(f"   ✓ Updated default geom: contype=0, conaffinity=0, margin=0, gap=0 (collisions disabled)")
    else:
        # Create geom element
        geom_elem = ET.SubElement(default_elem, 'geom')
        geom_elem.set('contype', '0')
        geom_elem.set('conaffinity', '0')
        geom_elem.set('margin', '0')
        geom_elem.set('gap', '0')
        print(f"   ✓ Created default geom with: contype=0, conaffinity=0, margin=0, gap=0 (collisions disabled)")

    # Fix individual joint armatures
    if not DoNotFixMassArmatureInertia:
        print(f"\n4. Fixing individual joint armatures...")
        joint_armature_fixed = 0
        for joint in root.findall('.//joint'):
            armature_str = joint.get('armature')
            if armature_str is not None:
                armature = float(armature_str)
                if armature < min_armature:
                    joint.set('armature', str(min_armature))
                    joint_name = joint.get('name', 'unnamed')
                    print(f"   Fixed joint '{joint_name}': {armature:.2e} -> {min_armature:.2e}")
                    joint_armature_fixed += 1

        if joint_armature_fixed > 0:
            print(f"   ✓ Fixed {joint_armature_fixed} individual joints")
        else:
            print(f"   No individual joint armatures needed fixing")
    else:
        print(f"\n4. Skipping individual joint armatures fix (DoNotFixMassArmatureInertia=True)")

    # Fix body masses
    if not DoNotFixMassArmatureInertia:
        print(f"\n5. Fixing body masses...")
        mass_fixed = 0
        for inertial in root.findall('.//inertial'):
            mass_str = inertial.get('mass')
            if mass_str is not None:
                mass = float(mass_str)
                if mass < min_mass:
                    parent_body = inertial.find('..')
                    body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                    inertial.set('mass', str(min_mass))
                    print(f"   Fixed body '{body_name}': {mass:.6f} -> {min_mass:.6f}")
                    mass_fixed += 1

        if mass_fixed > 0:
            print(f"   ✓ Fixed {mass_fixed} bodies with small/zero mass")
        else:
            print(f"   No body masses needed fixing")
    else:
        print(f"\n5. Skipping body masses fix (DoNotFixMassArmatureInertia=True)")

    # Fix body inertias
    if not DoNotFixMassArmatureInertia:
        print(f"\n6. Fixing body inertias...")
        inertia_fixed = 0
        for inertial in root.findall('.//inertial'):
            # Check fullinertia attribute (6 values: Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
            fullinertia_str = inertial.get('fullinertia')
            if fullinertia_str is not None:
                values = [float(x) for x in fullinertia_str.split()]
                if len(values) >= 3:  # Check diagonal elements
                    if any(v < min_inertia for v in values[:3]):
                        # Fix diagonal elements
                        values[:3] = [max(v, min_inertia) for v in values[:3]]
                        inertial.set('fullinertia', ' '.join(str(v) for v in values))
                        parent_body = inertial.find('..')
                        body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                        print(f"   Fixed inertia for body '{body_name}'")
                        inertia_fixed += 1

            # Check diaginertia attribute (3 values: Ixx, Iyy, Izz)
            diaginertia_str = inertial.get('diaginertia')
            if diaginertia_str is not None:
                values = [float(x) for x in diaginertia_str.split()]
                if any(v < min_inertia for v in values):
                    values = [max(v, min_inertia) for v in values]
                    inertial.set('diaginertia', ' '.join(str(v) for v in values))
                    parent_body = inertial.find('..')
                    body_name = parent_body.get('name', 'unnamed') if parent_body is not None else 'unknown'
                    print(f"   Fixed inertia for body '{body_name}'")
                    inertia_fixed += 1

        if inertia_fixed > 0:
            print(f"   ✓ Fixed {inertia_fixed} bodies with small inertias")
        else:
            print(f"   No body inertias needed fixing")
    else:
        print(f"\n6. Skipping body inertias fix (DoNotFixMassArmatureInertia=True)")

    # Map all joint names to their qpos indices BEFORE we remove anything
    print(f"\n6.5. Mapping Joint Indices...")
    joint_qpos_indices = {}
    current_qpos_idx = 0

    # Only scan joints strictly inside worldbody (skipping defaults)
    worldbody = root.find('worldbody')
    if worldbody is not None:
        # Find all joints in document order within worldbody
        for joint in worldbody.findall(".//joint"):
            jname = joint.get('name')
            jtype = joint.get('type', 'hinge')

            # Determine size of qpos for this joint
            size = 1
            if jtype == 'free': size = 7
            elif jtype == 'ball': size = 4
            elif jtype == 'slide': size = 1
            elif jtype == 'hinge': size = 1

            # Store indices
            if jname:
                joint_qpos_indices[jname] = list(range(current_qpos_idx, current_qpos_idx + size))

            current_qpos_idx += size
        print(f"   ✓ Mapped {len(joint_qpos_indices)} joints to {current_qpos_idx} qpos indices")
    else:
        print("   ⚠️ Worldbody not found! Cannot map indices.")

    # Remove Patella Bodies and associated joints/geoms/constraints/contacts
    print(f"\n7. Removing Patella Bodies and Cleanup...")
    patella_bodies = ['patella_r', 'patella_l']
    removed_count = 0

    # Lists to store names of removed elements for cleanup
    removed_joint_names = []
    removed_geom_names = []
    removed_site_names = []

    # Find parent bodies (femur_r and femur_l)
    for parent_name in ['femur_r', 'femur_l']:
        parent_body = root.find(f".//body[@name='{parent_name}']")
        if parent_body is not None:
            # Find patella child body
            patella_name = parent_name.replace('femur', 'patella')
            patella_body = parent_body.find(f"./body[@name='{patella_name}']")

            if patella_body is not None:
                # Collect names of joints, geoms, and sites inside the patella body before removing it
                for joint in patella_body.findall(".//joint"):
                    removed_joint_names.append(joint.get('name'))
                for geom in patella_body.findall(".//geom"):
                    removed_geom_names.append(geom.get('name'))
                for site in patella_body.findall(".//site"):
                    removed_site_names.append(site.get('name'))

                print(f"   ✓ Removing body: {patella_name}")
                parent_body.remove(patella_body)
                removed_count += 1
            else:
                print(f"   ⚠️ Body {patella_name} not found under {parent_name}")
        else:
            print(f"   ⚠️ Parent body {parent_name} not found")

    print(f"\n7.5. Knee coupling check...")
    arm_joints_removed_count = 0
    femur_len = femur_length_from_model_xml(xml_path)
    if knee_coupling_is_canonical_root(root, femur_len):
        print(f"   ✓ Knee coupling already canonical OpenCap-style (femur={femur_len:.4f} m).")
    else:
        print(f"   Canonicalizing OpenCap-style knee coupling (femur={femur_len:.4f} m).")
        rebuild_knee_coupling(root, femur_len)
        print("   ✓ Rebuilt walker-knee translations, rotations, axes, ranges, and equality constraints.")

    # Always perform cleanup if any joints/bodies were removed
    if removed_count > 0 or arm_joints_removed_count > 0:
        # Cleanup Equality Constraints
        equality = root.find('equality')
        if equality is not None:
            print("   Cleaning up equality constraints...")
            joints_to_remove = []
            for joint_eq in equality.findall('joint'):
                j1 = joint_eq.get('joint1')
                j2 = joint_eq.get('joint2')
                # Check if either joint in the equality constraint was removed
                if (j1 in removed_joint_names) or (j2 in removed_joint_names):
                    joints_to_remove.append(joint_eq)
                # Also check for locked joints (single joint attribute)
                elif joint_eq.get('joint1') in removed_joint_names: # Some formats use joint1 for single joint constraints
                     joints_to_remove.append(joint_eq)

            for j in joints_to_remove:
                equality.remove(j)
                print(f"     - Removed equality constraint for joint: {j.get('joint1')}")

        # Cleanup Contact Pairs
        contact = root.find('contact')
        if contact is not None:
            print("   Cleaning up contact pairs...")
            pairs_to_remove = []
            for pair in contact.findall('pair'):
                g1 = pair.get('geom1')
                g2 = pair.get('geom2')
                if (g1 in removed_geom_names) or (g2 in removed_geom_names):
                    pairs_to_remove.append(pair)

            for p in pairs_to_remove:
                contact.remove(p)
                print(f"     - Removed contact pair: {p.get('geom1')} <-> {p.get('geom2')}")

        # Cleanup Tendon Sites
        tendon = root.find('tendon')
        if tendon is not None:
            print("   Cleaning up tendon sites...")
            for spatial in tendon.findall('spatial'):
                sites_removed_from_this_tendon = False
                sites_to_remove = []
                for site in spatial.findall('site'):
                    if site.get('site') in removed_site_names:
                        sites_to_remove.append(site)

                if sites_to_remove:
                    sites_removed_from_this_tendon = True
                    for s in sites_to_remove:
                        spatial.remove(s)
                        print(f"     - Removed site {s.get('site')} from tendon {spatial.get('name')}")

                # Check for wrapping geoms that might use removed sites as sidesites
                geoms_to_remove = []
                for geom in spatial.findall('geom'):
                    if geom.get('sidesite') in removed_site_names:
                        geoms_to_remove.append(geom)

                for g in geoms_to_remove:
                    spatial.remove(g)
                    print(f"     - Removed wrapping geom {g.get('geom')} from tendon {spatial.get('name')} (sidesite removed)")

                # SPECIAL FIX: If we removed sites from a tendon (like patella sites), the wrapping geom might be left unbracketed or invalid.
                # To ensure the model loads, we remove the wrapping geom from these modified tendons.
                if sites_removed_from_this_tendon:
                     geoms_to_remove = spatial.findall('geom')
                     for g in geoms_to_remove:
                         spatial.remove(g)
                         print(f"     - Removed wrapping geom {g.get('geom')} from tendon {spatial.get('name')} (simplified path due to site removal)")

    # Drop stale keyframes if structural edits changed nq. They are defaults only.
    keyframe = root.find('keyframe')
    if keyframe is not None:
        print("   Checking keyframes...")
        expected_nq = _worldbody_nq(root)
        drop_keyframe = False
        for key in keyframe.findall('key'):
            qpos_str = key.get('qpos')
            if qpos_str:
                actual_nq = len(qpos_str.split())
                if actual_nq != expected_nq:
                    print(f"     - Dropping <keyframe>: key '{key.get('name')}' qpos size {actual_nq} != edited nq {expected_nq}")
                    drop_keyframe = True
                    break
        if drop_keyframe:
            root.remove(keyframe)

    # Save modified XML
    tree.write(output_path)
    print(f"\n{'='*70}")
    print(f"✓ SAVED FIXED MODEL TO: {output_path}")
    print(f"{'='*70}")

    return output_path


def verify_fixes(xml_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Verify that the fixes were applied correctly."""
    import numpy as np

    print(f"\n{'='*70}")
    print(f"VERIFYING FIXES")
    print(f"{'='*70}")

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)

    # Check masses
    masses = model.body_mass[1:]  # Skip world body
    print(f"\nBody Masses:")
    print(f"  Min: {masses.min():.6f}")
    print(f"  Max: {masses.max():.6f}")

    if masses.min() < min_mass:
        print(f"  ⚠️  WARNING: Min mass {masses.min():.6f} < {min_mass}")
        zero_mass = np.where(masses < min_mass)[0] + 1
        print(f"     Problematic bodies: {zero_mass[:5]}")
    else:
        print(f"  ✓ All masses >= {min_mass}")

    # Check inertias
    inertias = model.body_inertia[1:]
    inertia_min = inertias[inertias > 0].min() if (inertias > 0).any() else 0
    print(f"\nBody Inertias:")
    print(f"  Min (non-zero): {inertia_min:.6e}")

    if inertia_min < min_inertia and inertia_min > 0:
        print(f"  ⚠️  WARNING: Min inertia {inertia_min:.6e} < {min_inertia}")
    else:
        print(f"  ✓ All inertias >= {min_inertia}")

    # Check armature
    print(f"\nDOF Armature:")
    print(f"  Min: {model.dof_armature.min():.6e}")
    print(f"  Max: {model.dof_armature.max():.6e}")

    if model.dof_armature.min() < min_armature:
        print(f"  ⚠️  WARNING: Min armature {model.dof_armature.min():.6e} < {min_armature}")
        small_armature = np.where(model.dof_armature < min_armature)[0]
        print(f"     Problematic DOFs: {small_armature[:5]}")
    else:
        print(f"  ✓ All armatures >= {min_armature}")

    print(f"\n{'='*70}")

    return model


# Complete workflow
def fix_and_load_model(xml_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """Complete workflow: fix XML, verify, and convert to MJX."""
    import numpy as np

    # Step 1: Fix XML
    fixed_xml_path = xml_path.replace('.xml', '_FIXED.xml')
    fix_xml_masses(xml_path, fixed_xml_path, min_mass, min_inertia, min_armature)

    # Step 2: Verify fixes
    model = verify_fixes(fixed_xml_path, min_mass, min_inertia, min_armature)

    # Step 3: Apply additional runtime fixes if needed
    print(f"\n{'='*70}")
    print(f"APPLYING RUNTIME FIXES")
    print(f"{'='*70}")

    # Ensure armature (sometimes doesn't transfer from XML)
    # if model.dof_armature.min() < min_armature:
    #     print(f"Applying runtime armature fix...")
    #     model.dof_armature[:] = np.maximum(model.dof_armature, min_armature)
    #     print(f"  ✓ Armature now: {model.dof_armature.min():.6e}")

    # Ensure masses
    # if model.body_mass.min() < min_mass:
    #     print(f"Applying runtime mass fix...")
    #     model.body_mass = np.maximum(model.body_mass, min_mass)
    #     print(f"  ✓ Mass now: {model.body_mass.min():.6f}")

    # Set solver options
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    model.opt.tolerance = 1e-6
    print(f"  ✓ Jacobian: SPARSE")
    print(f"  ✓ Tolerance: {model.opt.tolerance}")

    # Step 4: Convert to MJX
    print(f"\n{'='*70}")
    print(f"CONVERTING TO MJX")
    print(f"{'='*70}")

    mjx_model = mjx.put_model(model)

    print(f"MJX Model:")
    print(f"  Min body mass: {mjx_model.body_mass.min():.6f}")
    print(f"  Min armature: {mjx_model.dof_armature.min():.6e}")

    # Step 5: Post-MJX fixes if needed
    # if mjx_model.body_mass.min() < min_mass or mjx_model.dof_armature.min() < min_armature:
    #     print(f"\nApplying post-MJX fixes...")
    #     mjx_model = mjx_model.tree_replace({
    #         'body_mass': jnp.maximum(mjx_model.body_mass, min_mass),
    #         'body_inertia': jnp.maximum(mjx_model.body_inertia, min_inertia),
    #         'dof_armature': jnp.maximum(mjx_model.dof_armature, min_armature),
    #     })
    #     print(f"  ✓ Final min mass: {mjx_model.body_mass.min():.6f}")
    #     print(f"  ✓ Final min armature: {mjx_model.dof_armature.min():.6e}")

    # # Step 6: Test that it works
    # print(f"\n{'='*70}")
    # print(f"TESTING MODEL")
    # print(f"{'='*70}")

    # try:
    #     data = mjx.make_data(mjx_model)
    #     data = data.replace(
    #         qpos=jnp.zeros(mjx_model.nq),
    #         qvel=jnp.zeros(mjx_model.nv),
    #         qacc=jnp.zeros(mjx_model.nv)
    #     )

    #     # Test inverse dynamics
    #     data = mjx.inverse(mjx_model, data)
    #     print(f"  ✓ mjx.inverse() works!")

    #     # Test step
    #     data = mjx.step(mjx_model, data)
    #     print(f"  ✓ mjx.step() works!")

    #     print(f"\n{'='*70}")
    #     print(f"✓✓✓ MODEL READY TO USE ✓✓✓")
    #     print(f"{'='*70}")

    # except Exception as e:
    #     print(f"\n{'='*70}")
    #     print(f"✗✗✗ MODEL TEST FAILED ✗✗✗")
    #     print(f"{'='*70}")
    #     print(f"Error: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None, None, None

    return mjx_model, model, fixed_xml_path


def update_model(xml_path, min_mass=0.5, min_inertia=0.01, min_armature=0.1):
    """
    Main function to fix and load a MuJoCo model with MJX.

    Args:
        xml_path (str): Path to the input XML file
        min_mass (float): Minimum body mass (default: 0.5)
        min_inertia (float): Minimum body inertia (default: 0.01)
        min_armature (float): Minimum joint armature (default: 0.1)

    Returns:
        tuple: (mjx_model, mujoco_model, fixed_xml_path)
            - mjx_model: MJX model ready for simulation
            - mujoco_model: Standard MuJoCo model
            - fixed_xml_path: Path to the fixed XML file
            Returns (None, None, None) if model fails to load

    Example:
        >>> from updateModel import update_model
        >>> mjx_model, mj_model, xml_path = update_model(
        ...     "model.xml",
        ...     min_mass=0.25,
        ...     min_inertia=0.01,
        ...     min_armature=0.01
        ... )
    """
    import jax

    # Configure JAX if not already done
    try:
        jax.config.update("jax_enable_x64", True)
    except:
        pass  # Already set

    # Run the complete workflow
    return fix_and_load_model(xml_path, min_mass, min_inertia, min_armature)


# Usage Example
if __name__ == "__main__":
    import jax
    import numpy as np

    # Configure JAX
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU to avoid cuSolver issues

    print(f"User: AlexAEdwards")
    print(f"Date: 2025-11-19 18:11:59 UTC")
    print(f"JAX devices: {jax.devices()}")

    # Fix and load your model using the main function
    mjx_model, mj_model, fixed_xml_path = update_model(
        xml_path="Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml",
        min_mass=0.25,
        min_inertia=0.01,
        min_armature=0.01
    )

    if mjx_model is not None:
        print(f"\n✓ Model ready! Use 'mjx_model' for your simulations")
        print(f"✓ Fixed XML saved to: {fixed_xml_path}")
        print(f"\nModel stats:")
        print(f"  Bodies: {mjx_model.nbody}")
        print(f"  DOFs: {mjx_model.nv}")
        print(f"  Constraints: {mjx_model.neq}")
