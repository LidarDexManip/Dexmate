"""Run one left Wuji Hand 2 on a fixed Vega 1U in MuJoCo.

The Vega URDF is converted by the repository's existing launcher. Its joints
are then removed (zero configuration welded) and its floating base fixed at
the launcher's 15 cm height. The hand is attached to the L_ee frame. This is a
simulation-only integration; it exposes no robot command transport.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco

from .simulation import MODEL_DIR, MujocoHand, run as run_teleop
from .sources import GloveSource, SyntheticSource


WORKSPACE = Path(__file__).resolve().parents[1]
VEGA_LAUNCHER = WORKSPACE / "scripts/launch_vega_1u_mujoco.py"
HAND_XML = MODEL_DIR / "left.xml"
OUTPUT = Path.home() / ".cache/dexmate-urdf/vega_1u/vega_1u_left_wuji.xml"


def _vega_build_function():
    spec = importlib.util.spec_from_file_location("dexmate_vega_launcher", VEGA_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Vega converter at {VEGA_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_mjcf


def _resolve_hand_meshes(hand_root: ET.Element) -> None:
    compiler = hand_root.find("compiler")
    meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
    base = (HAND_XML.parent / meshdir).resolve()
    for mesh in hand_root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        path = (base / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Wuji Hand mesh referenced by {HAND_XML} not found: {path}")
        mesh.set("file", str(path))


def build_attached_mjcf(output: Path = OUTPUT) -> Path:
    """Build an asset-resolved MJCF with a welded zero-pose Vega and left hand."""
    if not HAND_XML.is_file():
        raise FileNotFoundError(
            f"Pinned Wuji Hand 2 model missing: {HAND_XML}. Initialize third_party/wuji-description."
        )
    vega_path = Path(_vega_build_function()())
    root = ET.parse(vega_path).getroot()
    hand = ET.parse(HAND_XML).getroot()

    world = root.find("worldbody")
    robot = world.find("body[@name='base_link']") if world is not None else None
    if robot is None:
        raise ValueError(f"Converted Vega MJCF has no base_link body: {vega_path}")
    target = root.find(".//body[@name='L_ee']")
    if target is None:
        raise ValueError("Converted Vega MJCF has no L_ee wrist attachment body")

    # Keep the entire Vega at its URDF zero pose. Removing its generalized
    # coordinates also prevents accidental arm/base motion under gravity.
    for joint in list(robot.iter("joint")) + list(robot.iter("freejoint")):
        parent = next((node for node in robot.iter() if joint in list(node)), None)
        if parent is not None:
            parent.remove(joint)
    # Vega is a stationary visual reference in this milestone. Its collision
    # meshes overlap the mount and would push the simulated fingers around.
    for geom in robot.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    robot.set("pos", "0 0 0.15")
    actuators = root.find("actuator")
    if actuators is not None:
        root.remove(actuators)
    for section in ("keyframe",):
        existing = root.find(section)
        if existing is not None:
            root.remove(existing)

    _resolve_hand_meshes(hand)
    asset = root.find("asset")
    hand_asset = hand.find("asset")
    if asset is None or hand_asset is None:
        raise ValueError("Vega or Wuji MJCF has no asset section")
    existing_mesh_names = {item.get("name") for item in asset.findall("mesh")}
    for item in hand_asset:
        if item.tag == "mesh" and item.get("name") in existing_mesh_names:
            raise ValueError(f"Conflicting mesh asset name: {item.get('name')}")
        asset.append(item)

    hand_world = hand.find("worldbody")
    wrist = hand_world.find("body[@name='l_wrist']") if hand_world is not None else None
    if wrist is None:
        raise ValueError("Wuji Hand model has no l_wrist root body")
    wrist.set("name", "wuji_left_wrist")
    # L_ee's +Z points out of the arm, while Hand 2 fingers extend along
    # wrist -Z. Turn the hand around its local Y axis so fingers point out.
    wrist.set("quat", "0 0 1 0")
    # The hand's root <default> gives otherwise unspecified hinges armature.
    # Preserve it explicitly because the Vega scene has its own defaults.
    hand_joint_default = hand.find("./default/joint")
    if hand_joint_default is None:
        raise ValueError("Wuji Hand model has no default joint dynamics")
    for joint in wrist.iter("joint"):
        for attribute, value in hand_joint_default.attrib.items():
            joint.attrib.setdefault(attribute, value)
    target.append(wrist)
    # The upstream hand explicitly excludes wrist/proximal-finger contacts.
    # Without these entries the assembled scene starts with deep self contacts.
    hand_contacts = hand.find("contact")
    if hand_contacts is not None:
        contacts = root.find("contact")
        if contacts is None:
            contacts = ET.SubElement(root, "contact")
        for exclusion in hand_contacts:
            if exclusion.get("body1") == "l_wrist":
                exclusion.set("body1", "wuji_left_wrist")
            if exclusion.get("body2") == "l_wrist":
                exclusion.set("body2", "wuji_left_wrist")
            contacts.append(exclusion)
    if hand.find("actuator") is not None:
        actuator_section = ET.SubElement(root, "actuator")
        for actuator in hand.find("actuator"):
            actuator_section.append(actuator)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def validate_model(model: mujoco.MjModel) -> None:
    if model.nq != 20 or model.nv != 20 or model.nu != 20:
        raise ValueError(
            f"Expected fixed Vega plus exactly 20 Wuji hand coordinates/actuators; "
            f"got nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    for name in ("L_arm_j1", "L_arm_j7", "R_arm_j1", "Lift", "floating_base"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            raise ValueError(f"Vega joint {name} was not welded; expected stationary robot")
    if not math.isclose(model.opt.timestep, 0.002):
        raise ValueError(f"Expected 2 ms physics timestep, got {model.opt.timestep}")


def load_vega_hand(side):
    if side != "left":
        raise ValueError("The mounted Vega scene supports only the left hand")
    path = build_attached_mjcf()
    model = mujoco.MjModel.from_xml_path(str(path))
    validate_model(model)
    data = mujoco.MjData(model)
    hand = MujocoHand(model, data, "left")
    hand.initialize()
    return model, data, hand


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("glove", "synthetic"), default="glove")
    parser.add_argument("--glove-sn")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=positive_seconds)
    parser.add_argument("--report", help="Write final run metrics as JSON")
    parser.add_argument("--convert-only", action="store_true", help="Build and validate without opening a viewer")
    args = parser.parse_args(argv)
    if args.headless and args.duration is None and not args.convert_only:
        parser.error("--headless requires --duration")
    if args.input == "synthetic" and (args.glove_sn or args.user_id):
        parser.error("--glove-sn and --user-id apply only to glove input")
    try:
        if args.convert_only:
            path = build_attached_mjcf()
            model = mujoco.MjModel.from_xml_path(str(path))
            validate_model(model)
            print(f"Loaded fixed Vega 1U with left Wuji Hand 2: nq={model.nq} nv={model.nv} nu={model.nu}")
            return 0
        import wuji_sdk
        wuji_sdk.set_log_level("warn")
        source = (GloveSource(args.glove_sn, args.user_id, sdk=wuji_sdk) if args.input == "glove"
                  else SyntheticSource("left"))
        try:
            if source.side != "left":
                raise ValueError(f"This launcher requires a left glove; selected side is {source.side}")
            run_teleop(source, headless=args.headless, duration=args.duration,
                       report_path=args.report, scene_loader=load_vega_hand,
                       title="Vega 1U + left Wuji Hand 2 — Space: pause/resume",
                       camera_body="wuji_left_wrist", camera_distance=0.75,
                       render_hz=30)
        finally:
            source.close()
        return 0
    except KeyboardInterrupt:
        print("Stopped; glove connection closed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
