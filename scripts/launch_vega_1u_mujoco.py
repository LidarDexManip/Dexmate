#!/usr/bin/env python3
"""Convert and launch the Dexmate Vega 1U URDF in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1] / "dexmate-urdf"
URDF_PATH = REPO_ROOT / "robots/humanoid/vega_1u/vega_1u.urdf"
CACHE_DIR = Path.home() / ".cache/dexmate-urdf/vega_1u"
RAW_MJCF_PATH = CACHE_DIR / "vega_1u.raw.xml"
MJCF_PATH = CACHE_DIR / "vega_1u.xml"


def convert_mesh(source: Path, destination: Path) -> None:
    import trimesh

    loaded = trimesh.load(source, force="scene")
    destination.parent.mkdir(parents=True, exist_ok=True)
    loaded.export(destination)


def add_environment(tree: ET.ElementTree) -> None:
    root = tree.getroot()
    root.insert(0, ET.Element("option", gravity="0 0 -9.81", integrator="RK4"))

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Converted MJCF is missing its asset or worldbody section")
    root_body = worldbody.find("body")
    if root_body is None or root_body.get("name") is None:
        raise ValueError("Converted MJCF is missing its root robot body")

    ET.SubElement(
        asset,
        "material",
        name="ground_material",
        rgba="0.16 0.18 0.21 1",
        reflectance="0.15",
    )
    ET.SubElement(
        asset,
        "material",
        name="base_material",
        rgba="0.28 0.32 0.38 1",
        reflectance="0.2",
    )
    base_visual = root.find(".//geom[@name='base_link_visual']")
    if base_visual is not None:
        base_visual.set("material", "base_material")
    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos="2 -2 3",
        dir="-1 1 -2",
        diffuse="0.9 0.9 0.9",
        specular="0.3 0.3 0.3",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        pos="0 0 -0.15",
        size="5 5 0.1",
        material="ground_material",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="overview",
        pos="2.4 -2.4 1.8",
        mode="targetbody",
        target=root_body.get("name"),
    )


def build_mjcf() -> Path:
    """Convert the URDF and make its GLB assets readable by MuJoCo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    converter = Path(sys.executable).with_name("urdf2mjcf")
    if not converter.exists():
        converter = Path(shutil.which("urdf2mjcf") or "")
    if not converter.exists():
        raise RuntimeError(
            "urdf2mjcf is not installed. Install with: "
            "python -m pip install -r requirements-wuji-sim.txt"
        )

    subprocess.run(
        [
            str(converter),
            str(URDF_PATH),
            "--output",
            str(RAW_MJCF_PATH),
            "--log-level",
            "30",
        ],
        check=True,
    )

    tree = ET.parse(RAW_MJCF_PATH)
    for mesh in tree.getroot().findall(".//mesh"):
        mesh_file = mesh.get("file")
        if not mesh_file:
            continue
        source = (URDF_PATH.parent / mesh_file).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Mesh referenced by URDF was not found: {source}")
        if source.suffix.lower() == ".glb":
            destination = CACHE_DIR / "meshes" / f"{source.stem}.obj"
            if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
                convert_mesh(source, destination)
            mesh.set("file", str(destination))
        else:
            mesh.set("file", str(source))

    add_environment(tree)
    tree.write(MJCF_PATH, encoding="utf-8", xml_declaration=True)
    return MJCF_PATH


def launch_viewer(mjcf_path: Path) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[1] = 0
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Convert and validate the MJCF without opening a viewer.",
    )
    args = parser.parse_args()

    mjcf_path = build_mjcf()
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    print(
        f"Loaded Vega 1U: bodies={model.nbody}, "
        f"joints={model.njnt}, actuators={model.nu}"
    )
    if not args.convert_only:
        launch_viewer(mjcf_path)


if __name__ == "__main__":
    main()
