"""MuJoCo must retain every object from each full-body GLB asset."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
import trimesh


class MeshConversionTests(unittest.TestCase):
    def test_multiple_objects_keep_faces_and_transforms(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/launch_vega_1u_mujoco.py'
        spec = importlib.util.spec_from_file_location('vega_launcher_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(), node_name='one')
        transform = np.eye(4)
        transform[0, 3] = 3
        scene.add_geometry(trimesh.creation.box(), node_name='two', transform=transform)
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/'multi.glb', Path(folder)/'merged.obj'
            source.write_bytes(scene.export(file_type='glb'))
            module.convert_mesh(source, destination)
            mesh = trimesh.load(destination, force='mesh')
            np.testing.assert_allclose(mesh.bounds, [[-.5,-.5,-.5],[3.5,.5,.5]])
            model = mujoco.MjModel.from_xml_string(
                f'<mujoco><asset><mesh name="body" file="{destination}"/></asset>'
                '<worldbody><geom type="mesh" mesh="body"/></worldbody></mujoco>')
            self.assertEqual(model.mesh_facenum[0], 24)
