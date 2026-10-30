"""Conservative collision coverage for an unmeasured, stationary hand."""
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from trimesh.triangles import closest_point
from .collision_check import Clearance
from .vega_bimanual_sim import body_names


def hand_bound(model, side):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)['hand'])
    geometries, radius = set(), 0.0
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        reach = 0.0
        while body and body != root:
            reach += float(np.linalg.norm(model.body_pos[body]))
            for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('Inactive-hand envelope requires hinge-only fingers')
                # Rotation around an offset pivot can translate the body by at most 2|pivot|.
                reach += 2 * float(np.linalg.norm(model.jnt_pos[j]))
            body = int(model.body_parentid[body])
        if body == root:
            geometries.add(geom)
            radius = max(radius, reach + float(np.linalg.norm(model.geom_pos[geom])) + model.geom_rbound[geom])
    if root < 0 or not geometries or radius <= 0:
        raise ValueError('Missing inactive hand geometry')
    return root, float(radius), geometries


class ConvexPointDistance:
    """Point-to-convex-mesh distance; zero for points inside the mesh hull."""
    def __init__(self, vertices):
        self.hull = ConvexHull(np.asarray(vertices, dtype=np.float64))
        self.triangles = self.hull.points[self.hull.simplices]

    def distance(self, point):
        point = np.asarray(point, dtype=np.float64)
        planes = self.hull.equations
        if np.all(planes[:, :3] @ point + planes[:, 3] <= 1e-9):
            return 0.0
        points = closest_point(self.triangles, np.broadcast_to(point, (len(self.triangles), 3)))
        return float(np.min(np.linalg.norm(points - point, axis=1)))


class InactiveHandChecker:
    """Replace unknown fingers with a sphere bound, without using guessed joint angles.

    Only the selected arm moves. The inactive hand's existing contact with its own
    arm/table is not validated; all selected-arm/hand motion is checked against the
    entire conservative envelope. Bounding spheres are used only for broad-phase
    pruning. Candidate pairs use the distance to the active geometry's convex
    hull, retaining the complete inactive-hand envelope without inflating the arm.
    """
    def __init__(self, checker, active_side, inactive_sides):
        self.checker = checker
        self.model, self.data = checker.model, checker.data
        self.envelopes = {}
        self.unknown_hand_geoms = set()
        for side in inactive_sides:
            root, radius, geoms = hand_bound(self.model, side)
            self.envelopes[side] = (root, radius)
            self.unknown_hand_geoms.update(geoms)
        # Do not retain pair checks using an arbitrary zero pose for unknown fingers.
        keep = [not any(int(g) in self.unknown_hand_geoms for g in pair) for pair in checker._pairs]
        checker._pairs = checker._pairs[keep]
        checker._bound = checker._bound[keep]
        hand_root = body_names(active_side)['hand']
        prefix = 'L' if active_side == 'left' else 'R'
        self.active_geoms = []
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or ''
            if name.startswith(prefix + '_arm_') or name == hand_root or name.startswith(active_side[0] + '_'):
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    self.active_geoms.append(g)
        self.mesh_distances = {}
        if self.envelopes:
            for geom in self.active_geoms:
                mesh = int(self.model.geom_dataid[geom])
                if mesh in self.mesh_distances:
                    continue
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                try:
                    self.mesh_distances[mesh] = ConvexPointDistance(self.model.mesh_vert[start:start+count])
                except QhullError:
                    # Degenerate meshes retain the conservative sphere result.
                    self.mesh_distances[mesh] = None

    def set_fingers(self, side, angles):
        if side not in self.envelopes:
            self.checker.set_fingers(side, angles)

    def clearance(self, q):
        result = self.checker.clearance(q)
        for side, (root, radius) in self.envelopes.items():
            center = self.data.xpos[root]
            bounds = [(float(np.linalg.norm(self.data.geom_xpos[g] - center)) - radius -
                       self.model.geom_rbound[g], g) for g in self.active_geoms]
            for lower, geom in sorted(bounds):
                if lower >= result.distance_m:
                    break
                distance = self.mesh_distances[int(self.model.geom_dataid[geom])]
                if distance is None:
                    gap = lower
                else:
                    rotation = self.data.geom_xmat[geom].reshape(3, 3)
                    local_center = rotation.T @ (center - self.data.geom_xpos[geom])
                    gap = distance.distance(local_center) - radius - 1e-6
                if gap < result.distance_m:
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            int(self.model.geom_bodyid[geom]))
                    result = Clearance(gap, f'{name} <-> unmeasured {side} hand envelope')
        return result
