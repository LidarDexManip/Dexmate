import unittest
import numpy as np
from wuji_teleop.ee_dashboard import jog_target, drag_target


class TargetInputTests(unittest.TestCase):
    def test_translation_and_orientation_are_separate(self):
        q = np.array([.3, -.2, 1., 10., 20., 30.])
        np.testing.assert_allclose(jog_target(q, 'Up'), q + [.005, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(jog_target(q, 'i'), q + [0, 0, 0, 2, 0, 0])
        np.testing.assert_allclose(jog_target(q, 'Next', True), q + [0, 0, -.001, 0, 0, 0])
        np.testing.assert_array_equal(q, [.3, -.2, 1., 10., 20., 30.])

    def test_top_drag_preserves_z_and_orientation(self):
        q = [.3, -.2, 1., 10., 20., 30.]
        np.testing.assert_allclose(drag_target(q, q[:3], 'top', 150, 65),
                                   [.35, -.25, 1., 10., 20., 30.])

    def test_side_drag_preserves_y_and_orientation(self):
        q = [.3, -.2, 1., 10., 20., 30.]
        np.testing.assert_allclose(drag_target(q, q[:3], 'side', 150, 65),
                                   [.35, -.2, 1.05, 10., 20., 30.])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            jog_target([float('nan')] * 6, 'Up')
        with self.assertRaises(ValueError):
            drag_target(np.zeros(6), np.zeros(3), 'invalid', 0, 0)
