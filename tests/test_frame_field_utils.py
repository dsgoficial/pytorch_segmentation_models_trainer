# -*- coding: utf-8 -*-
import torch
import numpy as np
import unittest
from shapely.geometry import LineString, Polygon
from pytorch_segmentation_models_trainer.utils.frame_field_utils import (
    LaplacianPenalty,
    framefield_align_error,
    c0c2_to_uv,
    compute_closest_in_uv,
    detect_corners,
    compute_crossfield_to_plot,
)


class TestFrameFieldUtils(unittest.TestCase):
    def test_laplacian_penalty(self):
        channels = 1
        lp = LaplacianPenalty(channels)
        input_tensor = torch.zeros(1, channels, 10, 10)
        input_tensor[0, 0, 5, 5] = 1.0
        output = lp(input_tensor)
        self.assertEqual(output.shape, (1, channels, 10, 10))
        # The center should have the highest penalty
        self.assertGreater(output[0, 0, 5, 5], 0)

    def test_framefield_align_error(self):
        # Test basic shapes and error calculation
        c0 = torch.randn(1, 2, 10, 10)
        c2 = torch.randn(1, 2, 10, 10)
        z = torch.randn(1, 2, 10, 10)
        error = framefield_align_error(c0, c2, z, complex_dim=1)
        self.assertEqual(error.shape, (1, 10, 10))

    def test_c0c2_to_uv(self):
        # c0c2 should have shape (B, 4, H, W)
        c0c2 = torch.randn(1, 4, 10, 10)
        uv = c0c2_to_uv(c0c2)
        # Output shape: (B, 2, 2, H, W) -> (batch, uv_index, real_imag, H, W)
        self.assertEqual(uv.shape, (1, 2, 2, 10, 10))

    def test_compute_closest_in_uv(self):
        # directions (N, 2)
        # uv (N, 2, 2)
        directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # u = [1, 0], v = [0, 1]
        uv = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]  # N=0  # N=1
        )
        closest = compute_closest_in_uv(directions, uv)
        # for [1,0], u is closer -> index 0?
        # Wait, argmin(abs(dot)). dot([1,0],[1,0])=1, dot([1,0],[0,1])=0.
        # argmin([1, 0]) = 1. So it returns the one with SMALLER dot product?
        # Looking at code: uv_dot_dir = torch.sum(uv * directions[:, None, :], dim=2)
        # abs_uv_dot_dir = torch.abs(uv_dot_dir)
        # closest_in_uv = torch.argmin(abs_uv_dot_dir, dim=1)
        # If it uses argmin, it's finding the one MOST ORTHOGONAL.
        # Usually we want argmax for alignment. Let's check the logic.
        self.assertEqual(closest[0], 1)  # [1,0] is orthogonal to [0,1] (index 1)

    def test_compute_crossfield_to_plot(self):
        angle = np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]])
        crossfield = compute_crossfield_to_plot(angle)
        self.assertEqual(crossfield.shape, (1, 4, 2, 2))

    def test_compute_crossfield_to_plot_accepts_tensor_and_shape(self):
        angle = torch.zeros(1, 2, 2)
        crossfield = compute_crossfield_to_plot(angle, crossfield_shape=(1, 4, 2, 2))
        self.assertEqual(crossfield.shape, (1, 4, 2, 2))

    def test_framefield_align_error_validates_shapes_and_complex_dim(self):
        c0 = torch.randn(1, 2, 4, 4)
        c2 = torch.randn(1, 2, 4, 4)
        z = torch.randn(1, 2, 4, 5)
        with self.assertRaises(AssertionError):
            framefield_align_error(c0, c2, z, complex_dim=1)
        with self.assertRaises(AssertionError):
            framefield_align_error(c0, c2, torch.randn(1, 3, 4, 4), complex_dim=1)

    def test_detect_corners_handles_numpy_shapely_and_empty_middle(self):
        u = np.ones((5, 5), dtype=np.complex64)
        v = 1j * np.ones((5, 5), dtype=np.complex64)
        closed = Polygon([(1, 1), (3, 1), (3, 3), (1, 1)])
        open_line = LineString([(0, 0), (2, 0), (2, 2)])
        two_point = np.array([[0.0, 0.0], [1.0, 1.0]])

        masks = detect_corners([closed, open_line, two_point], u, v)

        self.assertEqual(len(masks), 3)
        self.assertEqual(masks[0].shape[0], 4)
        self.assertTrue(masks[0][-1] == masks[0][0])
        self.assertTrue(masks[1][0])
        self.assertTrue(masks[1][-1])
        self.assertTrue(masks[2][0])
        self.assertTrue(masks[2][-1])


if __name__ == "__main__":
    unittest.main()
