# -*- coding: utf-8 -*-
import unittest
import numpy as np
import torch
from pytorch_segmentation_models_trainer.utils.math_utils import (
    compute_crossfield_c0c2,
    compute_crossfield_uv,
    bilinear_interpolate,
    AverageMeter,
    RunningDecayingAverage,
)


class TestMathUtils(unittest.TestCase):
    def test_compute_crossfield_conversions(self):
        u = np.array([1.0 + 0j, 0.0 + 1j])
        v = np.array([0.0 + 1j, 1.0 + 0j])

        c0c2 = compute_crossfield_c0c2(u, v)
        self.assertEqual(c0c2.shape, (2, 4))

        u_back, v_back = compute_crossfield_uv(c0c2)
        # Check if {u_back^2, v_back^2} == {u^2, v^2}
        for i in range(len(u)):
            set_back = {
                np.round(np.power(u_back[i], 2), 6),
                np.round(np.power(v_back[i], 2), 6),
            }
            set_orig = {np.round(np.power(u[i], 2), 6), np.round(np.power(v[i], 2), 6)}
            self.assertEqual(set_back, set_orig)

    def test_bilinear_interpolate(self):
        im = torch.zeros((1, 1, 10, 10))
        im[0, 0, 5, 5] = 1.0
        im[0, 0, 5, 6] = 2.0
        im[0, 0, 6, 5] = 3.0
        im[0, 0, 6, 6] = 4.0

        # Center of these 4 pixels
        pos = torch.tensor([[5.5, 5.5]])
        val = bilinear_interpolate(im, pos)
        # (1+2+3+4)/4 = 2.5
        self.assertAlmostEqual(val.item(), 2.5)

    def test_average_meter(self):
        am = AverageMeter("test")
        am.update(10, n=2)
        am.update(20, n=1)
        self.assertEqual(am.avg, (10 * 2 + 20 * 1) / 3)
        self.assertEqual(am.val, 20)
        am.reset()
        self.assertEqual(am.avg, 0)

    def test_running_decaying_average(self):
        rda = RunningDecayingAverage(decay=0.9, init_val=10)
        rda.update(20)
        # avg = 20 * (1 - 0.9) + 10 * 0.9 = 20 * 0.1 + 9 = 11
        self.assertAlmostEqual(rda.avg, 11.0)


if __name__ == "__main__":
    unittest.main()
