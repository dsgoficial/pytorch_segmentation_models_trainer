# -*- coding: utf-8 -*-
import torch
import unittest
from pytorch_segmentation_models_trainer.utils.complex_utils import (
    get_real,
    get_imag,
    complex_mul,
    complex_sqrt,
    complex_abs_squared,
    complex_abs,
    complex_arg,
)


class TestComplexUtils(unittest.TestCase):
    def setUp(self):
        self.t1 = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
        self.t2 = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
        self.complex_dim = -1

    def test_get_real(self):
        real = get_real(self.t1, self.complex_dim)
        expected = torch.tensor([2.0, 0.0, -1.0, 0.0])
        self.assertTrue(torch.allclose(real, expected))

    def test_get_imag(self):
        imag = get_imag(self.t1, self.complex_dim)
        expected = torch.tensor([0.0, 2.0, 0.0, -1.0])
        self.assertTrue(torch.allclose(imag, expected))

    def test_complex_mul(self):
        # (2+0i)*(2+0i) = 4+0i
        # (0+2i)*(0+2i) = -4+0i
        # (-1+0i)*(-1+0i) = 1+0i
        # (0-1i)*(0-1i) = -1+0i
        res = complex_mul(self.t1, self.t2, self.complex_dim)
        expected = torch.tensor([[4.0, 0.0], [-4.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        self.assertTrue(torch.allclose(res, expected))

    def test_complex_abs_squared(self):
        res = complex_abs_squared(self.t1, self.complex_dim)
        expected = torch.tensor([4.0, 4.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(res, expected))

    def test_complex_abs(self):
        res = complex_abs(self.t1, self.complex_dim)
        expected = torch.tensor([2.0, 2.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(res, expected))

    def test_complex_arg(self):
        res = complex_arg(self.t1, self.complex_dim)
        # atan2(0, 2) = 0
        # atan2(2, 0) = pi/2
        # atan2(0, -1) = pi
        # atan2(-1, 0) = -pi/2
        expected = torch.tensor([0.0, torch.pi / 2, torch.pi, -torch.pi / 2])
        self.assertTrue(torch.allclose(res, expected))

    def test_complex_sqrt(self):
        # sqrt(4+0i) = 2+0i
        # sqrt(-4+0i) = 0+2i
        t = torch.tensor([[4.0, 0.0], [-4.0, 0.0]])
        res = complex_sqrt(t, self.complex_dim)
        expected = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        # torch.allclose might have small precision issues with trig functions
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
