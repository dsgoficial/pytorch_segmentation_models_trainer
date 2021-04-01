# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-31
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer 
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Code inspired by the one in                                           *
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""

import torch


def get_real(input_tensor, complex_dim=-1):
    return input_tensor.select(complex_dim, 0)

def get_imag(input_tensor, complex_dim=-1):
    return input_tensor.select(complex_dim, 1)

def complex_mul(input_tensor1, input_tensor2, complex_dim=-1):
    t1_real = get_real(input_tensor1, complex_dim)
    t1_imag = get_imag(input_tensor1, complex_dim)
    t2_real = get_real(input_tensor2, complex_dim)
    t2_imag = get_imag(input_tensor2, complex_dim)

    ac = t1_real * t2_real
    bd = t1_imag * t2_imag
    ad = t1_real * t2_imag
    bc = t1_imag * t2_real
    tr_real = ac - bd
    tr_imag = ad + bc

    tr = torch.stack([tr_real, tr_imag], dim=complex_dim)

    return tr

def complex_sqrt(input_tensor, complex_dim=-1):
    sqrt_t_abs = torch.sqrt(complex_abs(input_tensor, complex_dim))
    sqrt_t_arg = complex_arg(input_tensor, complex_dim) / 2
    # Overwrite input_tensor with cos(\theta / 2) + i sin(\theta / 2):
    sqrt_t = sqrt_t_abs.unsqueeze(complex_dim) * torch.stack([torch.cos(sqrt_t_arg), torch.sin(sqrt_t_arg)], dim=complex_dim)
    return sqrt_t

def complex_abs_squared(input_tensor, complex_dim=-1):
    return get_real(input_tensor, complex_dim)**2 + get_imag(input_tensor, complex_dim)**2

def complex_abs(input_tensor, complex_dim=-1):
    return torch.sqrt(complex_abs_squared(input_tensor, complex_dim=complex_dim))

def complex_arg(input_tensor, complex_dim=-1):
    return torch.atan2(get_imag(input_tensor, complex_dim), get_real(input_tensor, complex_dim))

def main():
    device = None

    input_tensor1 = torch.Tensor([
        [2, 0],
        [0, 2],
        [-1, 0],
        [0, -1],
    ]).to(device)
    input_tensor2 = torch.Tensor([
        [2, 0],
        [0, 2],
        [-1, 0],
        [0, -1],
    ]).to(device)
    complex_dim = -1

    print(input_tensor1.int())
    print(input_tensor2.int())
    t1_mul_t2 = complex_mul(input_tensor1, input_tensor2, complex_dim)
    print(t1_mul_t2.int())

    sqrt_t1_mul_t2 = complex_sqrt(t1_mul_t2)
    print(sqrt_t1_mul_t2.int())

if __name__ == "__main__":
    main()