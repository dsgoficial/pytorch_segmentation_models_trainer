# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-07
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

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import skan
import torch


class Skeleton:
    def __init__(self, coordinates=None, paths=None, degrees=None):
        self.coordinates = np.empty((0, 2), dtype=np.float) \
            if coordinates is None else coordinates
        self.paths = Paths() if paths is None else paths
        self.degrees = np.empty(0, dtype=np.int64) if degrees is None \
        else degrees

class Paths:
    def __init__(self, indices=None, indptr=None):
        self.indices = np.empty(0, dtype=np.int64) if indices is None else indices
        self.indptr = np.empty(0, dtype=np.int64) if indptr is None else indptr

class TensorSkeleton(object):
    def __init__(self, pos, degrees, path_index, path_delim, batch, batch_delim, batch_size):
        """
        In the text below, we use the following notation:
        - B: batch size
        - N: the number of points in all skeletons,
        - P: the number of paths in the skeletons
        - J: the number of junction nodes
        - Sd: the sum of the degrees of all the junction nodes

        :param pos (N, 2): union of skeleton points in ij format
        :param degrees (N,): Degrees of each node in the graph
        :param path_index (N - J + Sd,): Indices in pos of all paths (equivalent to 'indices' in the paths crs matrix)
        :param path_delim (P + 1,): Indices in path_index delimiting each path (equivalent to 'indptr' in the paths crs matrix)
        :param batch (N,): batch index of each point
        :param batch_delim (B + 1,): Indices in path_delim delimiting each batch
        """
        assert pos.shape[0] == batch.shape[0]
        self.pos = pos
        self.degrees = degrees
        self.path_index = path_index
        self.path_delim = path_delim
        self.batch = batch
        self.batch_delim = batch_delim
        self.batch_size = batch_size

    @property
    def num_nodes(self):
        return self.pos.shape[0]

    @property
    def num_paths(self):
        return max(0, self.path_delim.shape[0] - 1)

    def to(self, device):
        self.pos = self.pos.to(device)
        self.degrees = self.degrees.to(device)
        self.path_index = self.path_index.to(device)
        self.path_delim = self.path_delim.to(device)
        self.batch = self.batch.to(device)
        self.batch_delim = self.batch_delim.to(device)


def skeletons_to_tensorskeleton(skeletons_batch: List[Skeleton], device: str=None) -> TensorSkeleton:
    """
    In the text below, we use the following notation:
    - B: batch size
    - N: the number of points in all skeletons,
    - P: the number of paths in the skeletons
    - J: the number of junction nodes
    - Sd: the sum of the degrees of all the junction nodes

    Parametrizes B skeletons into PyTorch tensors:
    - pos (N, 2): union of skeleton points in ij format
    - path_index (N - J + Sd,): Indices in pos of all paths (equivalent to 'indices' in the paths crs matrix)
    - path_delim (P + 1,): Indices in path_index delimiting each path (equivalent to 'indptr' in the paths crs matrix)
    - batch (N,): batch index of each point

    :param skeletons_batch: Batch of coordinates of skeletons [Skeleton(coordinates, paths(indices, indptr), degrees), ...]
    :return: TensorSkeleton(pos, path_index, path_delim, batch, batch_size)
    """
    batch_size = len(skeletons_batch)
    batch_delim_list = []
    batch_list = []
    path_delim_list = []
    path_index_list = []
    batch_delim_offset = 0
    path_delim_offset = 0
    path_index_offset = 0
    if batch_size > 0:
        batch_delim_list.append(0)
    for batch_i, skeleton in enumerate(skeletons_batch):
        path_index_list.append(
            skeleton.paths.indices + path_index_offset
        )
        path_delim_list.append(
            path_delim_offset + skeleton.paths.indptr[:-1] if batch_i < batch_size - 1 \
                else path_delim_offset + skeleton.paths.indptr
        )
        n_points = skeleton.coordinates.shape[0]
        batch_list.append(batch_i * np.ones(n_points, dtype=np.int64))
        path_index_offset += skeleton.coordinates.shape[0]
        path_delim_offset += skeleton.paths.indices.shape[0]
        batch_delim_offset += max(0, skeleton.paths.indptr.shape[0] - 1)
        batch_delim_list.append(batch_delim_offset)

    return TensorSkeleton(
        pos=torch.from_numpy(
            np.concatenate([i.coordinates for i in skeletons_batch], axis=0)
        ).float().to(device),
        degrees=torch.from_numpy(
            np.concatenate([i.degrees for i in skeletons_batch], axis=0),
        ).long().to(device),
        path_index=torch.from_numpy(
            np.concatenate(path_index_list, axis=0)
        ).long().to(device),
        path_delim=torch.from_numpy(
            np.concatenate(path_delim_list, axis=0)
        ).long().to(device),
        batch=torch.from_numpy(
            np.concatenate(batch_list, axis=0)
        ).long().to(device),
        batch_delim=torch.tensor(
            batch_delim_list,
            dtype=torch.long,
            device=device
        ),
        batch_size=batch_size
    )

def tensorskeleton_to_skeletons(tensorskeleton: TensorSkeleton) -> List[Skeleton]:
    skeletons_list = []
    path_index_offset = 0
    path_delim_offset = 0
    for batch_i in range(tensorskeleton.batch_size):
        batch_slice = tensorskeleton.batch_delim[batch_i:batch_i+2]
        indptr = tensorskeleton.path_delim[batch_slice[0]:batch_slice[1] + 1].cpu().numpy()
        indices = tensorskeleton.path_index[indptr[0]:indptr[-1]].cpu().numpy()
        if indptr.shape[0] >= 2:
            coordinates = tensorskeleton.pos[tensorskeleton.batch == batch_i].detach().cpu().numpy()
            indices = indices - path_index_offset
            indptr = indptr - path_delim_offset
            skeleton = Skeleton(coordinates, Paths(indices, indptr))
            skeletons_list.append(skeleton)
            n_points = coordinates.shape[0]
            paths_length = indices.shape[0]
            path_index_offset += n_points
            path_delim_offset += paths_length
        else:
            skeleton = Skeleton()
            skeletons_list.append(skeleton)
    return skeletons_list

def plot_skeleton(skeleton: Skeleton):
    fig1, ax1 = plt.subplots()
    for path_i in range(skeleton.paths.indptr.shape[0] - 1):
        start, stop = skeleton.paths.indptr[path_i:path_i + 2]
        path_indices = skeleton.paths.indices[start:stop]
        path_coordinates = skeleton.coordinates[path_indices]
        ax1.plot(path_coordinates[:, 1], path_coordinates[:, 0])

def main():

    device = "cpu"

    np.random.seed(0)
    torch.manual_seed(0)

    spatial_shape = (10, 10)

    skan_skeletons_batch = []
    skeleton_image = np.zeros(spatial_shape, dtype=np.bool)
    skeleton_image[2, :] = True
    skeleton_image[:, 2] = True
    skeleton_image[7, :] = True
    skeleton_image[:, 7] = True
    skan_skeleton = skan.Skeleton(skeleton_image, keep_images=False)
    skan_skeletons_batch.append(skan_skeleton)
    # plt.imshow(skeleton_image)
    # plot_skeleton(skan_skeleton)
    # plt.show()

    skeleton_image = np.zeros(spatial_shape, dtype=np.bool)
    skeleton_image[5, :] = True
    skeleton_image[:, 5] = True
    skan_skeleton = skan.Skeleton(skeleton_image, keep_images=False)
    skan_skeletons_batch.append(skan_skeleton)
    # plt.imshow(skeleton_image)
    # plot_skeleton(skan_skeleton)
    # plt.show()

    skeletons_batch = [Skeleton(skan_skeleton.coordinates, Paths(skan_skeleton.paths.indices, skan_skeleton.paths.indptr)) for skan_skeleton in skan_skeletons_batch]

    print("# --- skeletons_to_tensorskeleton() --- #")
    tensorskeleton = skeletons_to_tensorskeleton(skeletons_batch, device=device)
    print("# --- --- #")
    # print("batch:")
    # print(tensorskeleton.batch)
    # print("pos:")
    # print(tensorskeleton.pos.shape)
    # print(tensorskeleton.pos)
    print("path_index:")
    print(tensorskeleton.path_index.shape)
    print(tensorskeleton.path_index)
    print("path_delim:")
    print(tensorskeleton.path_delim.shape)
    print(tensorskeleton.path_delim)
    print("batch_delim:")
    print(tensorskeleton.batch_delim.shape)
    print(tensorskeleton.batch_delim)

    print("# --- tensorskeleton_to_skeletons() --- #")
    skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)

    # Plot
    import os
    for i, skeleton in enumerate(skeletons_batch):
        plot_skeleton(skeleton)
        plt.savefig(
            os.path.join('/Users/philipeborba/github_repos/pytorch_segmentation_models_trainer/tests/testing_data/expected_outputs/tensor_tools', f'output{i}.png')
        )
        plt.close()
        # plt.show()

    print("# --- --- #")


if __name__ == "__main__":
    main()
