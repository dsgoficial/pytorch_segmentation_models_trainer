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

import torch
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import \
    Paths, Skeleton, TensorSkeleton, skeletons_to_tensorskeleton, tensorskeleton_to_skeletons

class AlignLoss:
    def __init__(self, tensorskeleton: TensorSkeleton, indicator: torch.Tensor, level: float, c0c2: torch.Tensor, loss_params):
        """
        :param tensorskeleton: skeleton graph in tensor format
        :return:
        """
        self.tensorskeleton = tensorskeleton
        self.indicator = indicator
        self.level = level
        self.c0c2 = c0c2
        # self.uv = frame_field_utils.c0c2_to_uv(c0c2)

        # Prepare junction_corner_index:

        # TODO: junction_corner_index: list
        self.junction_corner_index = get_junction_corner_index(tensorskeleton)

        # Loss coefs
        self.data_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                           loss_params["coefs"]["data"])
        self.length_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["length"])
        self.crossfield_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                                 loss_params["coefs"]["crossfield"])
        self.curvature_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                                loss_params["coefs"]["curvature"])
        self.corner_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["corner"])
        self.junction_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["junction"])

        self.curvature_dissimilarity_threshold = loss_params["curvature_dissimilarity_threshold"]
        self.corner_angles = np.pi * torch.tensor(loss_params["corner_angles"]) / 180  # Convert to radians
        self.corner_angle_threshold = np.pi * loss_params["corner_angle_threshold"] / 180  # Convert to radians
        self.junction_angles = np.pi * torch.tensor(loss_params["junction_angles"]) / 180  # Convert to radians
        self.junction_angle_weights = torch.tensor(loss_params["junction_angle_weights"])
        self.junction_angle_threshold = np.pi * loss_params["junction_angle_threshold"] / 180  # Convert to radians

        # Pre-compute useful pointers
        # edge_index_start = tensorskeleton.path_index[:-1]
        # edge_index_end = tensorskeleton.path_index[1:]
        #
        # self.tensorskeleton.edge_index = edge_index

    def __call__(self, pos: torch.Tensor, iter_num: int):
        # --- Align to frame field loss
        path_pos = pos[self.tensorskeleton.path_index]
        detached_path_pos = path_pos.detach()
        path_batch = self.tensorskeleton.batch[self.tensorskeleton.path_index]
        tangents = path_pos[1:] - path_pos[:-1]
        # Compute edge mask to remove edges that connect two different paths from loss
        edge_mask = torch.ones((tangents.shape[0]), device=tangents.device)
        edge_mask[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out edges between paths

        midpoints = (path_pos[1:] + path_pos[:-1]) / 2
        midpoints_batch = self.tensorskeleton.batch[self.tensorskeleton.path_index[:-1]]  # Same as start point of edge

        midpoints_int = midpoints.round().long()
        midpoints_int[:, 0] = torch.clamp(midpoints_int[:, 0], 0, self.c0c2.shape[2] - 1)
        midpoints_int[:, 1] = torch.clamp(midpoints_int[:, 1], 0, self.c0c2.shape[3] - 1)
        midpoints_c0 = self.c0c2[midpoints_batch, :2, midpoints_int[:, 0], midpoints_int[:, 1]]
        midpoints_c2 = self.c0c2[midpoints_batch, 2:, midpoints_int[:, 0], midpoints_int[:, 1]]

        norms = torch.norm(tangents, dim=-1)
        edge_mask[norms < 0.1] = 0  # Zero out very small edges
        normed_tangents = tangents / (norms[:, None] + 1e-6)

        align_loss = frame_field_utils.framefield_align_error(midpoints_c0, midpoints_c2, normed_tangents, complex_dim=1)
        align_loss = align_loss * edge_mask
        total_align_loss = torch.sum(align_loss)

        # --- Align to level set of indicator:
        pos_value = bilinear_interpolate(self.indicator[:, None, ...], pos, batch=self.tensorskeleton.batch)
        # TODO: use grid_sample with batch: put batch dim to height dim and make a single big image.
        # TODO: Convert pos accordingly and take care of borders
        # height = self.indicator.shape[1]
        # width = self.indicator.shape[2]
        # normed_xy = tensorskeleton.pos.roll(shifts=1, dims=-1)
        # normed_xy[: 0] /= (width-1)
        # normed_xy[: 1] /= (height-1)
        # centered_xy = 2*normed_xy - 1
        # pos_value = torch.nn.functional.grid_sample(self.indicator[None, None, ...],
        #                                             centered_batch_xy[None, None, ...], align_corners=True).squeeze()
        level_loss = torch.sum(torch.pow(pos_value - self.level, 2))

        # --- Prepare useful tensors for curvature loss:
        prev_pos = detached_path_pos[:-2]
        middle_pos = path_pos[1:-1]
        next_pos = detached_path_pos[2:]
        prev_tangent = middle_pos - prev_pos
        next_tangent = next_pos - middle_pos
        prev_norm = torch.norm(prev_tangent, dim=-1)
        next_norm = torch.norm(next_tangent, dim=-1)

        # --- Apply length penalty with sum of squared norm to penalize uneven edge lengths on selected edges
        prev_length_loss = torch.pow(prev_norm, 2)
        next_length_loss = torch.pow(next_norm, 2)
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out invalid norms between paths
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid norms between paths
        length_loss = prev_length_loss + next_length_loss
        total_length_loss = torch.sum(length_loss)

        # --- Detect corners:
        with torch.no_grad():
            middle_pos_int = middle_pos.round().long()
            middle_pos_int[:, 0] = torch.clamp(middle_pos_int[:, 0], 0, self.c0c2.shape[2] - 1)
            middle_pos_int[:, 1] = torch.clamp(middle_pos_int[:, 1], 0, self.c0c2.shape[3] - 1)
            middle_batch = path_batch[1:-1]
            middle_c0c2 = self.c0c2[middle_batch, :, middle_pos_int[:, 0], middle_pos_int[:, 1]]
            middle_uv = frame_field_utils.c0c2_to_uv(middle_c0c2)
            prev_tangent_closest_in_uv = frame_field_utils.compute_closest_in_uv(prev_tangent, middle_uv)
            next_tangent_closest_in_uv = frame_field_utils.compute_closest_in_uv(next_tangent, middle_uv)
            is_corner = prev_tangent_closest_in_uv != next_tangent_closest_in_uv
            is_corner[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid corners between sub-paths
            is_corner[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out invalid corners between sub-paths
            is_corner_index = torch.nonzero(is_corner)[:, 0] + 1  # Shift due to first vertex not being represented in is_corner
            # TODO: evaluate running time of torch.sort: does it slow down the optimization much?
            sub_path_delim, sub_path_sort_indices = torch.sort(torch.cat([self.tensorskeleton.path_delim, is_corner_index]))
            sub_path_delim_is_corner = self.tensorskeleton.path_delim.shape[0] <= sub_path_sort_indices  # If condition is true, then the delimiter is from is_corner_index

        # --- Compute sub-path dissimilarity in the sense of the Ramer-Douglas-Peucker alg
        # dissimilarity is equal to the max distance of vertices to the straight line connecting the start and end points of the sub-path.
        with torch.no_grad():
            sub_path_start_index = sub_path_delim[:-1]
            sub_path_end_index = sub_path_delim[1:].clone()
            sub_path_end_index[~sub_path_delim_is_corner[1:]] -= 1  # For non-corner delimitators, have to shift
            sub_path_start_pos = path_pos[sub_path_start_index]
            sub_path_end_pos = path_pos[sub_path_end_index]
            sub_path_normal = sub_path_end_pos - sub_path_start_pos
            sub_path_normal = sub_path_normal / (torch.norm(sub_path_normal, dim=1)[:, None] + 1e-6)
            expanded_sub_path_start_pos = torch_scatter.gather_csr(sub_path_start_pos,
                                                                   sub_path_delim)
            expanded_sub_path_normal = torch_scatter.gather_csr(sub_path_normal,
                                                                 sub_path_delim)
            relative_path_pos = path_pos - expanded_sub_path_start_pos
            relative_path_pos_projected_lengh = torch.sum(relative_path_pos * expanded_sub_path_normal, dim=1)
            relative_path_pos_projected = relative_path_pos_projected_lengh[:, None] * expanded_sub_path_normal
            path_pos_distance = torch.norm(relative_path_pos - relative_path_pos_projected, dim=1)
            sub_path_max_distance = torch_scatter.segment_max_csr(path_pos_distance, sub_path_delim)[0]
            sub_path_small_dissimilarity_mask = sub_path_max_distance < self.curvature_dissimilarity_threshold

        # --- Compute curvature loss:
        # print("prev_norm:", prev_norm.min().item(), prev_norm.max().item())
        prev_dir = prev_tangent / (prev_norm[:, None] + 1e-6)
        next_dir = next_tangent / (next_norm[:, None] + 1e-6)
        dot = prev_dir[:, 0] * next_dir[:, 0] + \
              prev_dir[:, 1] * next_dir[:, 1]  # dot product
        det = prev_dir[:, 0] * next_dir[:, 1] - \
              prev_dir[:, 1] * next_dir[:, 0]  # determinant
        vertex_angles = torch.acos(dot) * torch.sign(det)  # TODO: remove acos for speed? Switch everything to signed dot product?
        # Save angles of detected corners:
        corner_angles = vertex_angles[is_corner_index - 1]  # -1 because of the shift of vertex_angles relative to path_pos
        # Compute the mean vertex angle for each sub-path separately:
        vertex_angles[sub_path_delim[1:-1] - 1] = 0  # Zero out invalid angles between paths as well as corner angles
        vertex_angles[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid angles between paths (caused by the junction points being in all paths of the junction)
        sub_path_vertex_angle_delim = sub_path_delim.clone()
        sub_path_vertex_angle_delim[-1] -= 2
        sub_path_sum_vertex_angle = torch_scatter.segment_sum_csr(vertex_angles, sub_path_vertex_angle_delim)
        sub_path_lengths = sub_path_delim[1:] - sub_path_delim[:-1]
        sub_path_lengths[sub_path_delim_is_corner[1:]] += 1  # Fix length of paths split by corners
        sub_path_valid_angle_count = sub_path_lengths - 2
        # print("sub_path_valid_angle_count:", sub_path_valid_angle_count.min().item(), sub_path_valid_angle_count.max().item())
        sub_path_mean_vertex_angles = sub_path_sum_vertex_angle / sub_path_valid_angle_count
        sub_path_mean_vertex_angles[sub_path_small_dissimilarity_mask] = 0  # Optimize sub-path with a small dissimilarity to have straight edges
        expanded_sub_path_mean_vertex_angles = torch_scatter.gather_csr(sub_path_mean_vertex_angles,
                                                                        sub_path_vertex_angle_delim)
        curvature_loss = torch.pow(vertex_angles - expanded_sub_path_mean_vertex_angles, 2)
        curvature_loss[sub_path_delim[1:-1] - 1] = 0  # Zero out loss for start vertex of inner sub-paths
        curvature_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out loss for end vertex of inner paths (caused by the junction points being in all paths of the junction)
        total_curvature_loss = torch.sum(curvature_loss)

        # --- Computer corner loss:
        corner_abs_angles = torch.abs(corner_angles)
        self.corner_angles = self.corner_angles.to(corner_abs_angles.device)
        corner_snap_dist = torch.abs(corner_abs_angles[:, None] - self.corner_angles)
        corner_snap_dist_optim_mask = corner_snap_dist < self.corner_angle_threshold
        corner_snap_dist_optim = corner_snap_dist[corner_snap_dist_optim_mask]
        corner_loss = torch.pow(corner_snap_dist_optim, 2)
        total_corner_loss = torch.sum(corner_loss)

        # --- Compute junction corner loss
        junction_corner = pos[self.junction_corner_index, :]
        junction_prev_tangent = junction_corner[:, 1, :] - junction_corner[:, 0, :]
        junction_next_tangent = junction_corner[:, 2, :] - junction_corner[:, 1, :]
        junction_prev_dir = junction_prev_tangent / (torch.norm(junction_prev_tangent, dim=-1)[:, None] + 1e-6)
        junction_next_dir = junction_next_tangent / (torch.norm(junction_next_tangent, dim=-1)[:, None] + 1e-6)
        junction_dot = junction_prev_dir[:, 0] * junction_next_dir[:, 0] + \
              junction_prev_dir[:, 1] * junction_next_dir[:, 1]  # dot product
        junction_abs_angles = torch.acos(junction_dot)
        self.junction_angles = self.junction_angles.to(junction_abs_angles.device)
        self.junction_angle_weights = self.junction_angle_weights.to(junction_abs_angles.device)
        junction_snap_dist = torch.abs(junction_abs_angles[:, None] - self.junction_angles)
        junction_snap_dist_optim_mask = junction_snap_dist < self.junction_angle_threshold
        junction_snap_dist *= self.junction_angle_weights[None, :]  # Apply weights per target angle (as we use the L1 norm, it works applying before the norm)
        junction_snap_dist_optim = junction_snap_dist[junction_snap_dist_optim_mask]
        junction_loss = torch.abs(junction_snap_dist_optim)
        total_junction_loss = torch.sum(junction_loss)

        losses_dict = {
            "align": total_align_loss.item(),
            "level": level_loss.item(),
            "length": total_length_loss.item(),
            "curvature": total_curvature_loss.item(),
            "corner": total_corner_loss.item(),
            "junction": total_junction_loss.item(),
        }
        # Get the loss coefs depending on the current step:
        data_coef = float(self.data_coef_interp(iter_num))
        length_coef = float(self.length_coef_interp(iter_num))
        crossfield_coef = float(self.crossfield_coef_interp(iter_num))
        curvature_coef = float(self.curvature_coef_interp(iter_num))
        corner_coef = float(self.corner_coef_interp(iter_num))
        junction_coef = float(self.junction_coef_interp(iter_num))
        # total_loss = data_coef * level_loss + length_coef * total_length_loss + crossfield_coef * total_align_loss + \
        #              curvature_coef * total_curvature_loss + corner_coef * total_corner_loss + junction_coef * total_junction_loss
        total_loss = data_coef * level_loss + length_coef * total_length_loss + crossfield_coef * total_align_loss + \
                     curvature_coef * total_curvature_loss + corner_coef * total_corner_loss + junction_coef * total_junction_loss

        # print(iter_num)
        # input("<Enter>...")

        return total_loss, losses_dict

