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
import numpy as np
import scipy.interpolate
import torch
import torch_scatter
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import (
    TensorSkeleton)
from pytorch_segmentation_models_trainer.utils import (frame_field_utils,
                                                       math_utils)


class AlignLoss:
    def __init__(
        self,
        tensorskeleton: TensorSkeleton,
        indicator: torch.Tensor,
        level: float,
        c0c2: torch.Tensor,
        loss_params
    ):
        """
        :param tensorskeleton: skeleton graph in tensor format
        :return:
        """
        self.tensorskeleton = tensorskeleton
        self.indicator = indicator
        self.level = level
        self.c0c2 = c0c2
        self.junction_corner_index = get_junction_corner_index(tensorskeleton)
        self.data_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.data)
        self.length_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.length)
        self.crossfield_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.crossfield)
        self.curvature_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.curvature)
        self.corner_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.corner)
        self.junction_coef_interp = scipy.interpolate.interp1d(
            loss_params.coefs.step_thresholds, loss_params.coefs.junction)
        self.curvature_dissimilarity_threshold = loss_params.curvature_dissimilarity_threshold
        self.corner_angles = np.pi * torch.tensor(loss_params.corner_angles) / 180  # Convert to radians
        self.corner_angle_threshold = np.pi * loss_params.corner_angle_threshold / 180  # Convert to radians
        self.junction_angles = np.pi * torch.tensor(loss_params.junction_angles) / 180  # Convert to radians
        self.junction_angle_weights = torch.tensor(loss_params.junction_angle_weights)
        self.junction_angle_threshold = np.pi * loss_params.junction_angle_threshold / 180  # Convert to radians

    def __call__(self, pos: torch.Tensor, iter_num: int):
        # --- Align to frame field loss
        path_pos = pos[self.tensorskeleton.path_index]
        detached_path_pos = path_pos.detach()
        path_batch = self.tensorskeleton.batch[self.tensorskeleton.path_index]
        tangents = path_pos[1:] - path_pos[:-1]
        # Compute edge mask to remove edges that connect two different paths from loss
        total_align_loss = self._compute_edge_mask(tangents, path_pos)

        # --- Align to level set of indicator:
        pos_value = math_utils.bilinear_interpolate(
            self.indicator[:, None, ...], pos, batch=self.tensorskeleton.batch
        )
        level_loss = torch.sum(torch.pow(pos_value - self.level, 2))

        # --- Prepare useful tensors for curvature loss:
        (
            prev_norm, next_norm, prev_tangent, next_tangent, middle_pos
        ) = self._prepare_tensors_for_curvature_loss(detached_path_pos, path_pos)
        # --- Apply length penalty with sum of squared norm to penalize uneven edge lengths on selected edges
        total_length_loss = self._compute_total_length_loss(prev_norm, next_norm)

        # --- Detect corners:
        (
            sub_path_delim_is_corner, sub_path_delim, is_corner_index
        ) = self._detect_corners(middle_pos, path_batch, prev_tangent, next_tangent)

        sub_path_small_dissimilarity_mask = self._compute_sub_path_dissimilarity(
            sub_path_delim, sub_path_delim_is_corner, path_pos
        )
        # --- Compute curvature loss:
        vertex_angles, corner_angles = self._compute_vertex_angles_for_curvature_loss(
            prev_tangent, next_tangent, prev_norm, next_norm, is_corner_index
        )
        total_curvature_loss = self._compute_curvature_loss(
                                    vertex_angles,
                                    sub_path_delim,
                                    sub_path_delim_is_corner,
                                    sub_path_small_dissimilarity_mask
                                )

        # --- Computer corner loss:
        total_corner_loss = self._compute_corner_loss(corner_angles)

        # --- Compute junction corner loss
        total_junction_loss = self._compute_junction_corner_loss(pos)

        losses_dict = {
            "align": total_align_loss.item(),
            "level": level_loss.item(),
            "length": total_length_loss.item(),
            "curvature": total_curvature_loss.item(),
            "corner": total_corner_loss.item(),
            "junction": total_junction_loss.item(),
        }
        # Get the loss coefs depending on the current step:
        total_loss = self._compute_total_loss(
                        iter_num,
                        level_loss,
                        total_length_loss,
                        total_align_loss,
                        total_curvature_loss,
                        total_corner_loss,
                        total_junction_loss
                    )
        return total_loss, losses_dict
    
    def _compute_edge_mask(self, tangents, path_pos):
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
        return torch.sum(align_loss)
    
    def _compute_total_length_loss(self, prev_norm, next_norm):
        prev_length_loss = torch.pow(prev_norm, 2)
        next_length_loss = torch.pow(next_norm, 2)
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out invalid norms between paths
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid norms between paths
        length_loss = prev_length_loss + next_length_loss
        return torch.sum(length_loss)
    
    def _prepare_tensors_for_curvature_loss(self, detached_path_pos, path_pos):
        prev_pos = detached_path_pos[:-2]
        middle_pos = path_pos[1:-1]
        next_pos = detached_path_pos[2:]
        prev_tangent = middle_pos - prev_pos
        next_tangent = next_pos - middle_pos
        prev_norm = torch.norm(prev_tangent, dim=-1)
        next_norm = torch.norm(next_tangent, dim=-1)
        return prev_norm, next_norm, prev_tangent, next_tangent, middle_pos

    def _detect_corners(self, middle_pos, path_batch, prev_tangent, next_tangent):
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
            return sub_path_delim_is_corner, sub_path_delim, is_corner_index
    
    def _compute_total_loss(self, iter_num, level_loss, total_length_loss, total_align_loss, total_curvature_loss, total_corner_loss, total_junction_loss):
        data_coef = float(self.data_coef_interp(iter_num))
        length_coef = float(self.length_coef_interp(iter_num))
        crossfield_coef = float(self.crossfield_coef_interp(iter_num))
        curvature_coef = float(self.curvature_coef_interp(iter_num))
        corner_coef = float(self.corner_coef_interp(iter_num))
        junction_coef = float(self.junction_coef_interp(iter_num))
        return data_coef * level_loss + length_coef * total_length_loss + crossfield_coef * total_align_loss + \
                     curvature_coef * total_curvature_loss + corner_coef * total_corner_loss + junction_coef * total_junction_loss

    
    def _compute_sub_path_dissimilarity(self, sub_path_delim, sub_path_delim_is_corner, path_pos):
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
            return sub_path_small_dissimilarity_mask
    
    def _compute_vertex_angles_for_curvature_loss(self, prev_tangent, next_tangent, prev_norm, next_norm, is_corner_index):
        prev_dir = prev_tangent / (prev_norm[:, None] + 1e-6)
        next_dir = next_tangent / (next_norm[:, None] + 1e-6)
        dot = prev_dir[:, 0] * next_dir[:, 0] + \
              prev_dir[:, 1] * next_dir[:, 1] # dot prod
        det = prev_dir[:, 0] * next_dir[:, 1] - \
              prev_dir[:, 1] * next_dir[:, 0]  # determinant
        vertex_angles = torch.acos(dot) * torch.sign(det)
        # Save angles of detected corners:
        corner_angles = vertex_angles[is_corner_index - 1]  # -1 because of the shift of vertex_angles relative to path_pos
        return vertex_angles, corner_angles
    
    def _compute_curvature_loss(self, vertex_angles, sub_path_delim, sub_path_delim_is_corner, sub_path_small_dissimilarity_mask):
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
        return torch.sum(curvature_loss)
    
    def _compute_corner_loss(self, corner_angles):
        corner_abs_angles = torch.abs(corner_angles)
        self.corner_angles = self.corner_angles.to(corner_abs_angles.device)
        corner_snap_dist = torch.abs(corner_abs_angles[:, None] - self.corner_angles)
        corner_snap_dist_optim = corner_snap_dist[corner_snap_dist < self.corner_angle_threshold]
        corner_loss = torch.pow(corner_snap_dist_optim, 2)
        return torch.sum(corner_loss)
    
    def _compute_junction_corner_loss(self, pos):
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
        junction_snap_dist *= self.junction_angle_weights[None, :]  # Apply weights per target angle (as we use the L1 norm, it works applying before the norm)
        junction_snap_dist_optim = junction_snap_dist[junction_snap_dist < self.junction_angle_threshold]
        junction_loss = torch.abs(junction_snap_dist_optim)
        return torch.sum(junction_loss)

class PolygonAlignLoss:
    def __init__(
        self,
        indicator,
        level,
        c0c2,
        data_coef,
        length_coef,
        crossfield_coef,
        dist=None,
        dist_coef=None
    ):
        self.indicator = indicator
        self.level = level
        self.c0c2 = c0c2
        self.dist = dist

        self.data_coef = data_coef
        self.length_coef = length_coef
        self.crossfield_coef = crossfield_coef
        self.dist_coef = dist_coef

    def __call__(self, tensorpoly):
        """

        :param tensorpoly: closed polygon
        :return:
        """
        polygon = tensorpoly.pos[tensorpoly.to_padded_index]
        polygon_batch = tensorpoly.batch[tensorpoly.to_padded_index]

        # Compute edges:
        edges = polygon[1:] - polygon[:-1]
        # Compute edge mask to remove edges that connect two different polygons from loss
        # Also note the last poly_slice is not used, because the last edge of the last polygon is not connected to a non-existant next polygon:
        edge_mask = torch.ones((edges.shape[0]), device=edges.device)
        edge_mask[tensorpoly.to_unpadded_poly_slice[:-1, 1]] = 0

        midpoints = (polygon[1:] + polygon[:-1]) / 2
        midpoints_batch = polygon_batch[1:]

        midpoints_int = midpoints.round().long()
        midpoints_int[:, 0] = torch.clamp(midpoints_int[:, 0], 0, self.c0c2.shape[2] - 1)
        midpoints_int[:, 1] = torch.clamp(midpoints_int[:, 1], 0, self.c0c2.shape[3] - 1)
        midpoints_c0 = self.c0c2[midpoints_batch, :2, midpoints_int[:, 0], midpoints_int[:, 1]]
        midpoints_c2 = self.c0c2[midpoints_batch, 2:, midpoints_int[:, 0], midpoints_int[:, 1]]

        norms = torch.norm(edges, dim=-1)
        # Add edges with small norms to the edge mask so that losses are not computed on them
        edge_mask[norms < 0.1] = 0  # Less than 10% of a pixel
        z = edges / (norms[:, None] + 1e-3)

        # Align to crossfield
        align_loss = frame_field_utils.framefield_align_error(midpoints_c0, midpoints_c2, z, complex_dim=1)
        align_loss = align_loss * edge_mask
        total_align_loss = torch.sum(align_loss)

        # Align to level set of indicator:
        pos_indicator_value = math_utils.bilinear_interpolate(
            self.indicator[:, None, ...],
            tensorpoly.pos,
            batch=tensorpoly.batch
        )
        level_loss = torch.sum(torch.pow(pos_indicator_value - self.level, 2))

        # Align to minimum distance from the boundary
        dist_loss = None
        if self.dist is not None:
            pos_dist_value = math_utils.bilinear_interpolate(self.dist[:, None, ...], tensorpoly.pos, batch=tensorpoly.batch)
            dist_loss = torch.sum(torch.pow(pos_dist_value, 2))

        length_penalty = torch.sum(
            torch.pow(norms * edge_mask, 2))  # Sum of squared norm to penalise uneven edge lengths
        # length_penalty = torch.sum(norms)

        losses_dict = {
            "align": total_align_loss.item(),
            "level": level_loss.item(),
            "length": length_penalty.item(),
        }
        coef_list = [self.data_coef, self.length_coef, self.crossfield_coef]
        coef_sum = sum(coef_list)
        total_loss = torch.dot(coef_list, [level_loss, length_penalty, total_align_loss])
        if dist_loss is not None:
            losses_dict["dist"] = dist_loss.item()
            total_loss += self.dist_coef * dist_loss
            coef_sum += self.dist_coef
        total_loss /= coef_sum
        return total_loss, losses_dict


def get_junction_corner_index(tensorskeleton: TensorSkeleton):
    """
    Returns as a tensor the list of 3-tuples each representing a corner of a junction.
    The 3-tuple contains the indices of the 3 vertices making up the corner.

    In the text below, we use the following notation:
        - J: the number of junction nodes
        - Sd: the sum of the degrees of all the junction nodes
        - T: number of tip nodes
    @return: junction_corner_index of shape (Sd*J - T, 3) which is a list of 3-tuples (for each junction corner)
    """
    junction_edge_index = _build_junction_index(tensorskeleton)
    junction_edge_index = _remove_tip_junctions(tensorskeleton, junction_edge_index)
    grouped_junction_edge_index = _group_junction_by_sorting(junction_edge_index)
    junction_angle_to_axis = _compute_angle_to_vertical_axis_at_junction(
        tensorskeleton, grouped_junction_edge_index
    )
    junction_corner_index, junction_end_index = _build_junction_corner_index(
        tensorskeleton, grouped_junction_edge_index
    )
    return _slice_over_junction_index(
        junction_corner_index,
        junction_end_index,
        junction_angle_to_axis,
        grouped_junction_edge_index
    )

def _build_junction_index(ts: TensorSkeleton) -> torch.tensor:
    junction_edge_index = torch.empty(
        (2 * ts.num_paths, 2),
        dtype=torch.long,
        device=ts.path_index.device
    )
    junction_edge_index[:ts.num_paths, 0] = ts.path_index[ts.path_delim[:-1]]
    junction_edge_index[:ts.num_paths, 1] = ts.path_index[ts.path_delim[:-1] + 1]
    junction_edge_index[ts.num_paths:, 0] = ts.path_index[ts.path_delim[1:] - 1]
    junction_edge_index[ts.num_paths:, 1] = ts.path_index[ts.path_delim[1:] - 2]
    return junction_edge_index

def _remove_tip_junctions(ts: TensorSkeleton, junction_edge_index: torch.Tensor):
    degrees = ts.degrees[junction_edge_index[:, 0]]
    return junction_edge_index[1 < degrees, :]

def _group_junction_by_sorting(junction_edge_index: torch.Tensor):
    group_indices = torch.argsort(junction_edge_index[:, 0], dim=0)
    return junction_edge_index[group_indices, :]

def _slice_over_junction_index(junction_corner_index, junction_end_index, \
    junction_angle_to_axis, grouped_junction_edge_index, slice_start=0):
    for slice_end in junction_end_index:
        slice_angle_to_axis = junction_angle_to_axis[slice_start:slice_end]
        slice_junction_edge_index = grouped_junction_edge_index[slice_start:slice_end]
        sort_indices = torch.argsort(slice_angle_to_axis, dim=0)
        slice_junction_edge_index = slice_junction_edge_index[sort_indices]
        junction_corner_index[slice_start:slice_end, 0] = slice_junction_edge_index[:, 1]
        junction_corner_index[slice_start:slice_end, 1] = slice_junction_edge_index[:, 0]
        junction_corner_index[slice_start:slice_end, 2] = slice_junction_edge_index[:, 1].roll(-1, dims=0)
        slice_start = slice_end
    return junction_corner_index

def _compute_angle_to_vertical_axis_at_junction(ts: TensorSkeleton, grouped_junction_edge_index):
    junction_edge = ts.pos.detach()[grouped_junction_edge_index, :]
    junction_tangent = junction_edge[:, 1, :] - junction_edge[:, 0, :]
    return torch.atan2(
        junction_tangent[:, 1],
        junction_tangent[:, 0]
    )

def _build_junction_corner_index(ts: TensorSkeleton, grouped_junction_edge_index):
    unique = torch.unique_consecutive(
        grouped_junction_edge_index[:, 0]
    )
    count = ts.degrees[unique]
    junction_end_index = torch.cumsum(count, dim=0)
    return (
        torch.empty(
            (grouped_junction_edge_index.shape[0], 3),
            dtype=torch.long,
            device=ts.path_index.device
        ),
        junction_end_index
    )
