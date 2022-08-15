# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-08
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
 ****
"""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from affine import Affine
import torch
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    AbstractDataWriter,
    VectorDatabaseDataWriter,
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.tools.polygonization.methods import (
    active_contours,
    active_skeletons,
    simple,
    polygon_rnn_polygonization,
)
from pytorch_segmentation_models_trainer.utils.polygon_utils import (
    coerce_polygons_to_single_geometry,
    polygons_to_world_coords,
)
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


@dataclass
class TemplatePolygonizerProcessor(ABC):
    data_writer: AbstractDataWriter = field(default_factory=VectorFileDataWriter)

    @abstractmethod
    def __post_init__(self):
        """Must be reimplemented in each child.
        self.polygonize_method must be set.
        """
        pass

    def process(
        self,
        inference: Dict[str, Union[np.ndarray, torch.Tensor]],
        profile: dict,
        pool: ThreadPoolExecutor = None,
        parent_dir_name: str = None,
        convert_output_to_world_coords: bool = True,
    ):
        """Processes the polygonization.

        Args:
            inference (Dict[str, np.array]): numpy inference from the neural network.
            pool (concurrent.futures.ThreadPool, optional): Thread object in case of parallel execution.
                Defaults to None.
        """
        try:
            out_contours_batch, out_probs_batch = self.polygonize_method(
                inference["seg"], inference["crossfield"], self.config
            )
        except Exception as e:
            # logger.exception(e)
            out_contours_batch = []
            for idx, (seg, crossfield) in enumerate(
                zip(inference["seg"], inference["crossfield"])
            ):
                try:
                    out_contours, _ = self.polygonize_method(
                        seg.unsqueeze(0), crossfield.unsqueeze(0), self.config
                    )
                    out_contours_batch.append(out_contours[0])
                except Exception as e1:
                    logger.exception(
                        f"An error occurred while polygonizing the image {parent_dir_name[idx]}. Skipping this image."
                    )
                    logger.exception(e1)

        if inference["seg"].shape[0] == 1:
            return self.post_process(
                out_contours_batch[0],
                profile,
                parent_dir_name=parent_dir_name[0]
                if parent_dir_name is not None and len(parent_dir_name)
                else None,
                convert_output_to_world_coords=convert_output_to_world_coords,
            )
        # ignore profile for now, just wanna get some results, I'll fix it later
        if pool is None and profile is None:
            return [
                self.post_process(
                    out_contour,
                    None,
                    parent_dir_name=parent_dir,
                    convert_output_to_world_coords=convert_output_to_world_coords,
                    np_crossfield=np_crossfield,
                    np_indicator=np_indicator,
                )
                for out_contour, parent_dir, np_crossfield, np_indicator in zip(
                    out_contours_batch,
                    parent_dir_name,
                    inference["crossfield"],
                    inference["seg"],
                )
            ]
        futures = []
        profile_list = len(out_contours_batch) * [None] if profile is None else profile
        for out_contour, parent_dir, profile in zip(
            out_contours_batch, parent_dir_name, profile_list
        ):
            futures.append(
                pool.submit(
                    self.post_process,
                    out_contour,
                    profile,
                    parent_dir,
                    convert_output_to_world_coords=convert_output_to_world_coords,
                )
            )
        return futures

    def post_process(
        self,
        polygons: List[Polygon],
        profile: dict,
        parent_dir_name: str = None,
        convert_output_to_world_coords: bool = True,
    ):
        """Post-processes generated polygons from process method.

        Args:
            polygons (List[Polygon]): list of shapely polygons
        """
        if profile is None:
            profile = {"crs": None}
        projected_polygons = (
            polygons_to_world_coords(
                polygons,
                transform=profile["transform"]
                if profile["crs"] is not None
                else Affine(1, 0, 0, 0, -1, 0),
                epsg_number=profile["crs"].to_epsg()
                if profile["crs"] is not None
                else None,
            )
            if convert_output_to_world_coords
            else coerce_polygons_to_single_geometry(polygons)
        )
        if self.data_writer is not None:
            self.data_writer.write_data(
                projected_polygons, profile, folder_name=parent_dir_name
            )
        return projected_polygons


@dataclass
class LossParamsCoefs:
    step_thresholds: list = field(default_factory=lambda: [0, 100, 200, 300])
    data: list = field(default_factory=lambda: [1.0, 0.1, 0.0, 0.0])
    crossfield: list = field(default_factory=lambda: [0.0, 0.05, 0.0, 0.0])
    length: list = field(default_factory=lambda: [0.1, 0.01, 0.0, 0.0])
    curvature: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    corner: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    junction: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class LossParams:
    coefs: LossParamsCoefs = field(default_factory=LossParamsCoefs)
    curvature_dissimilarity_threshold: int = 15
    corner_angles: list = field(default_factory=lambda: [45, 90, 135])
    corner_angle_threshold: float = 22.5
    junction_angles: list = field(default_factory=lambda: [0, 45, 90, 135])
    junction_angle_weights: list = field(default_factory=lambda: [1, 0.01, 0.1, 0.01])
    junction_angle_threshold: float = 22.5


@dataclass
class ASMConfig:
    init_method: str = "skeleton"  # skeleton or marching_squares
    data_level: float = 0.5
    loss_params: LossParams = field(default_factory=LossParams)
    lr: float = 0.001
    gamma: float = 0.0001
    device: str = "cpu"
    tolerance: float = 22
    seg_threshold: float = 0.5
    min_area: float = 12


@dataclass
class ASMPolygonizerProcessor(TemplatePolygonizerProcessor):
    config: ASMConfig = field(default_factory=ASMConfig)

    def __post_init__(self):
        self.polygonize_method = active_skeletons.polygonize


@dataclass
class InnerPolylinesParams:
    enable: bool = False
    max_traces: int = 1000
    seed_threshold: float = 0.5
    low_threshold: float = 0.1
    min_width: int = 2
    max_width: int = 8
    step_size: int = 1


@dataclass
class ACMConfig:
    indicator_add_edge: bool = False
    steps: int = 500
    data_level: float = 0.5
    data_coef: float = 0.1
    length_coef: float = 0.4
    crossfield_coef: float = 0.5
    poly_lr: float = 0.01
    warmup_iters: int = 100
    warmup_factor: float = 0.1
    device: str = "cpu"
    tolerance: float = 0.5
    seg_threshold: float = 0.5
    min_area: int = 1
    inner_polylines_params: InnerPolylinesParams = field(
        default_factory=InnerPolylinesParams
    )


@dataclass
class ACMPolygonizerProcessor(TemplatePolygonizerProcessor):
    config: ACMConfig = field(default_factory=ACMConfig)

    def __post_init__(self):
        self.polygonize_method = active_contours.polygonize


@dataclass
class SimplePolConfig:
    data_level: float = 0.5
    tolerance: float = 1.0
    seg_threshold: float = 0.5
    min_area: float = 10


@dataclass
class SimplePolygonizerProcessor(TemplatePolygonizerProcessor):
    config: SimplePolConfig = field(default_factory=SimplePolConfig)

    def __post_init__(self):
        self.polygonize_method = simple.polygonize

    def process(
        self,
        inference: Dict[str, np.ndarray],
        profile: dict,
        pool: ThreadPoolExecutor = None,
        parent_dir_name: str = None,
    ):
        """Processes the polygonization. Reimplemented from template due to signature
        differences on polygonize method.

        Args:
            inference (Dict[str, np.array]): numpy inference from the neural network.
            pool (concurrent.futures.ThreadPool, optional): Thread object in case of
            parallel execution. Defaults to None.
        """
        out_contours_batch, out_probs_batch = self.polygonize_method(
            inference["seg"], self.config, pool=pool
        )
        return self.post_process(
            out_contours_batch[0], profile, parent_dir_name=parent_dir_name
        )


@dataclass
class PolygonRNNConfig:
    tolerance: float = 0.0
    grid_size: int = 28
    min_area: float = 10


@dataclass
class PolygonRNNPolygonizerProcessor(TemplatePolygonizerProcessor):
    config: PolygonRNNConfig = field(default_factory=PolygonRNNConfig)

    def __post_init__(self):
        self.polygonize_method = polygon_rnn_polygonization.polygonize

    def process(
        self,
        inference: Dict[str, Union[torch.Tensor, np.ndarray]],
        profile: dict,
        pool: ThreadPoolExecutor = None,
        parent_dir_name: str = None,
        convert_output_to_world_coords: bool = True,
    ):
        """Processes the polygonization. Reimplemented from template due to signature
        differences on polygonize method.

        Args:
            inference (Dict[str, np.array]): numpy inference from the neural network.
            pool (concurrent.futures.ThreadPool, optional): Thread object in case of
            parallel execution. Defaults to None.
        """
        out_contours_batch = self.polygonize_method(inference, self.config, pool=pool)
        return self.post_process(
            out_contours_batch,
            profile,
            parent_dir_name=parent_dir_name,
            convert_output_to_world_coords=convert_output_to_world_coords,
        )
