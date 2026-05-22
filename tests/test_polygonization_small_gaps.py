# -*- coding: utf-8 -*-
from types import SimpleNamespace

import numpy as np
import pytest
import shapely.geometry
import torch

from pytorch_segmentation_models_trainer.tools.polygonization import polygonize_utils
from pytorch_segmentation_models_trainer.tools.polygonization.methods import (
    active_contours,
    active_skeletons,
    polygon_rnn_polygonization,
    simple,
)


def test_compute_init_contours_batch_uses_pool():
    class DummyPool:
        def map(self, func, batch):
            return [func(item) for item in batch]

    batch = np.zeros((2, 8, 8), dtype=np.float32)
    batch[:, 2:6, 2:6] = 1

    output = polygonize_utils.compute_init_contours_batch(batch, 0.5, pool=DummyPool())

    assert len(output) == 2


def test_split_polylines_corner_accepts_polygon_linestring_and_array():
    polygon = shapely.geometry.Polygon([(0, 0), (3, 0), (3, 3), (0, 0)])
    line = shapely.geometry.LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    array = np.array([(0, 0), (1, 1), (2, 2), (3, 3)])

    output = polygonize_utils.split_polylines_corner(
        [polygon, line, array],
        [
            np.array([False, True, False, False]),
            np.array([True, False, True, False]),
            np.array([False, False, False, False]),
        ],
    )

    assert len(output) >= 3


def test_compute_geom_prob_iterable_polygon_and_invalid_type():
    prob_map = np.ones((1, 8, 8), dtype=np.float32)
    polygon = shapely.geometry.Polygon([(1, 1), (4, 1), (4, 4), (1, 1)])

    probs = polygonize_utils.compute_geom_prob([polygon], prob_map)

    assert probs == [1.0]
    with pytest.raises(NotImplementedError):
        polygonize_utils.compute_geom_prob(
            shapely.geometry.LineString([(0, 0), (1, 1)]), prob_map
        )


def test_simple_simplify_list_tolerances_and_pool_polygonize():
    polygons = [shapely.geometry.Polygon([(0, 0), (4, 0), (4, 4), (0, 0)])]
    probs = [0.9]

    poly_dict, prob_dict = simple.simplify(polygons, probs, [0, 1])

    assert set(poly_dict) == {"tol_0", "tol_1"}
    assert prob_dict["tol_0"] == probs

    class DummyPool:
        def map(self, func, batch):
            return [func(item) for item in batch]

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    seg = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    seg[:, :, 3:12, 3:12] = 1
    config = SimpleNamespace(
        data_level=0.5,
        min_area=1,
        seg_threshold=0.1,
        tolerance=0,
    )

    polygons_batch, probs_batch = simple.polygonize(seg, config, pool=DummyPool())

    assert len(polygons_batch) == 1
    assert len(probs_batch) == 1

    polygons_batch, probs_batch = simple.polygonize(seg, config, pool=None)

    assert len(polygons_batch) == 1
    assert len(probs_batch) == 1


def test_polygon_rnn_postprocess_tolerance_zero_and_simplify():
    polygon = shapely.geometry.Polygon([(0, 0), (4, 0), (4, 4), (0, 0)])
    config = SimpleNamespace(min_area=1, tolerance=0)

    output = polygon_rnn_polygonization.shapely_postprocess([polygon], config)

    assert output[0].equals(polygon)
    simplified = polygon_rnn_polygonization.simplify([polygon], tolerance=1)
    assert len(simplified) == 1
    postprocessed = polygon_rnn_polygonization.shapely_postprocess(
        [polygon],
        SimpleNamespace(min_area=1, tolerance=1),
    )
    assert len(postprocessed) == 1


def test_polygon_rnn_polygonize_filters_short_polygons(monkeypatch):
    config = SimpleNamespace(grid_size=28, min_area=1, tolerance=0)
    batch = {
        "output_batch_polygons": torch.zeros((2, 4, 2)),
        "scale_h": torch.ones(2),
        "scale_w": torch.ones(2),
        "min_col": torch.zeros(2),
        "min_row": torch.zeros(2),
    }

    monkeypatch.setattr(
        polygon_rnn_polygonization.polygonrnn_utils,
        "get_vertex_list_from_batch_tensors",
        lambda *args, **kwargs: [
            np.array([[0, 0], [1, 1]]),
            np.array([[0, 0], [3, 0], [3, 3], [0, 0]]),
        ],
    )

    output = polygon_rnn_polygonization.polygonize(batch, config)

    assert len(output) == 1


def test_active_contours_stats_shapely_list_tolerance_and_pool_polygonize(monkeypatch):
    contour = np.array([[1, 1], [1, 5], [5, 5], [5, 1], [1, 1]], dtype=np.float32)
    short_contour = contour[:3]
    long_contour = np.vstack([contour, contour[:2]])
    active_contours.print_contours_stats([contour, short_contour, long_contour])

    monkeypatch.setattr(
        active_contours.frame_field_utils,
        "detect_corners",
        lambda contours, _u, _v: [np.zeros(len(contours[0]), dtype=bool)],
    )
    config = SimpleNamespace(min_area=1, seg_threshold=0.0, tolerance=[0, 1])
    indicator = np.ones((8, 8), dtype=np.float32)
    polygons, probs = active_contours.shapely_postprocess(
        [contour],
        np.ones((8, 8), dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
        indicator,
        [0, 1],
        config,
    )
    assert set(polygons) == {"tol_0", "tol_1"}
    assert set(probs) == {"tol_0", "tol_1"}

    class FakeTensorPoly:
        batch_size = 1
        poly_slice = torch.tensor([[0, 3]])
        pos = torch.tensor([[1.0, 1.0], [1.0, 5.0], [5.0, 5.0]])
        is_endpoint = torch.tensor([False, False, False])
        batch = torch.tensor([0, 0, 0])

        def to(self, _device):
            return self

    class FakeOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self):
            return FakeTensorPoly()

    class DummyPool:
        def map(self, func, batch):
            return [func(item) for item in batch]

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    monkeypatch.setattr(
        active_contours, "contours_batch_to_tensorpoly", lambda _batch: FakeTensorPoly()
    )
    monkeypatch.setattr(active_contours, "TensorPolyOptimizer", FakeOptimizer)
    monkeypatch.setattr(
        active_contours,
        "post_process",
        lambda *_args, **_kwargs: (
            [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            [0.9],
        ),
    )
    seg = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    seg[:, :, 2:6, 2:6] = 1
    crossfield = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    config = SimpleNamespace(
        device="cpu",
        data_level=0.5,
        data_coef=0.1,
        length_coef=0.1,
        crossfield_coef=0.1,
        dist_coef=0.1,
    )

    polygons_batch, probs_batch = active_contours.polygonize(
        seg,
        crossfield,
        config,
        pool=DummyPool(),
        pre_computed={"init_contours_batch": [[contour]]},
    )

    assert len(polygons_batch) == 1
    assert probs_batch[0] == [0.9]


def test_active_skeletons_helpers_and_polygonizer_paths(monkeypatch):
    config = SimpleNamespace(
        init_method="marching_squares",
        data_level=0.5,
        min_area=1,
        seg_threshold=0.0,
        tolerance=0,
        device="cpu",
    )
    assert (
        active_skeletons.get_skeleton(
            np.zeros((6, 6), dtype=bool), config
        ).coordinates.shape[0]
        == 0
    )

    monkeypatch.setattr(
        active_skeletons.skan,
        "Skeleton",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert (
        active_skeletons.get_skeleton(
            np.ones((6, 6), dtype=bool), config
        ).coordinates.shape[0]
        == 0
    )

    class BadSkanSkeleton:
        coordinates = np.ones((3, 2), dtype=float)
        degrees = np.ones(2, dtype=np.int64)
        paths = SimpleNamespace(
            indices=np.array([0, 1, 2], dtype=np.int64),
            indptr=np.array([0, 3], dtype=np.int64),
        )

    monkeypatch.setattr(
        active_skeletons.skan, "Skeleton", lambda *_args, **_kwargs: BadSkanSkeleton()
    )
    assert (
        active_skeletons.get_skeleton(
            np.ones((6, 6), dtype=bool), config
        ).coordinates.shape[0]
        == 3
    )

    assert (
        active_skeletons.get_marching_squares_skeleton(
            np.zeros((6, 6), dtype=np.float32), config
        ).coordinates.shape[0]
        == 0
    )
    monkeypatch.setattr(
        active_skeletons.skimage.measure,
        "find_contours",
        lambda *_args, **_kwargs: [np.array([[1, 1], [1, 4], [4, 4]], dtype=float)],
    )
    open_skeleton = active_skeletons.get_marching_squares_skeleton(
        np.ones((6, 6), dtype=np.float32), config
    )
    assert open_skeleton.degrees[0] == 1
    monkeypatch.undo()

    prob = np.zeros((8, 8), dtype=np.float32)
    prob[2:6, 2:6] = 1
    skeleton = active_skeletons.get_marching_squares_skeleton(prob, config)
    assert skeleton.paths.indptr.shape[0] >= 2
    assert len(active_skeletons.skeleton_to_polylines(skeleton)) >= 1

    seg = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    seg[:, :, 2:6, 2:6] = 1

    class DummyPool:
        def map(self, func, batch):
            return [func(item) for item in batch]

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    class BadConfig(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

    with pytest.raises(NotImplementedError):
        active_skeletons.compute_skeletons(
            seg,
            BadConfig(init_method="bad"),
            lambda tensor: tensor.repeat(1, 2, 1, 1),
        )

    skeletons = active_skeletons.compute_skeletons(
        seg[:, :1], config, None, pool=DummyPool()
    )
    assert len(skeletons) == 1

    skeleton_config = SimpleNamespace(
        init_method="skeleton",
        data_level=0.5,
        min_area=1,
        seg_threshold=0.0,
        tolerance=0,
        device="cpu",
    )
    skeletons = active_skeletons.compute_skeletons(
        seg,
        skeleton_config,
        lambda tensor: tensor.repeat(1, 2, 1, 1),
        pool=DummyPool(),
    )
    assert len(skeletons) == 1

    monkeypatch.setattr(
        active_skeletons,
        "post_process",
        lambda *_args, **_kwargs: (
            [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            [1.0],
        ),
    )
    polygonizer = active_skeletons.PolygonizerASM(config, pool=None)
    polygons_batch, probs_batch = polygonizer._skeletons_to_polygons(
        torch.ones((1, 4, 8, 8)),
        torch.ones((1, 8, 8)),
        [skeleton],
    )
    assert len(polygons_batch) == 1

    polygonizer = active_skeletons.PolygonizerASM(config, pool=DummyPool())
    polygons_batch, probs_batch = polygonizer._skeletons_to_polygons(
        torch.ones((1, 4, 8, 8)),
        torch.ones((1, 8, 8)),
        [skeleton],
    )
    assert len(polygons_batch) == 1

    monkeypatch.setattr(
        active_skeletons,
        "compute_skeletons",
        lambda *_args, **_kwargs: [
            active_skeletons.Skeleton(
                coordinates=np.empty((0, 2), dtype=float),
                paths=active_skeletons.Paths(
                    indices=np.empty(0, dtype=np.int64),
                    indptr=np.array([0], dtype=np.int64),
                ),
                degrees=np.empty(0, dtype=np.int64),
            )
        ],
    )
    polygonizer = active_skeletons.PolygonizerASM(config, pool=None)
    polygons_batch, probs_batch = polygonizer(
        torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        torch.ones((1, 4, 8, 8), dtype=torch.float32),
    )
    assert polygons_batch == [[]]
    assert probs_batch == [[]]
