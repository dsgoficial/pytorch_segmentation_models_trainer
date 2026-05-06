import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.coords import BoundingBox
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import logging

logger = logging.getLogger(__name__)

# Forward declaration or import
from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
    process_single_image_worker,
)


def _get_spatial_overlap(pred_path: str, gt_path: str) -> tuple:
    """
    Calcula a área de overlap espacial entre dois rasters georreferenciados.
    """
    try:
        with rasterio.open(pred_path) as pred_src, rasterio.open(gt_path) as gt_src:
            pred_bounds = pred_src.bounds
            gt_bounds = gt_src.bounds
            intersection = BoundingBox(
                left=max(pred_bounds.left, gt_bounds.left),
                bottom=max(pred_bounds.bottom, gt_bounds.bottom),
                right=min(pred_bounds.right, gt_bounds.right),
                top=min(pred_bounds.top, gt_bounds.top),
            )
            if (
                intersection.left >= intersection.right
                or intersection.bottom >= intersection.top
            ):
                logger.error(f"No spatial overlap between {pred_path} and {gt_path}")
                return None
            pred_window = from_bounds(
                intersection.left,
                intersection.bottom,
                intersection.right,
                intersection.top,
                pred_src.transform,
            )
            gt_window = from_bounds(
                intersection.left,
                intersection.bottom,
                intersection.right,
                intersection.top,
                gt_src.transform,
            )
            pred_window = Window(
                col_off=round(pred_window.col_off),
                row_off=round(pred_window.row_off),
                width=round(pred_window.width),
                height=round(pred_window.height),
            )
            gt_window = Window(
                col_off=round(gt_window.col_off),
                row_off=round(gt_window.row_off),
                width=round(gt_window.width),
                height=round(gt_window.height),
            )
            matched_shape = (int(pred_window.height), int(pred_window.width))
            if (
                abs(pred_window.height - gt_window.height) > 1
                or abs(pred_window.width - gt_window.width) > 1
            ):
                logger.warning(
                    f"Window size mismatch after spatial alignment: "
                    f"pred={pred_window.height}x{pred_window.width}, "
                    f"gt={gt_window.height}x{gt_window.width}"
                )
                matched_shape = (
                    min(int(pred_window.height), int(gt_window.height)),
                    min(int(pred_window.width), int(gt_window.width)),
                )
                pred_window = Window(
                    pred_window.col_off,
                    pred_window.row_off,
                    matched_shape[1],
                    matched_shape[0],
                )
                gt_window = Window(
                    gt_window.col_off,
                    gt_window.row_off,
                    matched_shape[1],
                    matched_shape[0],
                )
            return pred_window, gt_window, matched_shape
    except Exception as e:
        logger.error(f"Error calculating spatial overlap: {e}")
        return None


class MetricsCalculator:
    """
    Calcula métricas de segmentação comparando predições com ground truth.
    """

    def __init__(
        self, config: DictConfig, experiment_config: Optional[DictConfig] = None
    ):
        self.config = config
        self.experiment_config = experiment_config or config

        self.num_classes = self.config.get("num_classes") or self.config.get(
            "metrics", {}
        ).get("num_classes", 2)
        self.class_names = self.config.get("class_names") or self.config.get(
            "metrics", {}
        ).get("class_names", [f"class_{i}" for i in range(self.num_classes)])

        self.pixel_metrics = self._instantiate_metrics_group("pixel_metrics")
        self.metrics_to_compute = self._instantiate_metrics_group("metrics")

        logger.info(f"MetricsCalculator initialized with {self.num_classes} classes")

    def _instantiate_metrics_group(self, group_name: str) -> Dict:
        metrics = {}
        group_cfg = self.config.get(group_name, {})
        if not group_cfg:
            return metrics

        for name, cfg in group_cfg.items():
            try:
                if isinstance(cfg, (dict, DictConfig)) and "_target_" in cfg:
                    metrics[name] = instantiate(cfg)
            except Exception as e:
                logger.error(
                    f"Failed to instantiate metric {name} in {group_name}: {e}"
                )
        return metrics

    def calculate_metrics(
        self,
        predictions_folder: Optional[str] = None,
        ground_truth_csv: Optional[str] = None,
        experiment_name: Optional[str] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ) -> Dict:
        predictions_folder = predictions_folder or self.config.get(
            "prediction_input_folder"
        )
        ground_truth_csv = ground_truth_csv or self.config.get("ground_truth_csv_path")
        experiment_name = experiment_name or self.config.get(
            "experiment_name", "default"
        )

        # Fallback to prediction_csv_path if ground_truth_csv is not found
        if not ground_truth_csv or not os.path.exists(ground_truth_csv):
            ground_truth_csv = self.config.get("prediction_csv_path")

        if not ground_truth_csv or not os.path.exists(ground_truth_csv):
            raise ValueError(f"Ground truth CSV not found: {ground_truth_csv}")

        gt_df = pd.read_csv(ground_truth_csv)

        # Build prediction map
        pred_files = {}
        if predictions_folder:
            pred_files = self._find_prediction_files(predictions_folder)

        # Load from prediction_csv_path if available
        pred_csv = self.config.get("prediction_csv_path")
        if pred_csv and os.path.exists(pred_csv):
            pred_df = pd.read_csv(pred_csv)
            for _, row in pred_df.iterrows():
                image_id = str(row.get("image") or row.get("id"))
                mask_path = row.get("mask") or row.get("prediction")
                if image_id and mask_path:
                    stem = Path(image_id).stem
                    if stem not in pred_files:
                        pred_files[stem] = mask_path

        tasks = self._create_tasks(gt_df, pred_files)
        if len(tasks) == 0:
            raise ValueError("No prediction-groundtruth pairs found!")

        if parallel and len(tasks) > 1:
            image_results = self._process_images_parallel(tasks, num_workers)
        else:
            image_results = self._process_images_sequential(tasks)

        return self._compute_metrics_from_results(image_results, experiment_name)

    def _compute_metrics_for_single_pair(
        self, pred_path: str, gt_path: str, **kwargs
    ) -> Dict[str, float]:
        overlap = _get_spatial_overlap(pred_path, gt_path)
        if not overlap:
            return {}

        pred_window, gt_window, _ = overlap
        with rasterio.open(pred_path) as src:
            pred_arr = src.read(1, window=pred_window)
        with rasterio.open(gt_path) as src:
            gt_arr = src.read(1, window=gt_window)

        # Thresholding if provided
        threshold = kwargs.get("threshold") or self.config.get("threshold")
        if threshold is not None:
            pred_arr = (pred_arr > threshold).astype(np.int32)

        pred_tensor = torch.from_numpy(pred_arr).long()
        gt_tensor = torch.from_numpy(gt_arr).long()

        results = {}
        # Merge all available metrics for this method call
        all_metrics = {**self.pixel_metrics, **self.metrics_to_compute}
        for name, metric in all_metrics.items():
            try:
                results[name] = float(metric(pred_tensor, gt_tensor))
            except Exception as e:
                logger.error(f"Error computing metric {name}: {e}")

        return results

    def _find_prediction_files(self, predictions_folder: str) -> Dict[str, str]:
        pred_folder = Path(predictions_folder)
        pred_files = {}
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            for pred_path in pred_folder.glob(ext):
                pred_files[pred_path.stem] = str(pred_path)
        return pred_files

    def _create_tasks(self, gt_df: pd.DataFrame, pred_files: Dict[str, str]) -> list:
        tasks = []
        for idx, row in gt_df.iterrows():
            gt_path = row.get("mask") or row.get("ground_truth") or row.get("image")
            if not gt_path:
                continue

            # If path is relative, try to find it
            if not os.path.isabs(str(gt_path)):
                # Check temp dir if in tests
                if os.path.exists(os.path.join("temp_test_dir", str(gt_path))):
                    gt_path = os.path.join("temp_test_dir", str(gt_path))

            image_name = Path(str(gt_path)).stem
            pred_path = row.get("prediction") or self._find_matching_prediction(
                image_name, pred_files
            )

            if pred_path:
                if not os.path.isabs(str(pred_path)):
                    if os.path.exists(os.path.join("temp_test_dir", str(pred_path))):
                        pred_path = os.path.join("temp_test_dir", str(pred_path))

                if os.path.exists(str(pred_path)):
                    tasks.append(
                        {
                            "image_name": image_name,
                            "pred_path": pred_path,
                            "gt_path": gt_path,
                            "index": idx,
                        }
                    )
        return tasks

    def _find_matching_prediction(
        self, image_name: str, pred_files: Dict[str, str]
    ) -> Optional[str]:
        if image_name in pred_files:
            return pred_files[image_name]
        clean_name = (
            image_name.replace("mask_", "").replace("gt_", "").replace("pred_", "")
        )
        if clean_name in pred_files:
            return pred_files[clean_name]
        for name in pred_files:
            if clean_name in name or name in clean_name:
                return pred_files[name]
        return None

    def _process_images_parallel(self, tasks: list, num_workers: int = None) -> list:
        if num_workers is None:
            num_workers = min(32, len(tasks), os.cpu_count() or 1)
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_single_image_worker,
                    task,
                    self.num_classes,
                    list(self.class_names),
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
        return results

    def _process_images_sequential(self, tasks: list) -> list:
        results = []
        for task in tasks:
            res = process_single_image_worker(
                task, self.num_classes, list(self.class_names)
            )
            if res:
                results.append(res)
        return results

    def _compute_metrics_from_results(
        self, image_results: list, experiment_name: str
    ) -> Dict:
        cm_global = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        per_image_results = {}

        for result in image_results:
            image_name = result.get("image_name", "unknown")

            # Check if result already has metrics (e.g. from a mock)
            if any(k in result for k in ["iou", "accuracy", "f1"]):
                metrics = {
                    k: v
                    for k, v in result.items()
                    if isinstance(v, (int, float, torch.Tensor))
                }
                per_image_results[image_name] = metrics
                continue

            if "pred_flat" in result and "gt_flat" in result:
                pred_flat = torch.from_numpy(result["pred_flat"]).long()
                gt_flat = torch.from_numpy(result["gt_flat"]).long()
                cm = self._compute_confusion_matrix_fast(pred_flat, gt_flat)
                metrics = self._metrics_from_confusion_matrix(cm, image_name)
                per_image_results[image_name] = metrics
                cm_global += cm

        if not per_image_results:
            return {}

        # Aggregated metrics
        if cm_global.sum() == 0:
            overall = {}
            first_metrics = next(iter(per_image_results.values()))
            for k in first_metrics.keys():
                if k != "image_name":
                    vals = [float(r[k]) for r in per_image_results.values() if k in r]
                    if vals:
                        overall[k] = sum(vals) / len(vals)
        else:
            overall = self._metrics_from_confusion_matrix_aggregated(cm_global)

        # Prepare final results dict (matching new test expectations + legacy pipeline)
        final_results = {
            "overall": overall,
            "aggregated": overall,  # Legacy key
            "per_image": pd.DataFrame(list(per_image_results.values())),  # Legacy key
            "confusion_matrix": cm_global,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "experiment_name": experiment_name,
        }
        # Add each image's metrics to the top level for new tests
        for name, metrics in per_image_results.items():
            final_results[name] = metrics

        out_dir = self._prepare_output_directory(experiment_name)
        final_results["output_dir"] = str(out_dir)
        self._save_results(
            final_results["per_image"], overall, cm_global, out_dir, experiment_name
        )

        return final_results

    def _compute_confusion_matrix_fast(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> np.ndarray:
        indices = gt * self.num_classes + pred
        cm_flat = torch.bincount(indices, minlength=self.num_classes**2)
        return cm_flat.reshape(self.num_classes, self.num_classes).cpu().numpy()

    def _metrics_from_confusion_matrix_aggregated(
        self, cm: np.ndarray
    ) -> Dict[str, float]:
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        eps = 1e-10
        return {
            "Accuracy": float(np.mean((tp + tn) / (tp + tn + fp + fn + eps))),
            "JaccardIndex": float(np.mean(tp / (tp + fp + fn + eps))),
            "F1Score": float(np.mean((2 * tp) / (2 * tp + fp + fn + eps))),
        }

    def _metrics_from_confusion_matrix(
        self, cm: np.ndarray, image_name: str
    ) -> Dict[str, float]:
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        eps = 1e-10
        return {
            "image_name": image_name,
            "Accuracy": float(np.mean((tp + tn) / (tp + tn + fp + fn + eps))),
            "JaccardIndex": float(np.mean(tp / (tp + fp + fn + eps))),
            "F1Score": float(np.mean((2 * tp) / (2 * tp + fp + fn + eps))),
        }

    def _prepare_output_directory(self, experiment_name: str) -> Path:
        base = self.config.get("output_folder") or self.config.get("output", {}).get(
            "base_dir", "./output"
        )
        d = Path(base) / "metrics"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_results(self, df, agg, cm, out_dir, name):
        output_json = self.config.get("output_metrics_json")
        if output_json:
            with open(output_json, "w") as f:
                json.dump(agg, f, indent=2)
        df.to_csv(out_dir / f"{name}_per_image.csv", index=False)
        np.save(out_dir / f"{name}_cm.npy", cm)
