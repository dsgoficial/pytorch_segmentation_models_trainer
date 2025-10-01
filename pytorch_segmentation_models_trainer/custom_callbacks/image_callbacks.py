# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                                     -------------------
        begin                          : 2021-03-10
        git sha                        : $Format:%H$
        copyright                      : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                                     @ Brazilian Army
        email                          : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 * *
 * This program is free software; you can redistribute it and/or modify  *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation; either version 2 of the License, or      *
 * (at your option) any later version.                                    *
 * *
 * ****
"""
import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_segmentation_models_trainer.custom_models.mod_polymapper.modpolymapper import (
    GenericModPolyMapper,
)
from pytorch_segmentation_models_trainer.tools.visualization.base_plot_tools import (
    batch_denormalize_tensor,
    denormalize_np_array,
    generate_bbox_visualization,
    generate_visualization,
    visualize_image_with_bboxes,
)
from pytorch_segmentation_models_trainer.tools.visualization.crossfield_plot import (
    get_tensorboard_image_seg_display,
    plot_polygons,
)
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils


class ImageSegmentationResultCallback(pl.callbacks.Callback):
    def __init__(
        self,
        n_samples: int = None,
        output_path: str = None,
        normalized_input=True,
        norm_params=None,
        log_every_k_epochs=1,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.normalized_input = normalized_input
        self.output_path = None if output_path is None else output_path
        self.norm_params = norm_params if norm_params is not None else {}
        self.save_outputs = False
        self.log_every_k_epochs = log_every_k_epochs

    def prepare_image_to_plot(self, image):
        image = image.squeeze(0) if image.shape[0] == 1 else image
        image = (
            denormalize_np_array(image, **self.norm_params)
            if self.normalized_input
            else image
        )
        return (
            np.moveaxis(image, 0, -1) if min(image.shape) == image.shape[0] else image
        )

    def prepare_mask_to_plot(self, mask):
        return np.squeeze(mask).astype(np.float64)

    def log_data_to_tensorboard(self, saved_image, image_path, logger, current_epoch):
        image = Image.open(saved_image)
        data = np.array(image)
        data = np.moveaxis(data, -1, 0)
        data = torch.from_numpy(data)
        logger.experiment.add_image(image_path, data, current_epoch)

    def save_plot_to_disk(self, plot, image_name, current_epoch):
        image_name = Path(image_name).name.split(".")[0]
        report_path = os.path.join(
            self.output_path,
            "report_image_{name}_epoch_{epoch}_{date}.png".format(
                name=image_name,
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=current_epoch,
            ),
        )
        plot.savefig(report_path, format="png", bbox_inches="tight")
        return report_path

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.save_outputs = True
        self.output_path = os.path.join(trainer.log_dir, "image_logs")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader().dataset
        device = pl_module.device
        logger = trainer.logger
        self.n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        for i in range(self.n_samples):
            image, mask = val_ds[i].values()
            image = image.unsqueeze(0)
            image = image.to(device)
            predicted_mask = pl_module(image)
            image = image.to("cpu")
            predicted_mask = predicted_mask.to("cpu")
            plot_title = val_ds.get_path(i)
            plt_result, fig = generate_visualization(
                fig_title=plot_title,
                image=self.prepare_image_to_plot(image.numpy()),
                ground_truth_mask=self.prepare_mask_to_plot(mask.numpy()),
                predicted_mask=self.prepare_mask_to_plot(predicted_mask.numpy()),
            )
            if self.save_outputs:
                saved_image = self.save_plot_to_disk(
                    plt_result, plot_title, trainer.current_epoch
                )
                self.log_data_to_tensorboard(
                    saved_image, plot_title, logger, trainer.current_epoch
                )
            plt.close(fig)
        return


class FrameFieldResultCallback(ImageSegmentationResultCallback):
    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader()
        device = pl_module.device
        logger = trainer.logger
        n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        current_item = 0
        for batch in val_ds:
            if current_item >= n_samples:
                break
            images = batch["image"]
            image_display = batch_denormalize_tensor(images).to("cpu")
            pred = pl_module(images.to(device))
            for i in range(images.shape[0]):
                if current_item >= n_samples:
                    break
                mask = batch["gt_polygons_image"][i]
                predicted_mask = pred["seg"][i]
                predicted_mask = predicted_mask.to("cpu")
                plot_title = batch["path"][i]
                image_to_plot = np.transpose(image_display[i], (1, 2, 0))
                axarr, fig = generate_visualization(
                    fig_title=plot_title,
                    image=image_to_plot,
                    ground_truth_mask=self.prepare_mask_to_plot(mask.numpy()[0]),
                    predicted_mask=self.prepare_mask_to_plot(predicted_mask.numpy()[0]),
                    ground_truth_boundary=self.prepare_mask_to_plot(mask.numpy()[1]),
                    predicted_boundary=self.prepare_mask_to_plot(
                        predicted_mask.numpy()[1]
                    ),
                )
                fig.tight_layout()
                fig.subplots_adjust(top=0.95)
                if self.save_outputs:
                    saved_image = self.save_plot_to_disk(
                        plt, plot_title, trainer.current_epoch
                    )
                    self.log_data_to_tensorboard(
                        saved_image, plot_title, logger, trainer.current_epoch
                    )
                plt.close(fig)
                current_item += 1
        return


class FrameFieldOverlayedResultCallback(ImageSegmentationResultCallback):
    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader()
        n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        current_item = 0
        for batch in val_ds:
            if current_item >= n_samples:
                break
            images = batch["image"]
            image_display = batch_denormalize_tensor(images).to("cpu")
            pred = pl_module(images.to(pl_module.device))
            crossfield = pred["crossfield"].to("cpu") if "crossfield" in pred else None
            seg = pred["seg"].to("cpu")
            if "seg" not in pred:
                return
            for idx in range(images.shape[0]):
                if current_item >= n_samples:
                    break
                image_seg_display = get_tensorboard_image_seg_display(
                    image_display[idx].unsqueeze(0),
                    seg[idx].unsqueeze(0),
                    crossfield=crossfield[idx].unsqueeze(0)
                    if crossfield is not None
                    else None,
                )
                trainer.logger.experiment.add_images(
                    f"overlay-{batch['path'][idx]}",
                    image_seg_display,
                    trainer.current_epoch,
                )
                current_item += 1


class ObjectDetectionResultCallback(ImageSegmentationResultCallback):
    def __init__(
        self,
        n_samples: int = None,
        output_path: str = None,
        normalized_input=True,
        norm_params=None,
        log_every_k_epochs=1,
        threshold=0.5,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            output_path=output_path,
            normalized_input=normalized_input,
            norm_params=norm_params,
            log_every_k_epochs=log_every_k_epochs,
        )
        self.threshold = threshold

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader()
        n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        current_item = 0
        for images, targets, indexes in val_ds:
            if current_item >= n_samples:
                break
            image_display = batch_denormalize_tensor(
                images, clip_range=[0, 255], output_type=torch.uint8
            ).to("cpu")
            outputs = pl_module(images.to(pl_module.device))
            visualization_list = self.generate_vis_list(
                pl_module, image_display, outputs
            )
            for idx, vis in enumerate(visualization_list):
                trainer.logger.experiment.add_image(
                    val_ds.dataset.get_path(int(indexes[idx])),
                    vis,
                    trainer.current_epoch,
                )
                current_item += 1
        return

    def generate_vis_list(
        self,
        pl_module: pl.LightningModule,
        image_display: torch.Tensor,
        outputs: List[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        boxes: List[torch.Tensor] = [
            out["boxes"][out["scores"] > self.threshold].to("cpu") for out in outputs
        ]
        visualization_list = visualize_image_with_bboxes(image_display.to("cpu"), boxes)
        return visualization_list


class PolygonRNNResultCallback(ImageSegmentationResultCallback):
    def build_polygon_vis(
        self,
        image_path,
        original_image,
        gt_polygon_list,
        predicted_polygon_list,
        trainer,
    ):
        plot_title = image_path
        plt_result, fig = generate_visualization(
            fig_title=plot_title,
            fig_size=(10, 6),
            expected_output=original_image,
            predicted_output=original_image,
        )
        gt_axes, predicted_axes = plt.gcf().get_axes()
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plot_polygons(gt_axes, gt_polygon_list, markersize=5)
        plot_polygons(predicted_axes, predicted_polygon_list, markersize=5)
        saved_image = None
        if self.save_outputs:
            saved_image = self.save_plot_to_disk(plt, plot_title, trainer.current_epoch)
            self.log_data_to_tensorboard(
                saved_image, plot_title, trainer.logger, trainer.current_epoch
            )
        plt.close(fig)
        return saved_image

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader()
        self.n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        prepared_input = val_ds.dataset.get_n_image_path_dict_list(self.n_samples)
        for image_path, prepared_item in prepared_input.items():
            output_batch_polygons = pl_module.model.test(
                prepared_item["croped_images"].to(pl_module.device),
                pl_module.val_seq_len,
            )
            gt_polygon_list = prepared_item["shapely_polygon_list"]
            predicted_polygon_list = (
                polygonrnn_utils.get_vertex_list_from_batch_tensors(
                    output_batch_polygons,
                    prepared_item["scale_h"],
                    prepared_item["scale_w"],
                    prepared_item["min_col"],
                    prepared_item["min_row"],
                    grid_size=val_ds.dataset.grid_size,
                )
            )
            saved_image = self.build_polygon_vis(
                image_path,
                prepared_item["original_image"],
                gt_polygon_list,
                predicted_polygon_list,
                trainer,
            )
        return


class ModPolyMapperResultCallback(PolygonRNNResultCallback):
    def __init__(
        self,
        n_samples: int = None,
        output_path: str = None,
        normalized_input=True,
        norm_params=None,
        log_every_k_epochs=1,
        threshold=0.5,
        show_label_scores=False,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            output_path=output_path,
            normalized_input=normalized_input,
            norm_params=norm_params,
            log_every_k_epochs=log_every_k_epochs,
        )
        self.threshold = threshold
        self.show_label_scores = show_label_scores
        self.n_samples = 16 if n_samples is None else n_samples

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_ds = pl_module.val_dataloader().loaders  # type: ignore
        current_item = 0
        prepared_input = val_ds[
            "polygon_rnn"
        ].loader.dataset.get_n_image_path_dict_list(self.n_samples)
        model: GenericModPolyMapper = pl_module.model  # type: ignore
        grid_size = model.polygonrnn_model.grid_size
        model.eval()  # type: ignore
        aug_func = A.Compose([A.Normalize(), ToTensorV2()])
        for image_path, prepared_item in prepared_input.items():
            if current_item >= self.n_samples:
                break
            image_tensor = aug_func(image=prepared_item["original_image"])["image"].to(
                pl_module.device
            )
            with torch.no_grad():
                outputs = model(
                    image_tensor.unsqueeze(0), threshold=self.threshold
                )  # type: ignore
            self.prepare_and_build_polygonrnn_vis(
                trainer=trainer,
                image_path=image_path,
                prepared_item=prepared_item,
                detection_dict=outputs[0],
                grid_size=grid_size,
            )
            current_item += 1
        return

    def prepare_and_build_polygonrnn_vis(
        self,
        trainer: pl.Trainer,
        image_path: str,
        prepared_item: Dict[str, torch.Tensor],
        detection_dict: Dict[str, torch.Tensor],
        grid_size: int,
    ) -> None:
        gt_polygon_list = prepared_item["shapely_polygon_list"]
        predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            detection_dict["polygonrnn_output"],
            scale_h=detection_dict["scale_h"],
            scale_w=detection_dict["scale_w"],
            min_row=detection_dict["min_row"],
            min_col=detection_dict["min_col"],
            grid_size=grid_size,
        )
        self.build_obj_det_and_polygon_vis(
            image_path=image_path,
            original_image=prepared_item["original_image"],
            detection_dict=detection_dict,
            gt_polygon_list=gt_polygon_list,
            predicted_polygon_list=predicted_polygon_list,
            trainer=trainer,
        )

    def build_obj_det_and_polygon_vis(
        self,
        image_path,
        original_image,
        detection_dict,
        gt_polygon_list,
        predicted_polygon_list,
        trainer,
    ):
        plot_title = image_path
        plt_result, fig = generate_visualization(
            fig_title=plot_title,
            fig_size=(10, 6),
            detected_bboxes=original_image,
            expected_polygons=original_image,
            predicted_polygons=original_image,
        )
        bbox_axes, gt_axes, predicted_axes = plt.gcf().get_axes()
        if detection_dict["boxes"].shape[0] > 0:
            generate_bbox_visualization(
                bbox_axes,
                {
                    k: v.cpu().numpy()
                    for k, v in detection_dict.items()
                    if k in ["boxes", "labels", "scores"]
                },
                show_scores=self.show_label_scores,
                colors=["chartreuse"],
            )
        plot_polygons(gt_axes, gt_polygon_list, markersize=5)
        plot_polygons(predicted_axes, predicted_polygon_list, markersize=5)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        saved_image = None
        if self.save_outputs:
            saved_image = self.save_plot_to_disk(plt, plot_title, trainer.current_epoch)
            self.log_data_to_tensorboard(
                saved_image, plot_title, trainer.logger, trainer.current_epoch
            )
        plt.close(fig)
        return saved_image

class EnhancedImageSegmentationResultCallback(pl.callbacks.Callback):
    def __init__(
        self,
        n_samples: int = None,
        output_path: str = None,
        normalized_input: bool = True,
        norm_params: Optional[Dict] = None,
        log_every_k_epochs: int = 1,
        class_colors: Optional[List[str]] = None,
        colormap: str = 'tab10',
        num_classes: Optional[int] = None,
        band_indices: Optional[List[int]] = None,
        alpha_mask: float = 0.7,
        show_class_legend: bool = True,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Enhanced callback for image segmentation visualization.
        
        Args:
            n_samples: Number of samples to visualize
            output_path: Path to save outputs
            normalized_input: Whether input images are normalized
            norm_params: Normalization parameters (dict with 'mean' and 'std' keys)
            log_every_k_epochs: Frequency of logging
            class_colors: Custom colors for each class (hex codes or color names)
            colormap: Matplotlib colormap name for automatic color generation
            num_classes: Number of segmentation classes
            band_indices: Indices of bands to use for RGB visualization (default: [0,1,2])
            alpha_mask: Transparency level for mask overlay (0-1)
            show_class_legend: Whether to show class legend in plots
            class_names: Names for each class (for legend)
        """
        super().__init__()
        self.n_samples = n_samples
        self.normalized_input = normalized_input
        self.output_path = None if output_path is None else output_path
        self.norm_params = norm_params if norm_params is not None else {}
        self.save_outputs = False
        self.log_every_k_epochs = log_every_k_epochs
        
        # Color and class configuration
        self.colormap_name = colormap
        self.num_classes = num_classes
        self.alpha_mask = alpha_mask
        self.show_class_legend = show_class_legend
        self.class_names = class_names
        
        # Band selection for RGB visualization
        self.band_indices = band_indices if band_indices is not None else [0, 1, 2]
        
        # Setup class colors and create matplotlib colormap
        self.class_colors = self._setup_class_colors(class_colors, num_classes)
        self.cmap, self.norm = self._create_colormap()

    def _setup_class_colors(self, class_colors: Optional[List[str]], num_classes: Optional[int]) -> List[str]:
        """Setup colors for each class."""
        if class_colors is not None:
            return class_colors
        
        if num_classes is None:
            num_classes = 10
            
        cmap = plt.get_cmap(self.colormap_name)
        colors = []
        
        is_categorical = self.colormap_name in [
            'tab10', 'tab20', 'tab20b', 'tab20c', 
            'Set1', 'Set2', 'Set3', 'Paired', 
            'Accent', 'Dark2', 'Pastel1', 'Pastel2'
        ]
        
        for i in range(num_classes):
            if is_categorical:
                rgba_color = cmap(i % cmap.N)
            else:
                if num_classes == 1:
                    color_val = 0.5
                else:
                    color_val = 0.1 + (i / (num_classes - 1)) * 0.8
                rgba_color = cmap(color_val)
            
            hex_color = mcolors.rgb2hex(rgba_color[:3])
            colors.append(hex_color)
        
        return colors

    def _create_colormap(self):
        """Create matplotlib ListedColormap from class colors."""
        # Convert hex colors to RGB tuples (normalized to 0-1)
        colors_rgb = [mcolors.to_rgb(color) for color in self.class_colors]
        
        # Create ListedColormap
        cmap = ListedColormap(colors_rgb)
        
        # Use same normalization as the working notebook example
        # For n classes (0, 1, 2, ..., n-1), use vmin=0, vmax=n
        num_colors = len(self.class_colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_colors)
        
        return cmap, norm

    def prepare_image_to_plot(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for plotting, ensuring RGB format."""
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        # Detect channel order
        if image.shape[0] <= 20 and image.shape[0] < min(image.shape[1], image.shape[2]):
            channels_first = True
        else:
            channels_first = False
        
        # Ensure channels-first for denormalization
        if not channels_first:
            image = np.moveaxis(image, -1, 0)
        
        # Denormalize
        if self.normalized_input:
            image = denormalize_np_array(image, **self.norm_params)
        
        # Convert to channels-last
        image = np.moveaxis(image, 0, -1)
        
        num_channels = image.shape[-1]
        
        # Select RGB bands
        if num_channels == 1:
            image = np.repeat(image, 3, axis=-1)
        elif num_channels == 2:
            third_channel = np.zeros_like(image[:, :, 0:1])
            image = np.concatenate([image, third_channel], axis=-1)
        elif num_channels >= 3:
            valid_indices = [idx for idx in self.band_indices if idx < num_channels]
            if len(valid_indices) >= 3:
                image = image[:, :, valid_indices[:3]]
            else:
                image = image[:, :, :3]
        
        # Normalize to [0, 1] using min-max scaling
        image_min = image.min()
        image_max = image.max()
        
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
        
        return image

    def prepare_mask_to_plot(self, mask: np.ndarray) -> np.ndarray:
        """Prepare segmentation mask for plotting (return class indices)."""
        if len(mask.shape) == 4:
            mask = mask.squeeze(0)
        
        # Handle multi-channel masks (convert to class indices)
        if len(mask.shape) == 3:
            if mask.shape[0] <= 20 and mask.shape[0] < min(mask.shape[1], mask.shape[2]):
                mask = np.argmax(mask, axis=0)
            else:
                mask = np.argmax(mask, axis=-1)
        
        # Return integer class indices (0, 1, 2, ..., num_classes-1)
        # Do NOT normalize - the colormap expects integer values
        mask = np.squeeze(mask).astype(np.uint8)
        return mask

    def apply_colormap_to_axes(self, axarr, mask_indices: List[int], masks: List[np.ndarray]):
        """Apply colormap to specific axes after generate_visualization."""
        for idx, mask in zip(mask_indices, masks):
            ax = axarr[idx] if isinstance(axarr, np.ndarray) else axarr
            # Clear the axis and replot with colormap
            ax.clear()
            ax.imshow(mask, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            # Reconstruct the title
            if idx == 1:
                ax.set_title("Ground Truth Mask")
            elif idx == 2:
                ax.set_title("Predicted Mask")

    def add_colorbar_legend(self, fig, axarr, mask: np.ndarray):
        """Add colorbar with class labels for ALL classes in class_names."""
        # Get the last axis for colorbar
        ax = axarr[-1] if isinstance(axarr, np.ndarray) else axarr
        
        # Create a dummy mappable for colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        
        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        
        # Set tick labels for ALL classes, not just unique ones in mask
        num_classes = len(self.class_colors)
        ticks = list(range(num_classes))
        
        if self.class_names and len(self.class_names) == num_classes:
            labels = self.class_names
        else:
            labels = [f'Class {c}' for c in ticks]
        
        # Set ticks at integer positions (0, 1, 2, ..., n-1)
        cbar.set_ticks([t + 0.5 for t in ticks])
        cbar.set_ticklabels(labels)

    def log_data_to_tensorboard(self, saved_image: str, image_path: str, logger, current_epoch: int):
        """Log visualization to tensorboard."""
        image = Image.open(saved_image)
        data = np.array(image)
        data = np.moveaxis(data, -1, 0)
        data = torch.from_numpy(data)
        logger.experiment.add_image(image_path, data, current_epoch)

    def save_plot_to_disk(self, plot, image_name: str, current_epoch: int) -> str:
        """Save plot to disk."""
        image_name = Path(image_name).name.split(".")[0]
        report_path = os.path.join(
            self.output_path,
            "report_image_{name}_epoch_{epoch}_{date}.png".format(
                name=image_name,
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=current_epoch,
            ),
        )
        plot.savefig(report_path, format="png", bbox_inches="tight", dpi=150)
        return report_path

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Setup output directory after sanity check."""
        self.save_outputs = True
        self.output_path = os.path.join(trainer.log_dir, "image_logs")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("CALLBACK INITIALIZATION - Color Configuration:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of class colors: {len(self.class_colors)}")
        print(f"Class colors list: {self.class_colors}")
        print(f"Colormap: {self.colormap_name}")
        print(f"Class names: {self.class_names}")
        print(f"Normalization: vmin={self.norm.vmin}, vmax={self.norm.vmax}")
        print("-" * 80)
        print("Expected mask values: 0 to", self.num_classes - 1, "(integers)")
        print("Color mapping:")
        for i, (color, name) in enumerate(zip(self.class_colors, self.class_names or [f'Class {i}' for i in range(len(self.class_colors))])):
            print(f"  {i} -> {name}: {color}")
        print("=" * 80)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and save visualizations at validation end."""
        if not self.save_outputs or trainer.current_epoch % self.log_every_k_epochs != 0:
            return
            
        val_ds = pl_module.val_dataloader().dataset
        device = pl_module.device
        logger = trainer.logger
        
        self.n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        
        for i in range(min(self.n_samples, len(val_ds))):
            # Get sample
            sample = val_ds[i]
            if isinstance(sample, dict):
                image = sample['image']
                mask = sample.get('mask', sample.get('target', None))
                plot_title = sample.get('path', val_ds.get_path(i) if hasattr(val_ds, 'get_path') else f'sample_{i}')
            else:
                image, mask = sample.values() if hasattr(sample, 'values') else sample
                plot_title = val_ds.get_path(i) if hasattr(val_ds, 'get_path') else f'sample_{i}'
            
            # Get prediction
            image_tensor = image.unsqueeze(0).to(device)
            predicted_mask = pl_module(image_tensor)
            predicted_mask = predicted_mask.to("cpu")
            
            # Prepare images for visualization
            image_rgb = self.prepare_image_to_plot(image.numpy())
            
            gt_mask_indices = self.prepare_mask_to_plot(mask.numpy())
            
            pred_mask_indices = self.prepare_mask_to_plot(predicted_mask.numpy())
            
            # Use the existing generate_visualization function
            axarr, fig = generate_visualization(
                fig_title=plot_title,
                image=image_rgb,
                ground_truth_mask=gt_mask_indices,  # Pass indices, will replot with colormap
                predicted_mask=pred_mask_indices,    # Pass indices, will replot with colormap
            )
            
            # Apply colormap to the mask axes (indices 1 and 2)
            self.apply_colormap_to_axes(
                axarr, 
                mask_indices=[1, 2], 
                masks=[gt_mask_indices, pred_mask_indices]
            )
            
            # Add class legend if requested
            if self.show_class_legend and self.class_names is not None:
                self.add_colorbar_legend(fig, axarr, gt_mask_indices)
            
            fig.tight_layout()
            
            # Save and log
            if self.save_outputs:
                saved_image = self.save_plot_to_disk(fig, plot_title, trainer.current_epoch)
                self.log_data_to_tensorboard(
                    saved_image, plot_title, logger, trainer.current_epoch
                )
            
            plt.close(fig)
        
        return
