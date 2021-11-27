# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-10
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
 ****
"""
import datetime
import os
from pathlib import Path
from typing import Dict, List

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_toolbelt.utils import image_to_tensor, rgb_image_from_tensor
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
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


class ImageSegmentationResultCallback(pl.callbacks.base.Callback):
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

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_outputs = True
        self.output_path = os.path.join(trainer.log_dir, "image_logs")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
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
    def on_validation_end(self, trainer, pl_module):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader()
        device = pl_module.device
        logger = trainer.logger
        batch = next(iter(val_ds))
        image_display = batch_denormalize_tensor(batch["image"]).to("cpu")
        pred = pl_module(batch["image"].to(device))
        self.n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        for i in range(self.n_samples):
            image = image_display[i].numpy()
            mask = batch["gt_polygons_image"][i]
            predicted_mask = pred["seg"][i]
            predicted_mask = predicted_mask.to("cpu")
            plot_title = val_ds.dataset.get_path(i)
            crossfield = pred["crossfield"]
            np_crossfield = crossfield.cpu().detach().numpy().transpose(0, 2, 3, 1)
            image_to_plot = np.transpose(image, (1, 2, 0))
            # image_plot_crossfield_list = [
            #     get_image_plot_crossfield(
            #         _crossfield, crossfield_stride=10, width=0.2
            #     )
            #     for _crossfield in np_crossfield
            # ]
            axarr, fig = generate_visualization(
                fig_title=plot_title,
                image=image_to_plot,
                ground_truth_mask=self.prepare_mask_to_plot(mask.numpy()[0]),
                predicted_mask=self.prepare_mask_to_plot(predicted_mask.numpy()[0]),
                ground_truth_boundary=self.prepare_mask_to_plot(mask.numpy()[1]),
                predicted_boundary=self.prepare_mask_to_plot(predicted_mask.numpy()[1]),
                # crossfield=image_plot_crossfield_list[i],
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
        return


class FrameFieldOverlayedResultCallback(ImageSegmentationResultCallback):
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
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
    def on_validation_end(self, trainer, pl_module):
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
    def on_validation_end(self, trainer, pl_module):
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
            predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
                output_batch_polygons,
                prepared_item["scale_h"],
                prepared_item["scale_w"],
                prepared_item["min_col"],
                prepared_item["min_row"],
                grid_size=val_ds.dataset.grid_size,
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

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_ds = pl_module.val_dataloader().loaders  # type: ignore
        self.n_samples = (
            val_ds["object_detection"].loader.batch_size
            if self.n_samples is None
            else self.n_samples
        )
        current_item = 0
        prepared_input = val_ds[
            "polygon_rnn"
        ].loader.dataset.get_n_image_path_dict_list(self.n_samples)
        model = pl_module.model
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
            )
            current_item += 1
        return

    def prepare_and_build_polygonrnn_vis(
        self,
        trainer: pl.Trainer,
        image_path: str,
        prepared_item: Dict[str, torch.Tensor],
        detection_dict: Dict[str, torch.Tensor],
    ) -> None:
        gt_polygon_list = prepared_item["shapely_polygon_list"]
        predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            detection_dict["polygonrnn_output"],
            scale_h=detection_dict["scale_h"],
            scale_w=detection_dict["scale_w"],
            min_row=detection_dict["min_row"],
            min_col=detection_dict["min_col"],
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
