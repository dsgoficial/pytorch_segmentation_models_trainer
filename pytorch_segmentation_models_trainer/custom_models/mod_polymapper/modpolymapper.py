# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-11-16
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
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import RoIAlign
from pytorch_segmentation_models_trainer.custom_models.rnn.polygon_rnn import (
    make_basic_conv_block,
    PolygonRNN,
    ConvLSTM,
)
from pytorch_segmentation_models_trainer.custom_models.models import (
    ObjectDetectionModel,
)
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils
from typing import List, Tuple, Union, Dict, Optional


class RoiAlignBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_scale: float,
        roi_output: int,
        kernel_size: Tuple = (3, 3),
        apply_pooling: bool = False,
        upsample_factor: int = 1.0,
    ):
        super().__init__()
        assert upsample_factor >= 1, "upsample_factor must be equal or greater than 1.0"
        self.roi_align = RoIAlign(
            output_size=roi_output,
            spatial_scale=spatial_scale,
            sampling_ratio=-1,
            aligned=False,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2) if apply_pooling else torch.nn.Identity(),
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Identity()
            if upsample_factor == 1.0
            else torch.nn.Upsample(scale_factor=upsample_factor, mode="bilinear"),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.biase is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, rois):
        x = self.roi_align(x, rois)
        x = self.conv_block(x)
        return x


class GenericPolyMapperRnnBlock(torch.nn.Module):
    def __init__(self, backbone, grid_size=28):
        super().__init__()
        self.backbone = backbone
        self.grid_size = grid_size
        self.roi_align_block1 = RoiAlignBlock(
            in_channels=256,
            out_channels=128,
            spatial_scale=56 / 224,
            roi_output=56,
            kernel_size=(3, 3),
            apply_pooling=True,
        )
        self.roi_align_block2 = RoiAlignBlock(
            in_channels=256,
            out_channels=128,
            spatial_scale=28 / 224,
            roi_output=28,
            kernel_size=(3, 3),
            apply_pooling=False,
        )
        self.roi_align_block3 = RoiAlignBlock(
            in_channels=256,
            out_channels=128,
            spatial_scale=14 / 224,
            roi_output=14,
            kernel_size=(3, 3),
            apply_pooling=False,
            upsample_factor=2.0,
        )
        self.roi_align_block4 = RoiAlignBlock(
            in_channels=256,
            out_channels=128,
            spatial_scale=14 / 224,
            roi_output=7,
            kernel_size=(3, 3),
            apply_pooling=False,
            upsample_factor=4.0,
        )
        self.convlayer5 = make_basic_conv_block(512, 128, 3, 1, 1)
        self.convlstm = ConvLSTM(
            input_size=(grid_size, grid_size),
            input_dim=131,
            hidden_dim=[32, 8],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=True,
        )
        self.lstmlayer = nn.LSTM(
            grid_size * grid_size * 8 + (grid_size * grid_size + 3) * 2,
            grid_size * grid_size * 2,
            batch_first=True,
        )
        self.linear = nn.Linear(grid_size * grid_size * 2, grid_size * grid_size + 3)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of PolygonNet
        :param load_vgg: bool
                    load pretrained vgg model or not
        """

        self._init_convlstm()
        self._init_convlstmlayer()
        self._init_convlayer()
        self._init_roialign()

    def _init_convlayer(self):
        for name, param in self.named_parameters():
            if "bias" in name and "convlayer" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name and "convlayer" in name and "0" in name:
                nn.init.xavier_normal_(param)

    def _init_convlstmlayer(self):
        for name, param in self.lstmlayer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1.0)
            elif "weight" in name:
                # nn.init.xavier_normal_(param)
                nn.init.orthogonal_(param)

    def _init_convlstm(self):
        for name, param in self.convlstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def _init_roialign(self):
        for block in [
            self.roi_align_block1,
            self.roi_align_block2,
            self.roi_align_block3,
            self.roi_align_block4,
        ]:
            block.init_weights()

    def get_backbone_output_features(self, x, rois):
        backbone_output = self.backbone(x)
        roi_output11 = self.roi_align_block1(backbone_output["0"], rois)
        roi_output22 = self.roi_align_block2(backbone_output["1"], rois)
        roi_output33 = self.roi_align_block3(backbone_output["2"], rois)
        roi_output44 = self.roi_align_block4(backbone_output["3"], rois)
        output = torch.cat(
            [roi_output11, roi_output22, roi_output33, roi_output44], dim=1
        )
        output = self.convlayer5(output)
        return output

    def get_rnn_output(self, output, first, second, third):
        bs, length_s = second.shape[0], second.shape[1]
        output = output.unsqueeze(1)
        output = output.repeat(1, length_s, 1, 1, 1)
        padding_f = torch.zeros([bs, 1, 1, self.grid_size, self.grid_size]).to(
            output.device
        )

        input_f = (
            first[:, :-3]
            .view(-1, 1, self.grid_size, self.grid_size)
            .unsqueeze(1)
            .repeat(1, length_s - 1, 1, 1, 1)
        )
        input_f = torch.cat([padding_f, input_f], dim=1)
        input_s = second[:, :, :-3].view(
            -1, length_s, 1, self.grid_size, self.grid_size
        )
        input_t = third[:, :, :-3].view(-1, length_s, 1, self.grid_size, self.grid_size)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)

        output = self.convlstm(output)[0][-1]

        output = output.contiguous().view(bs, length_s, -1)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstmlayer(output)[0]
        output = output.contiguous().view(bs * length_s, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, length_s, -1)

        return output

    def forward(self, x, targets):
        bboxes = [target["boxes"] for target in targets]
        bboxes = torch.stack(
            [
                torch.cat([torch.tensor([idx], device=roi_row.device), roi_row])
                for idx, roi in enumerate(bboxes)
                for roi_row in roi
            ]
        )  # num_rois, 5
        output = self.get_backbone_output_features(x, bboxes)
        return self.get_rnn_output(
            output,
            first=torch.cat([item["x1"] for item in targets]),
            second=torch.cat([item["x2"] for item in targets]),
            third=torch.cat([item["x3"] for item in targets]),
        )

    def test(self, input_data1: torch.Tensor, rois: torch.Tensor, len_s: int):
        bs = rois.shape[0]
        result = torch.zeros([bs, len_s]).to(input_data1.device)
        feature = self.get_backbone_output_features(input_data1, rois)
        if feature.shape[0] == 0:
            return torch.zeros([bs, 1, self.grid_size * self.grid_size + 3])

        padding_f = (
            torch.zeros([bs, 1, 1, self.grid_size, self.grid_size])
            .float()
            .to(input_data1.device)
        )
        input_s = (
            torch.zeros([bs, 1, 1, self.grid_size, self.grid_size])
            .float()
            .to(input_data1.device)
        )
        input_t = (
            torch.zeros([bs, 1, 1, self.grid_size, self.grid_size])
            .float()
            .to(input_data1.device)
        )

        output = torch.cat([feature.unsqueeze(1), padding_f, input_s, input_t], dim=2)

        output, hidden1 = self.convlstm(output)
        output = output[-1]
        output = output.contiguous().view(bs, 1, -1)
        second = torch.zeros([bs, 1, self.grid_size * self.grid_size + 3]).to(
            input_data1.device
        )
        second[:, 0, self.grid_size * self.grid_size + 1] = 1
        third = torch.zeros([bs, 1, self.grid_size * self.grid_size + 3]).to(
            input_data1.device
        )
        third[:, 0, self.grid_size * self.grid_size + 2] = 1
        output = torch.cat([output, second, third], dim=2)

        output, hidden2 = self.lstmlayer(output)
        output = output.contiguous().view(bs, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, 1, -1)
        output = (output == output.max(dim=2, keepdim=True)[0]).float()
        first = output
        result[:, 0] = (output.argmax(2))[:, 0]

        for i in range(len_s - 1):
            second = third
            third = output
            input_f = first[:, :, :-3].view(-1, 1, 1, self.grid_size, self.grid_size)
            input_s = second[:, :, :-3].view(-1, 1, 1, self.grid_size, self.grid_size)
            input_t = third[:, :, :-3].view(-1, 1, 1, self.grid_size, self.grid_size)
            input1 = torch.cat([feature.unsqueeze(1), input_f, input_s, input_t], dim=2)
            output, hidden1 = self.convlstm(input1, hidden1)
            output = output[-1]
            output = output.contiguous().view(bs, 1, -1)
            output = torch.cat([output, second, third], dim=2)
            output, hidden2 = self.lstmlayer(output, hidden2)
            output = output.contiguous().view(bs, -1)
            output = self.linear(output)
            output = output.contiguous().view(bs, 1, -1)
            output = (output == output.max(dim=2, keepdim=True)[0]).float()
            result[:, i + 1] = (output.argmax(2))[:, 0]

        return result

    def compute(self, images: torch.Tensor, targets) -> torch.Tensor:
        output = self.forward(images, targets)
        return output.contiguous().view(-1, self.grid_size * self.grid_size + 3)

    def compute_loss_and_accuracy(
        self, ta: torch.Tensor, result: torch.Tensor
    ) -> Tuple[_Loss, torch.Tensor]:
        target = ta.contiguous().view(-1)
        loss = nn.functional.cross_entropy(result, target)
        result_index = torch.argmax(result, 1)
        correct = (target == result_index).float().sum().item()
        acc = torch.tensor(correct * 1.0 / target.shape[0], device=loss.device)
        return loss, acc

    def get_polygonrnn_losses_and_accuracy(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[Dict[str, Union[_Loss, torch.Tensor]], torch.Tensor]:
        result = self.compute(x, targets)
        loss, acc = self.compute_loss_and_accuracy(
            ta=torch.cat([i["ta"] for i in targets]), result=result
        )
        return {"polygonrnn_loss": loss}, acc


class GenericModPolyMapper(nn.Module):
    def __init__(
        self,
        obj_detection_model: Union[ObjectDetectionModel, torch.nn.Module],
        polygonrnn_model: Optional[Union[GenericPolyMapperRnnBlock, PolygonRNN]] = None,
        grid_size: int = 28,
        val_seq_len: int = 60,
    ):
        super(GenericModPolyMapper, self).__init__()
        self.obj_detection_model = obj_detection_model
        self.backbone = self.obj_detection_model.backbone
        self.polygonrnn_model = (
            GenericPolyMapperRnnBlock(backbone=self.backbone, grid_size=grid_size)
            if polygonrnn_model is None
            else polygonrnn_model
        )
        self.val_seq_len = val_seq_len

    def _get_empty_entries(self, detections):
        empty_entries = []
        for idx, det in enumerate(detections):
            detections[idx].update({"polygonrnn_output": []})
            if det["boxes"].shape[0] == 0:
                empty_entries.append(idx)
        return empty_entries

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        if self.training:
            assert targets is not None, "Targets are required for training"
            losses = self.obj_detection_model(x, targets)
            polygonrnn_loss, acc = self.polygonrnn_model.get_polygonrnn_losses_and_accuracy(  # type: ignore
                x, targets
            )
            losses.update(polygonrnn_loss)
            return losses, acc
        detections = self.obj_detection_model(x)
        empty_entries = self._get_empty_entries(detections)
        if len(empty_entries) == len(detections):
            for idx, _ in enumerate(detections):
                detections[idx].update(
                    {key: [] for key in ["min_row", "min_col", "scale_h", "scale_w"]}
                )
            return detections
        bboxes = [
            item["boxes"]
            for idx, item in enumerate(detections)
            if idx not in empty_entries
        ]
        bboxes = torch.cat(
            [
                torch.column_stack([idx * torch.ones(bbox.shape[0]), bbox])
                for idx, bbox in enumerate(bboxes)
            ]
        )  # num_rois, 5
        polygonrnn_output = self.polygonrnn_model.test(
            x, bboxes, self.val_seq_len
        )  # type: ignore
        min_bound = 0
        for idx, det in enumerate(detections):
            if idx in empty_entries:
                detections[idx].update(
                    {key: [] for key in ["min_row", "min_col", "scale_h", "scale_w"]}
                )
                continue
            n_items = len(det["boxes"])
            detections[idx].update(
                {
                    "polygonrnn_output": polygonrnn_output[
                        min_bound : min_bound + n_items
                    ]
                }
            )
            detections[idx].update(
                polygonrnn_utils.build_polygonrnn_extra_info_from_bboxes(det["boxes"])
            )
            min_bound += n_items
        return detections


class ModPolyMapper(GenericModPolyMapper):
    """
    Modified PolyMapper proposed in the paper "Building outline delineation: From aerial images to polygons with an
    improved end-to-end learning framework" (https://doi.org/10.1016/j.isprsjprs.2021.02.014) .
    The backbone is a ResNet101.
    """

    def __init__(
        self,
        num_classes,
        backbone_trainable_layers: int = 3,
        pretrained: bool = True,
        grid_size: int = 28,
        val_seq_len: Optional[int] = 60,
        **kwargs
    ):
        backbone = resnet_fpn_backbone(
            "resnet101",
            trainable_layers=backbone_trainable_layers,
            pretrained=pretrained,
        )
        model = FasterRCNN(backbone, num_classes, **kwargs)
        super(ModPolyMapper, self).__init__(
            obj_detection_model=model, grid_size=grid_size, val_seq_len=val_seq_len
        )
