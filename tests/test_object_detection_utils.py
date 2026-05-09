# -*- coding: utf-8 -*-
import torch
import unittest
from pytorch_segmentation_models_trainer.utils.object_detection_utils import (
    evaluate_box_iou,
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
)


class TestObjectDetectionUtils(unittest.TestCase):
    def test_evaluate_box_iou(self):
        target = {
            "boxes": torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=torch.float)
        }
        # Perfect match
        pred = {
            "boxes": torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=torch.float)
        }
        iou = evaluate_box_iou(target, pred)
        self.assertAlmostEqual(iou.item(), 1.0)

        # No boxes in pred
        pred_empty = {"boxes": torch.zeros((0, 4), dtype=torch.float)}
        iou_empty = evaluate_box_iou(target, pred_empty)
        self.assertEqual(iou_empty.item(), 0.0)

    def test_bbox_conversions(self):
        xywh = [10, 20, 30, 40]
        xyxy = bbox_xywh_to_xyxy(xywh)
        self.assertEqual(xyxy, [10, 20, 40, 60])

        xywh_back = bbox_xyxy_to_xywh(xyxy)
        self.assertEqual(xywh_back, xywh)


if __name__ == "__main__":
    unittest.main()
