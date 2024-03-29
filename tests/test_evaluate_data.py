# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-30
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba -
                                    Cartographic Engineer @ Brazilian Army
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

import os
import json
from typing import List, Tuple
import unittest

from shapely.geometry import Polygon
import shapely.wkt


from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.tools.evaluation import evaluate_data, matching
from tests.utils import load_geometry_list_from_geojson

current_dir = os.path.dirname(__file__)
matching_root_dir = os.path.join(current_dir, "testing_data", "data", "matching_data")


def _load_test_data() -> Tuple[List[Polygon], List[Polygon], List[Polygon]]:
    reference_polygons = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "reference_polygons.geojson")
    )
    candidate_polygons = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "candidate_polygons.geojson")
    )
    expected_reference_matches = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "expected_reference_matches.geojson")
    )
    expected_reference_matches.sort(key=lambda x: x.area)
    return reference_polygons, candidate_polygons, expected_reference_matches


class Test_EvaluateData(unittest.TestCase):
    def test_compute_metrics_on_match_list_dict(self) -> None:
        (
            reference_polygons,
            candidate_polygons,
            expected_reference_matches,
        ) = _load_test_data()
        (
            matched_dict_list,
            unmatched_references_list,
            unmatched_targets_list,
        ) = matching.match_polygon_lists_by_iou(reference_polygons, candidate_polygons)

        def pol_acc(x, y):
            return metrics.polygon_accuracy(x, y, grid_size=300, sequence_length=10000)

        evaluated_data = evaluate_data.compute_metrics_on_match_list_dict(
            matched_dict_list,
            metric_list=[
                metrics.polygon_iou,
                metrics.polis,
                metrics.hausdorff_distance,
                metrics.frechet_distance,
                pol_acc,
                metrics.polygon_mean_max_tangent_angle_errors,
            ],
        )
        evaluated_data.sort(key=lambda x: x["reference"].area)
        self.assertAlmostEqual(evaluated_data[0]["iou"], 0.8051144144000144)
        self.assertAlmostEqual(evaluated_data[0]["intersection"], 54.76825052012902)
        self.assertAlmostEqual(evaluated_data[0]["union"], 68.02542538124013)
        self.assertAlmostEqual(evaluated_data[0]["polis"], 0.5665716069659223)
        self.assertAlmostEqual(
            evaluated_data[0]["hausdorff_distance"], 1.026199498916631
        )
        self.assertAlmostEqual(
            evaluated_data[0]["frechet_distance"], 1.1838713742401072
        )
        self.assertAlmostEqual(evaluated_data[0]["pol_acc"], 0.4283856771354271)
        self.assertAlmostEqual(
            evaluated_data[0]["polygon_mean_max_tangent_angle_errors"],
            0.18611127607744704,
        )

        self.assertAlmostEqual(evaluated_data[1]["iou"], 0.7500752058692223)
        self.assertAlmostEqual(evaluated_data[1]["intersection"], 958.5448025063769)
        self.assertAlmostEqual(evaluated_data[1]["union"], 1277.9315927335183)
        self.assertAlmostEqual(evaluated_data[1]["polis"], 1.758035238733818)
        self.assertAlmostEqual(
            evaluated_data[1]["hausdorff_distance"], 5.0103238459222785
        )
        self.assertAlmostEqual(
            evaluated_data[1]["frechet_distance"], 10.658268896825216
        )
        self.assertAlmostEqual(evaluated_data[1]["pol_acc"], 0.013502700540108022)
        self.assertAlmostEqual(
            evaluated_data[1]["polygon_mean_max_tangent_angle_errors"],
            0.5430984565886399,
        )

    def test_compute_metrics_real_data(self):
        input_list_dict = json.loads(
            '[{"reference": "POLYGON ((10.000 0.000, 10.000 34.000, -0.000 34.000, -0.000 0.000, 10.000 0.000))", "target": "POLYGON ((8.109 23.891, 8.000 0.000, 0.000 0.000, 0.000 34.000, 4.026 33.974, 4.975 33.025, 6.947 33.053, 8.109 23.891))"}, {"reference": "POLYGON ((115.000 33.000, 115.000 0.000, 179.000 0.000, 179.000 33.000, 115.000 33.000))", "target": "POLYGON ((114.000 0.000, 115.052 3.948, 113.020 10.980, 113.036 30.964, 116.950 33.050, 143.036 32.964, 143.936 32.064, 153.065 32.935, 153.934 32.066, 176.053 32.947, 178.999 30.001, 177.953 16.047, 178.972 4.028, 179.000 0.000, 114.000 0.000))"}, {"reference": "POLYGON ((259.000 8.000, 259.000 0.000, 300.000 -0.000, 300.000 35.000, 275.000 35.000, 275.000 33.000, 261.000 33.000, 261.000 8.000, 259.000 8.000))", "target": "POLYGON ((258.000 0.000, 259.013 3.987, 257.990 9.010, 256.043 8.958, 255.120 9.880, 250.946 10.054, 248.014 12.986, 247.912 22.088, 248.128 24.872, 251.181 27.819, 257.797 28.203, 262.940 32.060, 299.000 33.000, 299.000 0.000, 258.000 0.000))"}, {"reference": "POLYGON ((-0.000 42.000, 10.000 42.000, 10.000 61.000, 13.000 61.000, 13.000 75.000, 10.000 75.000, 10.000 88.000, -0.000 88.000, -0.000 42.000))", "target": "POLYGON ((6.994 41.006, 0.000 40.000, 0.000 86.000, 6.993 86.007, 9.051 83.950, 9.057 58.943, 8.025 57.975, 8.013 42.987, 7.026 41.974, 6.994 41.006))"}, {"reference": "POLYGON ((283.000 86.000, 283.000 84.000, 261.000 84.000, 260.000 38.000, 273.000 38.000, 273.000 40.000, 273.000 42.000, 300.000 41.000, 300.000 86.000, 283.000 86.000))", "target": "POLYGON ((299.000 40.000, 286.870 40.129, 286.120 40.880, 276.972 41.028, 274.993 43.007, 275.070 53.930, 274.015 56.984, 272.051 59.949, 262.981 60.019, 261.033 61.967, 261.039 80.961, 264.200 83.800, 271.064 83.936, 271.934 83.066, 299.000 84.000, 299.000 40.000))"}, {"reference": "POLYGON ((-0.000 91.000, 11.000 91.000, 11.000 111.000, 21.000 111.000, 22.000 116.000, 31.000 116.000, 32.000 126.000, 22.000 127.000, 22.000 135.000, -0.000 135.000, -0.000 91.000))", "target": "POLYGON ((3.021 89.979, 0.000 90.000, 0.000 135.000, 19.048 134.952, 20.978 131.022, 21.334 125.332, 29.985 125.015, 31.880 121.120, 30.976 115.024, 27.828 113.172, 21.164 112.503, 17.846 110.154, 9.948 109.052, 8.984 108.016, 8.954 92.046, 3.969 91.031, 3.021 89.979))"}, {"reference": "POLYGON ((283.000 137.000, 283.000 135.000, 276.000 135.000, 276.000 134.000, 264.000 134.000, 263.000 92.000, 300.000 92.000, 300.000 137.000, 283.000 137.000))", "target": "POLYGON ((299.000 91.000, 295.888 91.112, 295.100 91.899, 263.964 92.036, 262.001 93.998, 262.120 130.880, 264.946 133.054, 294.037 133.963, 294.937 133.063, 299.000 133.000, 299.000 91.000))"}, {"reference": "POLYGON ((-0.000 142.000, 13.000 142.000, 13.000 159.000, 20.000 159.000, 20.000 176.000, 13.000 176.000, 13.000 189.000, 0.000 189.000, -0.000 142.000))", "target": "POLYGON ((0.000 188.000, 5.982 189.893, 8.935 189.065, 11.976 184.024, 12.114 178.886, 14.975 175.025, 17.048 174.952, 18.974 172.025, 17.983 163.017, 17.969 161.031, 12.055 157.945, 11.023 156.977, 10.948 150.052, 10.002 143.998, 6.866 142.134, 0.000 142.000, 0.000 188.000))"}, {"reference": "POLYGON ((0.000 194.000, 12.000 194.000, 13.000 239.000, 0.000 239.000, 0.000 194.000))", "target": "POLYGON ((5.982 189.893, 4.057 192.943, 0.000 193.000, 0.000 239.000, 9.020 238.980, 11.063 236.937, 11.021 195.979, 10.031 194.969, 9.993 193.007, 5.982 189.893))"}, {"reference": "POLYGON ((219.000 289.000, 219.000 267.000, 233.000 267.000, 233.000 263.000, 219.000 263.000, 219.000 244.000, 300.000 245.000, 300.000 289.000, 219.000 289.000))", "target": "POLYGON ((219.872 259.128, 222.397 263.853, 220.992 267.008, 220.948 286.051, 223.104 287.896, 299.000 288.000, 299.000 243.000, 280.973 242.027, 280.024 242.976, 221.986 241.014, 219.976 243.024, 219.031 247.969, 220.021 257.979, 219.872 259.128))"}, {"reference": "POLYGON ((194.000 274.000, 209.000 274.000, 209.000 284.000, 194.000 284.000, 194.000 274.000))", "target": "POLYGON ((194.966 275.034, 193.972 282.028, 198.931 285.069, 205.023 284.978, 205.977 284.023, 208.045 283.955, 209.999 282.001, 210.029 275.971, 194.966 275.034))"}, {"reference": "POLYGON ((0.000 244.000, 13.000 244.000, 13.000 289.000, 0.000 289.000, 0.000 244.000))", "target": "POLYGON ((4.996 288.004, 7.016 288.234, 10.041 287.959, 12.000 286.001, 11.930 262.070, 12.088 260.912, 11.926 260.074, 11.926 247.074, 9.793 245.207, 3.012 243.988, 0.000 244.000, 0.000 289.000, 4.014 288.986, 4.996 288.004))"}, {"reference": "POLYGON ((284.000 189.000, 284.000 186.000, 261.000 186.000, 261.000 142.000, 271.000 142.000, 271.000 143.000, 278.000 143.000, 278.000 142.000, 297.000 142.000, 297.000 143.000, 299.000 143.000, 299.000 142.000, 300.000 142.000, 300.000 189.000, 284.000 189.000))", "target": "POLYGON ((299.000 141.000, 285.907 141.093, 285.090 141.910, 263.986 142.013, 262.025 143.975, 262.052 183.948, 264.959 186.042, 294.058 186.941, 294.932 186.068, 299.000 186.000, 299.000 141.000))"}, {"reference": "POLYGON ((0.000 294.000, 8.000 294.000, 8.000 297.000, 11.000 297.000, 11.000 294.000, 13.000 294.000, 13.000 300.000, 0.000 300.000, 0.000 294.000))", "target": "POLYGON ((6.665 292.669, 5.001 293.997, 0.000 293.000, 0.000 299.000, 12.000 299.000, 10.925 295.075, 6.665 292.669))"}, {"reference": "POLYGON ((193.000 300.000, 193.000 292.000, 209.000 292.000, 209.000 300.000, 193.000 300.000))", "target": "POLYGON ((197.925 292.825, 194.984 293.017, 192.056 295.944, 192.000 299.000, 210.000 299.000, 208.976 295.024, 208.970 294.030, 197.925 292.825))"}, {"reference": "POLYGON ((278.000 241.000, 278.000 216.000, 257.000 216.000, 257.000 196.000, 278.000 196.000, 300.000 195.000, 300.000 241.000, 278.000 241.000))", "target": "POLYGON ((299.000 192.000, 295.868 192.132, 295.108 192.891, 258.999 193.001, 256.014 195.986, 257.029 202.972, 256.879 204.121, 257.055 211.945, 259.208 213.792, 274.010 213.989, 277.985 216.015, 277.879 217.121, 278.057 217.943, 277.930 223.070, 277.060 223.941, 278.049 233.951, 280.172 235.827, 299.000 236.000, 299.000 192.000))"}, {"reference": "POLYGON ((279.000 300.000, 279.000 296.000, 300.000 296.000, 300.000 300.000, 279.000 300.000))", "target": "POLYGON ((299.000 295.000, 284.056 294.944, 278.993 298.007, 279.000 299.000, 299.000 299.000, 299.000 295.000))"}, {"reference": "POLYGON ((121.000 300.000, 121.000 293.000, 154.000 293.000, 154.000 295.000, 165.000 295.000, 165.000 293.000, 172.000 293.000, 172.000 300.000, 121.000 300.000))", "target": "POLYGON ((122.967 295.033, 122.000 299.000, 168.000 299.000, 164.857 294.143, 141.009 292.990, 139.958 294.041, 124.999 294.001, 124.088 294.912, 122.967 295.033))"}, {"reference": "POLYGON ((118.000 84.000, 118.000 39.000, 146.000 38.000, 146.000 40.000, 181.000 40.000, 182.000 70.000, 174.000 70.000, 174.000 84.000, 118.000 84.000))", "target": "POLYGON ((180.022 41.978, 175.969 40.031, 119.942 39.058, 116.984 42.016, 117.966 73.034, 117.898 80.102, 117.901 81.099, 117.932 82.068, 122.191 84.809, 126.064 84.936, 126.965 84.035, 136.034 83.966, 136.938 83.062, 171.010 83.990, 172.992 81.008, 173.068 71.932, 175.934 69.066, 178.010 68.990, 180.987 66.013, 179.999 61.002, 180.022 41.978))"}, {"reference": "POLYGON ((216.000 84.000, 216.000 58.000, 237.000 58.000, 237.000 84.000, 216.000 84.000))", "target": "POLYGON ((231.966 66.034, 220.924 65.077, 217.030 68.969, 218.014 81.986, 217.964 83.036, 221.142 84.858, 234.061 84.939, 237.008 81.992, 237.126 68.875, 237.008 67.992, 237.032 66.969, 233.002 64.998, 231.966 66.034))"}, {"reference": "POLYGON ((167.000 90.000, 167.000 115.000, 154.000 115.000, 154.000 134.000, 119.000 134.000, 119.000 90.000, 167.000 90.000))", "target": "POLYGON ((166.027 92.973, 162.091 90.909, 119.961 91.039, 117.963 94.037, 118.919 131.082, 118.955 132.045, 123.025 133.975, 150.993 134.007, 153.039 131.961, 153.066 117.934, 155.969 115.032, 163.999 115.001, 166.001 112.999, 165.921 96.079, 166.027 92.973))"}, {"reference": "POLYGON ((230.000 89.000, 243.000 89.000, 243.000 102.000, 230.000 102.000, 230.000 113.000, 218.000 113.000, 218.000 91.000, 230.000 91.000, 230.000 89.000))", "target": "POLYGON ((241.963 94.037, 242.054 91.946, 239.100 90.900, 230.013 91.987, 218.962 92.038, 218.039 110.961, 220.928 113.072, 226.023 112.977, 226.976 112.024, 228.997 112.003, 229.999 111.001, 229.024 102.976, 230.952 102.048, 231.912 101.088, 240.915 101.085, 242.841 97.159, 243.152 95.848, 241.963 94.037))"}, {"reference": "POLYGON ((120.000 185.000, 120.000 141.000, 155.000 141.000, 155.000 147.000, 167.000 147.000, 167.000 167.000, 155.000 167.000, 155.000 185.000, 120.000 185.000))", "target": "POLYGON ((153.996 142.004, 150.908 140.092, 122.996 140.004, 122.092 140.908, 120.971 141.029, 120.032 181.968, 125.182 185.817, 129.069 185.931, 129.938 185.062, 148.012 184.988, 148.973 184.027, 153.035 183.965, 155.048 181.952, 155.964 166.036, 164.972 166.028, 166.947 163.053, 167.082 157.918, 166.933 157.067, 167.075 152.925, 164.810 151.190, 155.084 149.916, 153.990 143.010, 153.996 142.004))"}, {"reference": "POLYGON ((217.000 189.000, 218.000 162.000, 242.000 162.000, 241.000 189.000, 217.000 189.000))", "target": "POLYGON ((241.024 163.976, 237.994 162.006, 220.990 162.010, 220.084 162.916, 218.972 163.028, 218.034 186.966, 220.098 188.902, 229.021 188.979, 229.942 188.058, 239.018 188.982, 241.043 186.957, 241.024 163.976))"}, {"reference": "POLYGON ((120.000 236.000, 121.000 192.000, 164.000 192.000, 163.000 217.000, 155.000 217.000, 155.000 236.000, 120.000 236.000))", "target": "POLYGON ((162.983 194.017, 157.098 191.902, 121.958 192.042, 119.940 195.060, 120.015 233.985, 154.028 234.972, 155.052 218.948, 156.948 216.052, 161.027 215.973, 163.008 213.992, 162.983 194.017))"}, {"reference": "POLYGON ((121.000 287.000, 121.000 242.000, 155.000 242.000, 155.000 244.000, 171.000 244.000, 171.000 280.000, 157.000 280.000, 157.000 287.000, 121.000 287.000))", "target": "POLYGON ((171.131 245.869, 167.060 242.940, 122.021 241.979, 120.923 247.078, 120.071 247.929, 122.018 284.982, 124.959 287.041, 155.012 286.988, 158.010 279.990, 168.016 279.984, 170.976 276.024, 170.941 247.059, 171.131 245.869))"}]'
        )
        expected_results = json.loads(
            '[{"polis": 0.7595855084366284, "iou": 0.7810529411764706, "intersection": 265.558, "union": 340.0, "polygon_mean_max_tangent_angle_errors": 2.1073424255447017e-08}, {"polis": 0.5709674144251844, "iou": 0.9605429278930533, "intersection": 2076.192207440712, "union": 2161.477792559288, "polygon_mean_max_tangent_angle_errors": 0.28130220215151686}, {"polis": 3.035214246728734, "iou": 0.7840629244233167, "intersection": 1249.8161939950799, "union": 1594.0253710049199, "polygon_mean_max_tangent_angle_errors": 0.9273263289744461}, {"polis": 1.4674991570362002, "iou": 0.7391038417691702, "intersection": 378.78086467522206, "union": 512.486667324778, "polygon_mean_max_tangent_angle_errors": 0.14285774600664053}, {"polis": 3.2392860148971443, "iou": 0.753951665199378, "intersection": 1341.1254510954418, "union": 1778.7949984045586, "polygon_mean_max_tangent_angle_errors": 0.7490442699432741}, {"polis": 1.1069276233704897, "iou": 0.8753381273786028, "intersection": 775.0150221957564, "union": 885.3893118042434, "polygon_mean_max_tangent_angle_errors": 0.8046625962571263}, {"polis": 1.4114570476609176, "iou": 0.8912191838935114, "intersection": 1470.420577032907, "union": 1649.8978069670932, "polygon_mean_max_tangent_angle_errors": 0.8912614734723082}, {"polis": 1.3588749027694953, "iou": 0.8617758480272056, "intersection": 632.5752363480287, "union": 734.0368586519714, "polygon_mean_max_tangent_angle_errors": 0.5924719327040762}, {"polis": 1.034552901921241, "iou": 0.8476059335136593, "intersection": 493.2054266055046, "union": 581.8805733944954, "polygon_mean_max_tangent_angle_errors": 1.007788516866259}, {"polis": 2.4518876342169507, "iou": 0.8889784655178544, "intersection": 3361.6473021025286, "union": 3781.4721418974714, "polygon_mean_max_tangent_angle_errors": 0.7977432152416802}, {"polis": 0.8545786798527895, "iou": 0.7391016751198747, "intersection": 121.435730248116, "union": 164.30179275188408, "polygon_mean_max_tangent_angle_errors": 0.7859219978178301}, {"polis": 0.9122105117598687, "iou": 0.8895062659156011, "intersection": 520.3775969815422, "union": 585.0184725184579, "polygon_mean_max_tangent_angle_errors": 0.012025128658265506}, {"polis": 1.0162207780191217, "iou": 0.9085119065677119, "intersection": 1620.8967791695393, "union": 1784.1227698304613, "polygon_mean_max_tangent_angle_errors": 0.7853981633974483}, {"polis": 0.7247574653338779, "iou": 0.5827251778143517, "intersection": 47.82558139534885, "union": 82.0722756046512, "polygon_mean_max_tangent_angle_errors": 0.673567233842361}, {"polis": 0.9168189930710355, "iou": 0.670758872812911, "intersection": 89.4505614248636, "union": 133.35725407513658, "polygon_mean_max_tangent_angle_errors": 0.7855689575888003}, {"polis": 1.7224068243770305, "iou": 0.7896575364878612, "intersection": 1223.6192305587008, "union": 1549.556831941299, "polygon_mean_max_tangent_angle_errors": 0.7853981633974484}, {"polis": 0.7680296277064764, "iou": 0.5652240897644931, "intersection": 56.68494307668528, "union": 100.28755692331458, "polygon_mean_max_tangent_angle_errors": 0.5440712345600704}, {"polis": 1.4751738669150365, "iou": 0.6213358281612769, "intersection": 215.76618138633364, "union": 347.2617731136669, "polygon_mean_max_tangent_angle_errors": 0.6943402733291576}, {"polis": 1.054487845331343, "iou": 0.9493964785960716, "intersection": 2620.5876469169475, "union": 2760.2668705830515, "polygon_mean_max_tangent_angle_errors": 0.7853981633974532}, {"polis": 2.7933753556011034, "iou": 0.6251434748878106, "intersection": 349.8611373090645, "union": 559.6493466909355, "polygon_mean_max_tangent_angle_errors": 0.7853981633974484}, {"polis": 1.1184623798083928, "iou": 0.9344868302785612, "intersection": 1764.9941087604204, "union": 1888.7308537395795, "polygon_mean_max_tangent_angle_errors": 0.7852258981106025}, {"polis": 1.1012555470578975, "iou": 0.8079096663792042, "intersection": 350.20768514423906, "union": 433.4738148557613, "polygon_mean_max_tangent_angle_errors": 0.1264104683330507}, {"polis": 1.1761195567299938, "iou": 0.9225051759867194, "intersection": 1678.8644402843518, "union": 1819.8970412156484, "polygon_mean_max_tangent_angle_errors": 0.7853981633974484}, {"polis": 0.9255714303055598, "iou": 0.9174984567901231, "intersection": 594.539, "union": 648.0000000000002, "polygon_mean_max_tangent_angle_errors": 2.1073424255447017e-08}, {"polis": 0.9324176337435838, "iou": 0.9367643530362703, "intersection": 1642.8212294767636, "union": 1753.7187705232373, "polygon_mean_max_tangent_angle_errors": 0.5796694631074284}, {"polis": 0.7877513599186978, "iou": 0.9683731426629166, "intersection": 2077.643321798474, "union": 2145.498703201526, "polygon_mean_max_tangent_angle_errors": 0.7859853623904435}]'
        )
        matched_list_dict = [
            {k: shapely.wkt.loads(v) for k, v in i.items()} for i in input_list_dict
        ]
        evaluated_data = evaluate_data.compute_metrics_on_match_list_dict(
            matched_list_dict,
            metric_list=[
                metrics.polis,
                metrics.polygon_iou,
                metrics.polygon_mean_max_tangent_angle_errors,
            ],
        )
        for idx, item in enumerate(evaluated_data):
            for key, value in item.items():
                if key in ["reference", "target"]:
                    continue
                self.assertAlmostEquals(value, expected_results[idx][key])
