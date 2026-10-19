import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rgbd_pointcloud import load_intrinsics, parse_args, pointcloud, prepare_image, write_ply


class ArgumentTests(unittest.TestCase):
    def test_camera_input_requires_serial(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                parse_args([])

    def test_camera_configuration_from_environment(self):
        environment = {
            "HIKROBOT_BASE_URL": "http://camera-service:8770",
            "HIKROBOT_SERIAL": "CAMERA_SERIAL",
            "YOLO_DEPTH_MODEL": "/models/depth.pt",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            args = parse_args([])
        self.assertEqual(args.base_url, environment["HIKROBOT_BASE_URL"])
        self.assertEqual(args.serial, environment["HIKROBOT_SERIAL"])
        self.assertEqual(args.model, Path(environment["YOLO_DEPTH_MODEL"]))


class GeometryTests(unittest.TestCase):
    def test_metric_projection_color_and_invalid_depth(self):
        bgr = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], np.uint8)
        depth = np.array([[2, 4], [np.nan, 0]], np.float32)
        k = np.array([[2, 0, .5], [0, 4, .5], [0, 0, 1.]])
        xyz, rgb = pointcloud(bgr, depth, k, stride=1)
        np.testing.assert_allclose(xyz, [[-.5, -.25, 2], [1, -.5, 4]])
        np.testing.assert_array_equal(rgb, [[30, 20, 10], [60, 50, 40]])

    def test_roi_resize_and_reverse_keep_sensor_rays(self):
        full_k = np.array([[100, 0, 49.5], [0, 100, 39.5], [0, 0, 1.]])
        calibration = dict(matrix=full_k, width=100, height=80, distortion=np.zeros(5), calibrated=True)
        image = np.zeros((20, 30, 3), np.uint8)
        image[:, -1] = [1, 2, 3]
        meta = dict(width=60, height=40, offset_x=20, offset_y=10, reverse_x=True)
        prepared, k, mask, _ = prepare_image(image, meta, calibration, max_width=30)
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(prepared[0, 0], [1, 2, 3])
        np.testing.assert_allclose(k, [[50, 0, 14.5], [0, 50, 14.5], [0, 0, 1.]])
        # Preview pixel 0 maps to full-sensor coordinate (20.5, 10.5).
        xyz, _ = pointcloud(prepared, np.full((20, 30), 2, np.float32), k, stride=1)
        np.testing.assert_allclose(xyz[0], [-.58, -.58, 2], atol=1e-7)

    def test_depth_clipping_and_rectification_mask(self):
        depth = np.array([[.01, .2, 2, np.inf, -1]], np.float32)
        mask = np.array([[True, False, True, True, True]])
        xyz, _ = pointcloud(np.zeros((1, 5, 3), np.uint8), depth, np.eye(3), mask, 1, .05, 3)
        np.testing.assert_array_equal(xyz, [[4, 0, 2]])

    def test_ply_binary_layout_and_rgb_channels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.ply"
            write_ply(path, np.array([[1, 2, 3]], np.float32), np.array([[13, 27, 241]], np.uint8))
            head, data = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1", head)
            self.assertEqual(len(data), 15)
            np.testing.assert_array_equal(np.frombuffer(data[:12], "<f4"), [1, 2, 3])
            self.assertEqual(data[12:], bytes([13, 27, 241]))

    def test_reject_planar_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"planar.xml"
            path.write_text('<CalibInfo><CalibInputParam><CalibParam ParamName="CalibType"><ParamValue>NPointCalib</ParamValue></CalibParam></CalibInputParam></CalibInfo>')
            with self.assertRaisesRegex(ValueError, "not camera intrinsics"):
                load_intrinsics(path)

    def test_calibration_json_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"k.json"
            data = dict(width=100, height=80, fx=90, fy=91, cx=49, cy=39)
            path.write_text(json.dumps(data))
            k = load_intrinsics(path)
            self.assertTrue(k["calibrated"])
            self.assertEqual(k["matrix"][1, 1], 91)
            data["fx"] = -1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_intrinsics(path)


if __name__ == "__main__":
    unittest.main()
