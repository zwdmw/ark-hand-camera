import unittest

import numpy as np

try:
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import PointField
except ImportError:
    Time = None

from ros_rgb_cloud import cloud_message


@unittest.skipIf(Time is None, "Source ROS 2 setup to test PointCloud2")
class RGBMessageTests(unittest.TestCase):
    def test_wire_layout_and_primary_colors(self):
        xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
        stamp = Time(sec=123, nanosec=456)
        msg = cloud_message(xyz, rgb, "camera_optical", stamp)
        self.assertEqual(msg.header.frame_id, "camera_optical")
        self.assertEqual(msg.header.stamp, stamp)
        self.assertEqual((msg.height, msg.width, msg.point_step, msg.row_step), (1, 3, 16, 48))
        self.assertEqual([(f.name, f.offset, f.datatype) for f in msg.fields],
                         [(n, i*4, PointField.FLOAT32) for i, n in enumerate(("x", "y", "z", "rgb"))])
        data = bytes(msg.data)
        values = np.frombuffer(data, "<f4").reshape(-1, 4)
        np.testing.assert_array_equal(values[:, :3], xyz)
        packed = np.frombuffer(data, "<u4").reshape(-1, 4)[:, 3]
        np.testing.assert_array_equal(packed, [0x00FF0000, 0x0000FF00, 0x000000FF])
        self.assertTrue(msg.is_dense)
        self.assertFalse(msg.is_bigendian)

    def test_marks_nonfinite_points_and_empty_cloud(self):
        msg = cloud_message(np.array([[np.nan, 0, 1]]), np.zeros((1, 3), np.uint8), "camera", Time())
        self.assertFalse(msg.is_dense)
        empty = cloud_message(np.empty((0, 3)), np.empty((0, 3), np.uint8), "camera", Time())
        self.assertEqual(empty.width, 0)
        self.assertEqual(len(empty.data), 0)


if __name__ == "__main__":
    unittest.main()
