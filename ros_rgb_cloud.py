"""ROS 2 XYZRGB publisher, imported only when --ros2 is enabled."""
from __future__ import annotations

import numpy as np


def cloud_message(xyz, rgb, frame_id, stamp):
    """Pack PCL-compatible float32 rgb bits as 0x00RRGGBB, preserving XYZ."""
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("Expected matching N x 3 XYZ and RGB arrays")
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                                        ("z", "<f4"), ("rgb", "<u4")])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    colors = rgb.astype(np.uint32)
    vertices["rgb"] = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return PointCloud2(
        header=Header(frame_id=frame_id, stamp=stamp), height=1, width=len(xyz),
        fields=[PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                for name, offset in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))],
        is_bigendian=False, point_step=16, row_step=16*len(xyz),
        data=vertices.tobytes(), is_dense=bool(np.isfinite(xyz).all()))


class RGBCloudPublisher:
    def __init__(self, topic, frame_id, reference_frame):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import PointCloud2
        self.rclpy = rclpy
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=[])
        self.node = rclpy.create_node("hikrobot_rgb_pointcloud")
        self.frame_id = frame_id
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.node.create_publisher(PointCloud2, topic, qos)
        self.tf = None
        if reference_frame != frame_id:
            from geometry_msgs.msg import TransformStamped
            from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
            self.tf = StaticTransformBroadcaster(self.node)
            transform = TransformStamped()
            transform.header.stamp = self.node.get_clock().now().to_msg()
            transform.header.frame_id = reference_frame
            transform.child_frame_id = frame_id
            # Camera link: X forward Y left Z up. Optical: X right Y down Z forward.
            transform.transform.rotation.x = -.5
            transform.transform.rotation.y = .5
            transform.transform.rotation.z = -.5
            transform.transform.rotation.w = .5
            self.tf.sendTransform(transform)
        self.node.get_logger().info(f"Publishing RGB PointCloud2: {topic}; frame={frame_id}")

    def capture_stamp(self, age=0.):
        from rclpy.time import Time
        now = self.node.get_clock().now()
        return Time(nanoseconds=max(0, now.nanoseconds-round(age*1e9)), clock_type=now.clock_type).to_msg()

    def ok(self):
        return self.rclpy.ok()

    def publish(self, xyz, rgb, stamp):
        self.publisher.publish(cloud_message(xyz, rgb, self.frame_id, stamp))
        self.rclpy.spin_once(self.node, timeout_sec=0.)

    def close(self):
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()
