#!/usr/bin/env python3
import copy
import math
from typing import Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener


def rpy_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class ScanStabilizer(Node):
    def __init__(self) -> None:
        super().__init__("mapless_scan_stabilizer")

        self.declare_parameter("input_scan_topic", "/scan")
        self.declare_parameter("output_scan_topic", "/scan_stable")
        self.declare_parameter("tilt_status_topic", "/scan_tilt_exceeded")
        self.declare_parameter("global_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")

        self.declare_parameter("max_roll_deg", 7.0)
        self.declare_parameter("max_pitch_deg", 7.0)
        self.declare_parameter("hard_stop_deg", 12.0)
        self.declare_parameter("hysteresis_deg", 1.0)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("warn_interval_sec", 2.0)

        self.input_scan_topic = str(self.get_parameter("input_scan_topic").value)
        self.output_scan_topic = str(self.get_parameter("output_scan_topic").value)
        self.tilt_status_topic = str(self.get_parameter("tilt_status_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.max_roll = math.radians(float(self.get_parameter("max_roll_deg").value))
        self.max_pitch = math.radians(float(self.get_parameter("max_pitch_deg").value))
        self.hard_stop = math.radians(float(self.get_parameter("hard_stop_deg").value))
        self.hysteresis = math.radians(float(self.get_parameter("hysteresis_deg").value))
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.warn_interval_sec = float(self.get_parameter("warn_interval_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(LaserScan, self.input_scan_topic, self.scan_callback, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.output_scan_topic, 10)
        self.tilt_status_pub = self.create_publisher(Bool, self.tilt_status_topic, 10)

        self.tilt_blocked = False
        self.last_warn_sec = 0.0

        self.get_logger().info(
            "Scan stabilizer enabled: in=%s out=%s max_roll=%.1fdeg max_pitch=%.1fdeg"
            % (
                self.input_scan_topic,
                self.output_scan_topic,
                math.degrees(self.max_roll),
                math.degrees(self.max_pitch),
            )
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def scan_callback(self, msg: LaserScan) -> None:
        tilt = self.lookup_tilt()
        if tilt is None:
            # If TF is temporarily unavailable, do not drop scan data.
            self.scan_pub.publish(msg)
            self.tilt_status_pub.publish(Bool(data=False))
            return

        roll, pitch = tilt
        abs_roll = abs(roll)
        abs_pitch = abs(pitch)
        max_tilt = max(abs_roll, abs_pitch)
        soft_limit = max(self.max_roll, self.max_pitch)

        if self.tilt_blocked:
            # Hysteresis to avoid rapid toggling near threshold.
            if abs_roll <= max(0.0, self.max_roll - self.hysteresis) and abs_pitch <= max(0.0, self.max_pitch - self.hysteresis):
                self.tilt_blocked = False
        else:
            if abs_roll > self.max_roll or abs_pitch > self.max_pitch:
                self.tilt_blocked = True

        if max_tilt >= self.hard_stop:
            self.tilt_blocked = True

        if self.tilt_blocked:
            out = copy.copy(msg)
            out.ranges = [float("inf")] * len(msg.ranges)
            if msg.intensities:
                out.intensities = [0.0] * len(msg.intensities)
            self.scan_pub.publish(out)
            self.tilt_status_pub.publish(Bool(data=True))
            self.maybe_warn(roll, pitch)
            return

        self.scan_pub.publish(msg)
        self.tilt_status_pub.publish(Bool(data=False))

    def maybe_warn(self, roll: float, pitch: float) -> None:
        now_sec = self.now_sec()
        if now_sec - self.last_warn_sec < self.warn_interval_sec:
            return
        self.last_warn_sec = now_sec
        self.get_logger().warn(
            "Tilt exceeded, suppressing scan: roll=%.1fdeg pitch=%.1fdeg"
            % (math.degrees(roll), math.degrees(pitch))
        )

    def lookup_tilt(self) -> Optional[Tuple[float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException:
            return None

        q = tf_msg.transform.rotation
        roll, pitch, _ = rpy_from_quaternion(q.x, q.y, q.z, q.w)
        return (roll, pitch)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanStabilizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

