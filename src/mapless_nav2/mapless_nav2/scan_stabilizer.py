#!/usr/bin/env python3
import copy
import math
from typing import Optional, Tuple

from nav_msgs.msg import Odometry
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
        self.declare_parameter("odom_topic", "/odom")

        self.declare_parameter("max_roll_deg", 7.0)
        self.declare_parameter("max_pitch_deg", 7.0)
        self.declare_parameter("hard_stop_deg", 12.0)
        self.declare_parameter("hysteresis_deg", 1.0)
        self.declare_parameter("max_yaw_rate_deg_s", 25.0)
        self.declare_parameter("max_yaw_delta_per_scan_deg", 3.0)
        self.declare_parameter("drop_scan_on_fast_turn", True)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("warn_interval_sec", 2.0)

        self.input_scan_topic = str(self.get_parameter("input_scan_topic").value)
        self.output_scan_topic = str(self.get_parameter("output_scan_topic").value)
        self.tilt_status_topic = str(self.get_parameter("tilt_status_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.max_roll = math.radians(float(self.get_parameter("max_roll_deg").value))
        self.max_pitch = math.radians(float(self.get_parameter("max_pitch_deg").value))
        self.hard_stop = math.radians(float(self.get_parameter("hard_stop_deg").value))
        self.hysteresis = math.radians(float(self.get_parameter("hysteresis_deg").value))
        self.max_yaw_rate = math.radians(float(self.get_parameter("max_yaw_rate_deg_s").value))
        self.max_yaw_delta_per_scan = math.radians(float(self.get_parameter("max_yaw_delta_per_scan_deg").value))
        self.drop_scan_on_fast_turn = bool(self.get_parameter("drop_scan_on_fast_turn").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.warn_interval_sec = float(self.get_parameter("warn_interval_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(LaserScan, self.input_scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.output_scan_topic, 10)
        self.tilt_status_pub = self.create_publisher(Bool, self.tilt_status_topic, 10)

        self.tilt_blocked = False
        self.last_warn_sec = 0.0
        self.current_yaw_rate = 0.0
        self.last_stable_scan: Optional[LaserScan] = None
        self.last_scan_stamp_sec: Optional[float] = None

        self.get_logger().info(
            "Scan stabilizer enabled: in=%s out=%s max_roll=%.1fdeg max_pitch=%.1fdeg max_yaw_rate=%.1fdeg/s max_yaw_delta=%.1fdeg"
            % (
                self.input_scan_topic,
                self.output_scan_topic,
                math.degrees(self.max_roll),
                math.degrees(self.max_pitch),
                math.degrees(self.max_yaw_rate),
                math.degrees(self.max_yaw_delta_per_scan),
            )
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg: Odometry) -> None:
        self.current_yaw_rate = float(msg.twist.twist.angular.z)

    def scan_callback(self, msg: LaserScan) -> None:
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        sensor_frame = msg.header.frame_id.strip() if msg.header.frame_id else self.base_frame
        tilt = self.lookup_tilt(sensor_frame)
        if tilt is None:
            # If TF is temporarily unavailable, do not drop scan data.
            self.scan_pub.publish(msg)
            self.tilt_status_pub.publish(Bool(data=False))
            self.last_scan_stamp_sec = stamp_sec
            return

        roll, pitch = tilt
        abs_roll = abs(roll)
        abs_pitch = abs(pitch)
        max_tilt = max(abs_roll, abs_pitch)
        yaw_rate = abs(self.current_yaw_rate)
        scan_duration = self.estimate_scan_duration(msg)
        yaw_delta = yaw_rate * scan_duration
        fast_turn = self.drop_scan_on_fast_turn and (
            (scan_duration > 1.0e-3 and yaw_delta > self.max_yaw_delta_per_scan)
            or yaw_rate > (1.8 * self.max_yaw_rate)
        )

        if self.tilt_blocked:
            # Hysteresis to avoid rapid toggling near threshold.
            if (
                abs_roll <= max(0.0, self.max_roll - self.hysteresis)
                and abs_pitch <= max(0.0, self.max_pitch - self.hysteresis)
                and not fast_turn
            ):
                self.tilt_blocked = False
        else:
            if abs_roll > self.max_roll or abs_pitch > self.max_pitch or fast_turn:
                self.tilt_blocked = True

        if max_tilt >= self.hard_stop:
            self.tilt_blocked = True

        if self.tilt_blocked:
            self.tilt_status_pub.publish(Bool(data=True))
            self.maybe_warn(roll, pitch, yaw_rate, scan_duration, fast_turn)
            self.last_scan_stamp_sec = stamp_sec
            return

        self.scan_pub.publish(msg)
        self.tilt_status_pub.publish(Bool(data=False))
        self.last_stable_scan = copy.deepcopy(msg)
        self.last_scan_stamp_sec = stamp_sec

    def make_suppressed_scan(self, msg: LaserScan) -> LaserScan:
        if self.last_stable_scan is not None:
            out = copy.deepcopy(self.last_stable_scan)
            out.header = msg.header
            out.scan_time = msg.scan_time
            out.time_increment = msg.time_increment
            return out

        out = copy.copy(msg)
        out.ranges = [float("inf")] * len(msg.ranges)
        if msg.intensities:
            out.intensities = [0.0] * len(msg.intensities)
        return out

    def estimate_scan_duration(self, msg: LaserScan) -> float:
        if msg.scan_time > 1e-6:
            return float(msg.scan_time)
        if msg.time_increment > 1e-9 and msg.ranges:
            return float(msg.time_increment) * float(len(msg.ranges))
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self.last_scan_stamp_sec is not None:
            dt = stamp_sec - self.last_scan_stamp_sec
            if 1.0e-3 < dt < 0.5:
                return dt
        return 0.0

    def maybe_warn(
        self,
        roll: float,
        pitch: float,
        yaw_rate: float,
        scan_duration: float,
        fast_turn: bool,
    ) -> None:
        now_sec = self.now_sec()
        if now_sec - self.last_warn_sec < self.warn_interval_sec:
            return
        self.last_warn_sec = now_sec
        if fast_turn:
            self.get_logger().warn(
                "Suppressing scan on fast turn: yaw_rate=%.1fdeg/s yaw_delta=%.1fdeg roll=%.1fdeg pitch=%.1fdeg"
                % (
                    math.degrees(yaw_rate),
                    math.degrees(yaw_rate * scan_duration),
                    math.degrees(roll),
                    math.degrees(pitch),
                )
            )
            return
        self.get_logger().warn(
            "Tilt exceeded, suppressing scan: roll=%.1fdeg pitch=%.1fdeg"
            % (math.degrees(roll), math.degrees(pitch))
        )

    def lookup_tilt(self, sensor_frame: str) -> Optional[Tuple[float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                sensor_frame,
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
