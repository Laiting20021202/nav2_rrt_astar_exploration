#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class SafetyController(Node):
    def __init__(self) -> None:
        super().__init__("mapless_safety_controller")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("input_cmd_topic", "/cmd_vel")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_safe")
        self.declare_parameter("status_topic", "/mapless_safety_active")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("front_arc_deg", 30.0)
        self.declare_parameter("stop_distance", 0.36)
        self.declare_parameter("danger_distance", 0.30)
        self.declare_parameter("slow_distance", 0.65)
        self.declare_parameter("clear_resume_distance", 0.48)
        self.declare_parameter("cmd_timeout_sec", 0.6)
        self.declare_parameter("max_forward_speed", 0.09)
        self.declare_parameter("max_turn_speed", 0.9)
        self.declare_parameter("turn_assist_speed", 0.55)
        self.declare_parameter("turn_assist_deadband", 0.10)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.front_arc = math.radians(float(self.get_parameter("front_arc_deg").value))
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.danger_distance = float(self.get_parameter("danger_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)
        self.clear_resume_distance = float(self.get_parameter("clear_resume_distance").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)
        self.turn_assist_speed = float(self.get_parameter("turn_assist_speed").value)
        self.turn_assist_deadband = float(self.get_parameter("turn_assist_deadband").value)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.input_cmd_topic, self.cmd_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.status_pub = self.create_publisher(Bool, self.status_topic, 10)

        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 0.1), self.timer_callback)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd: Optional[Twist] = None
        now = self.get_clock().now()
        self.last_cmd_time = now

        self.get_logger().info(
            "Safety controller active: scan=%s input=%s output=%s stop=%.2fm"
            % (self.scan_topic, self.input_cmd_topic, self.output_cmd_topic, self.stop_distance)
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def cmd_callback(self, msg: Twist) -> None:
        self.latest_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def timer_callback(self) -> None:
        cmd = Twist()
        now = self.get_clock().now()

        if self.latest_scan is None:
            self.status_pub.publish(Bool(data=False))
            self.cmd_pub.publish(cmd)
            return

        front_min = self.scan_min_in_arc(self.latest_scan, -self.front_arc, self.front_arc)
        blocked = 0.0 < front_min < self.stop_distance
        self.status_pub.publish(Bool(data=blocked))
        cmd = self.filter_nominal_command(front_min, now)
        self.cmd_pub.publish(cmd)

    def filter_nominal_command(self, front_min: float, now) -> Twist:
        cmd = Twist()
        if self.latest_cmd is None:
            return cmd
        if (now - self.last_cmd_time) > Duration(seconds=self.cmd_timeout_sec):
            return cmd

        cmd.linear.x = min(max(self.latest_cmd.linear.x, 0.0), self.max_forward_speed)
        cmd.angular.z = max(-self.max_turn_speed, min(self.max_turn_speed, self.latest_cmd.angular.z))
        left_min = self.scan_min_in_arc(self.latest_scan, 0.0, self.front_arc)
        right_min = self.scan_min_in_arc(self.latest_scan, -self.front_arc, 0.0)
        turn_sign = 1.0 if left_min >= right_min else -1.0

        if 0.0 < front_min < self.stop_distance:
            cmd.linear.x = 0.0
            if abs(cmd.angular.z) < self.turn_assist_deadband:
                cmd.angular.z = turn_sign * min(self.turn_assist_speed, self.max_turn_speed)
            return cmd

        if 0.0 < front_min < self.slow_distance and cmd.linear.x > 0.0:
            scale = (front_min - self.stop_distance) / max(self.slow_distance - self.stop_distance, 1e-3)
            scale = max(0.0, min(1.0, scale))
            cmd.linear.x *= scale
            if front_min < self.danger_distance and abs(cmd.angular.z) < self.turn_assist_deadband:
                cmd.angular.z = turn_sign * min(self.turn_assist_speed, self.max_turn_speed)

        return cmd

    def scan_min_in_arc(self, scan: LaserScan, arc_min: float, arc_max: float) -> float:
        if arc_max < arc_min:
            return 0.0

        angle_min = float(scan.angle_min)
        angle_max = float(scan.angle_max)
        angle_inc = float(scan.angle_increment)
        if angle_inc <= 0.0 or not scan.ranges:
            return 0.0

        start = max(arc_min, angle_min)
        end = min(arc_max, angle_max)
        if end < start:
            return 0.0

        idx_start = max(0, int((start - angle_min) / angle_inc))
        idx_end = min(len(scan.ranges) - 1, int((end - angle_min) / angle_inc))

        values = []
        for idx in range(idx_start, idx_end + 1):
            r = scan.ranges[idx]
            if math.isfinite(r):
                values.append(float(r))
            elif math.isinf(r) and r > 0.0:
                values.append(float(scan.range_max))

        if not values:
            return 0.0
        return min(values)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyController()
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
