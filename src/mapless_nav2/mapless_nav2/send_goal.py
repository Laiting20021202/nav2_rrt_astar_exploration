#!/usr/bin/env python3
import argparse
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.node import Node


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


class GoalSender(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("mapless_goal_sender")
        self.pub = self.create_publisher(PoseStamped, topic, 10)

    def publish_goal(self, x: float, y: float, yaw: float, frame: str) -> None:
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation = quaternion_from_yaw(yaw)

        # Publish multiple times to avoid startup race with subscriptions.
        for _ in range(5):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            "Goal published to %s: frame=%s x=%.3f y=%.3f yaw=%.3f"
            % (self.pub.topic_name, frame, x, y, yaw)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a mapless nav goal pose.")
    parser.add_argument("--x", type=float, required=True, help="Goal x in selected frame")
    parser.add_argument("--y", type=float, required=True, help="Goal y in selected frame")
    parser.add_argument("--yaw", type=float, default=0.0, help="Goal yaw in rad")
    parser.add_argument("--frame", type=str, default="odom", help="Frame id (default: odom)")
    parser.add_argument("--topic", type=str, default="/mapless_goal", help="Goal topic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GoalSender(args.topic)
    try:
        node.publish_goal(args.x, args.y, args.yaw, args.frame)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
