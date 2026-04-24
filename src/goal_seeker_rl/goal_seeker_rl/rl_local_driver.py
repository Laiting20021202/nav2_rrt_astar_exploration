"""Hierarchical RL local driver node.

This node acts as the local driver:
1) Read LiDAR from `/scan`.
2) Read FAR-style tactical waypoint from `/hrl_local_waypoint`.
3) Convert waypoint into base_link-relative distance/angle.
4) Run RL policy inference for local obstacle avoidance + local planning.
5) Publish cmd_vel.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
import torch

from .networks import Actor, ReferenceActor


class RLLocalDriver(Node):
    """Bridge between tactical waypoint and RL policy action output."""

    def __init__(self) -> None:
        super().__init__("rl_local_driver")
        self._declare_parameters()

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.scan_samples = int(self.get_parameter("scan_samples").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.linear_speed_max = float(self.get_parameter("linear_speed_max").value)
        self.angular_speed_max = float(self.get_parameter("angular_speed_max").value)
        self.waypoint_timeout_sec = float(self.get_parameter("waypoint_timeout_sec").value)
        self.obstacle_stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        self.obstacle_slow_distance = float(self.get_parameter("obstacle_slow_distance").value)
        self.obstacle_hard_stop_distance = float(
            self.get_parameter("obstacle_hard_stop_distance").value
        )
        self.side_guard_distance = float(self.get_parameter("side_guard_distance").value)
        self.avoid_turn_boost = float(self.get_parameter("avoid_turn_boost").value)
        self.progress_window_sec = float(self.get_parameter("progress_window_sec").value)
        self.progress_min_delta = float(self.get_parameter("progress_min_delta").value)
        self.waypoint_close_distance = float(self.get_parameter("waypoint_close_distance").value)
        self.waypoint_stop_distance = float(self.get_parameter("waypoint_stop_distance").value)
        self.turn_in_place_angle = float(self.get_parameter("turn_in_place_angle").value)
        self.policy_blend_far = float(self.get_parameter("policy_blend_far").value)
        self.policy_blend_near_obstacle = float(
            self.get_parameter("policy_blend_near_obstacle").value
        )
        self.policy_blend_obstacle_distance = float(
            self.get_parameter("policy_blend_obstacle_distance").value
        )
        self.enable_local_escape = bool(self.get_parameter("enable_local_escape").value)

        self.policy_source = str(self.get_parameter("policy_source").value).lower()
        self.model_path = str(self.get_parameter("model_path").value)
        self.network_variant = str(self.get_parameter("network_variant").value).lower()
        self.hidden_dim = int(self.get_parameter("hidden_dim").value)
        self.model_strict = bool(self.get_parameter("model_strict").value)
        self.append_prev_action_to_state = bool(self.get_parameter("append_prev_action_to_state").value)
        self.policy_max_goal_distance = float(self.get_parameter("policy_max_goal_distance").value)
        self.use_cuda = bool(self.get_parameter("use_cuda").value)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_scan_norm = np.ones(self.scan_samples, dtype=np.float32)
        self.latest_scan_ranges = np.full(self.scan_samples, self.lidar_max_range, dtype=np.float32)
        self.latest_scan_angles = np.linspace(-math.pi, math.pi, self.scan_samples, dtype=np.float32)
        self.latest_waypoint: Optional[PoseStamped] = None
        self.latest_waypoint_recv_sec = -1e9
        self.has_scan = False
        self._last_debug_sec = -1e9
        self._last_tf_warn_sec = -1e9
        self.prev_action_norm = np.zeros(2, dtype=np.float32)
        self.waypoint_progress: Deque[tuple[float, float]] = deque()
        self.escape_turn_sign = 1.0

        self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")
        self.actor: Optional[torch.nn.Module] = None
        self._init_policy()

        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.waypoint_topic, self._waypoint_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_timer(max(1e-3, 1.0 / self.control_rate_hz), self._control_loop)

        self.get_logger().info(
            "RL local driver ready. Waiting for scan + /hrl_local_waypoint ..."
        )

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by this node."""
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("waypoint_topic", "/hrl_local_waypoint")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("scan_samples", 40)
        self.declare_parameter("lidar_max_range", 3.5)
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("linear_speed_max", 0.20)
        self.declare_parameter("angular_speed_max", 2.2)
        self.declare_parameter("waypoint_timeout_sec", 2.0)
        self.declare_parameter("obstacle_stop_distance", 0.22)
        self.declare_parameter("obstacle_slow_distance", 0.38)
        self.declare_parameter("obstacle_hard_stop_distance", 0.28)
        self.declare_parameter("side_guard_distance", 0.24)
        self.declare_parameter("avoid_turn_boost", 0.42)
        self.declare_parameter("progress_window_sec", 4.0)
        self.declare_parameter("progress_min_delta", 0.20)
        self.declare_parameter("waypoint_close_distance", 0.45)
        self.declare_parameter("waypoint_stop_distance", 0.15)
        self.declare_parameter("turn_in_place_angle", 1.25)
        self.declare_parameter("policy_blend_far", 0.15)
        self.declare_parameter("policy_blend_near_obstacle", 0.40)
        self.declare_parameter("policy_blend_obstacle_distance", 0.70)
        self.declare_parameter("enable_local_escape", False)

        # RL policy settings.
        self.declare_parameter("policy_source", "reference_actor")
        self.declare_parameter("model_path", "/home/david/Desktop/laiting/rl_base_navigation/reference/turtlebot3_drlnav/src/turtlebot3_drl/model/examples/ddpg_0_stage9/actor_stage9_episode8000.pt")
        self.declare_parameter("network_variant", "reference")
        self.declare_parameter("hidden_dim", 512)
        self.declare_parameter("model_strict", False)
        self.declare_parameter("append_prev_action_to_state", True)
        self.declare_parameter("policy_max_goal_distance", 5.94)
        self.declare_parameter("use_cuda", True)

    def _init_policy(self) -> None:
        """Initialize model for local policy inference."""
        if self.policy_source in ("none", "heuristic", "fallback"):
            self.get_logger().warn("Local driver uses heuristic fallback (no model).")
            self.actor = None
            return

        if not self.model_path:
            self.get_logger().warn("model_path is empty. Falling back to heuristic local policy.")
            self.actor = None
            return

        state_dim = self.scan_samples + 2 + (2 if self.append_prev_action_to_state else 0)
        use_reference_actor = (self.policy_source == "reference_actor") or (self.network_variant == "reference")
        actor_cls = ReferenceActor if use_reference_actor else Actor

        try:
            self.actor = actor_cls(state_dim=state_dim, action_dim=2, hidden_dim=self.hidden_dim).to(self.device)
            payload = torch.load(self.model_path, map_location=self.device)
            actor_state = payload.get("actor") if isinstance(payload, dict) and ("actor" in payload) else payload
            self.actor.load_state_dict(actor_state, strict=self.model_strict)
            self.actor.eval()
            self.get_logger().info(
                f"Local policy loaded: source={self.policy_source} variant={self.network_variant} "
                f"state_dim={state_dim} device={self.device} path={self.model_path}"
            )
        except Exception as exc:
            self.actor = None
            self.get_logger().warn(
                f"Failed to load local policy model ({exc}). Falling back to heuristic policy."
            )

    def _scan_callback(self, msg: LaserScan) -> None:
        """Downsample scan into fixed-size LiDAR feature vector."""
        if not msg.ranges:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        fill = min(float(msg.range_max), self.lidar_max_range)
        ranges = np.nan_to_num(ranges, nan=fill, posinf=fill, neginf=fill)
        ranges = np.clip(ranges, float(msg.range_min), self.lidar_max_range)

        if len(ranges) >= self.scan_samples and (len(ranges) % self.scan_samples == 0):
            stride = len(ranges) // self.scan_samples
            idx = np.arange(0, stride * self.scan_samples, stride, dtype=np.int32)
        else:
            idx = np.linspace(0, len(ranges) - 1, self.scan_samples, dtype=np.int32)

        sampled = ranges[idx].astype(np.float32)
        sampled_angles = float(msg.angle_min) + idx.astype(np.float32) * float(msg.angle_increment)
        self.latest_scan_ranges = sampled
        self.latest_scan_angles = sampled_angles.astype(np.float32)
        self.latest_scan_norm = np.clip(sampled / self.lidar_max_range, 0.0, 1.0).astype(np.float32)
        self.has_scan = True

    def _waypoint_callback(self, msg: PoseStamped) -> None:
        """Store latest tactical waypoint."""
        reset_progress = False
        if self.latest_waypoint is None:
            reset_progress = True
        elif msg.header.frame_id != self.latest_waypoint.header.frame_id:
            reset_progress = True
        else:
            dx = float(msg.pose.position.x - self.latest_waypoint.pose.position.x)
            dy = float(msg.pose.position.y - self.latest_waypoint.pose.position.y)
            # Planner republishes every 0.5s; only reset when waypoint jumps noticeably.
            reset_progress = math.hypot(dx, dy) > max(0.25, 0.6 * self.waypoint_close_distance)

        self.latest_waypoint = msg
        self.latest_waypoint_recv_sec = self._now_sec()
        if reset_progress:
            self.waypoint_progress.clear()

    def _control_loop(self) -> None:
        """Main inference/control loop."""
        now = self._now_sec()
        if (not self.has_scan) or (self.latest_waypoint is None):
            self._publish_stop()
            return
        if (now - self.latest_waypoint_recv_sec) > self.waypoint_timeout_sec:
            self._publish_stop()
            return

        rel = self._waypoint_in_base_frame(self.latest_waypoint)
        if rel is None:
            self._publish_stop()
            return
        rel_x, rel_y = rel
        relative_dist = float(math.hypot(rel_x, rel_y))
        relative_angle = float(math.atan2(rel_y, rel_x))
        self._update_waypoint_progress(now, relative_dist)
        stuck_local = self._is_local_stuck(now, relative_dist)

        # Waypoint already reached: wait for planner to roll forward.
        if relative_dist < self.waypoint_stop_distance:
            self._publish_stop()
            return

        dist_norm = float(np.clip(relative_dist / max(self.policy_max_goal_distance, 1e-3), 0.0, 1.0))
        angle_norm = float(np.clip(relative_angle / math.pi, -1.0, 1.0))
        state = np.concatenate(
            [self.latest_scan_norm, np.array([dist_norm, angle_norm], dtype=np.float32)]
        )
        if self.append_prev_action_to_state:
            state = np.concatenate([state, self.prev_action_norm]).astype(np.float32)

        # Hard collision guard.
        front_mask = np.abs(self.latest_scan_angles) < math.radians(25.0)
        front_min = float(np.min(self.latest_scan_ranges[front_mask])) if np.any(front_mask) else self.lidar_max_range
        front_left_mask = (self.latest_scan_angles > math.radians(15.0)) & (
            self.latest_scan_angles < math.radians(75.0)
        )
        front_right_mask = (self.latest_scan_angles < -math.radians(15.0)) & (
            self.latest_scan_angles > -math.radians(75.0)
        )
        front_left_min = (
            float(np.min(self.latest_scan_ranges[front_left_mask]))
            if np.any(front_left_mask)
            else self.lidar_max_range
        )
        front_right_min = (
            float(np.min(self.latest_scan_ranges[front_right_mask]))
            if np.any(front_right_mask)
            else self.lidar_max_range
        )
        left_mask = self.latest_scan_angles > 0.0
        right_mask = self.latest_scan_angles < 0.0
        left_clear = float(np.mean(self.latest_scan_ranges[left_mask])) if np.any(left_mask) else self.lidar_max_range
        right_clear = float(np.mean(self.latest_scan_ranges[right_mask])) if np.any(right_mask) else self.lidar_max_range
        base_action = self._path_follow_action(
            relative_dist=relative_dist,
            relative_angle=relative_angle,
            front_min=front_min,
            left_clear=left_clear,
            right_clear=right_clear,
        )

        if self.actor is not None:
            # RL is the primary local planner / avoider.
            policy_action = np.asarray(
                self.predict_action(state, relative_dist, relative_angle),
                dtype=np.float32,
            )
            if front_min < self.policy_blend_obstacle_distance:
                path_assist = self.policy_blend_near_obstacle
            else:
                path_assist = self.policy_blend_far
            path_assist = float(np.clip(path_assist, 0.0, 0.8))

            # When waypoint is stale, rely even more on RL local exploration behavior.
            if stuck_local and relative_dist > max(0.30, self.waypoint_close_distance):
                path_assist = min(path_assist, 0.18)

            action = np.clip(
                (1.0 - path_assist) * policy_action + path_assist * base_action,
                -1.0,
                1.0,
            )
            rl_weight = 1.0 - path_assist
        else:
            action = base_action
            rl_weight = 0.0

        self.prev_action_norm = action.copy()
        linear = float((action[0] + 1.0) * 0.5 * self.linear_speed_max)
        angular = float(action[1] * self.angular_speed_max)
        angular = float(np.clip(angular, -self.angular_speed_max, self.angular_speed_max))

        # Do not keep pushing forward when heading error is large; rotate decisively.
        if abs(relative_angle) > self.turn_in_place_angle:
            linear = min(linear, 0.03 if front_min > (self.obstacle_stop_distance + 0.06) else 0.0)
            angular = float(
                np.clip(
                    math.copysign(max(abs(angular), 0.75 * self.angular_speed_max), relative_angle),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )

        # Near waypoint: prioritize passing through instead of orbiting in place.
        if (
            relative_dist < self.waypoint_close_distance
            and front_min > self.obstacle_slow_distance
            and abs(relative_angle) < 0.9
        ):
            # Keep moving through waypoint area, but avoid forced forward push near obstacles.
            linear = max(linear, min(0.22 * self.linear_speed_max, 0.05))

        # If distance-to-waypoint does not improve for several seconds, force exploration turn.
        if (
            self.enable_local_escape
            and stuck_local
            and relative_dist > max(0.30, self.waypoint_close_distance)
        ):
            if abs(left_clear - right_clear) > 0.08:
                self.escape_turn_sign = 1.0 if left_clear >= right_clear else -1.0
            angular = float(
                np.clip(
                    self.escape_turn_sign * max(abs(angular), 0.85 * self.angular_speed_max),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )
            linear = min(linear, 0.03 if front_min > (self.obstacle_stop_distance + 0.05) else 0.0)

        # Safety shield: never allow close-wall forward push due to policy or near-waypoint override.
        if front_min < self.obstacle_hard_stop_distance:
            linear = 0.0
            turn_sign = 1.0 if left_clear >= right_clear else -1.0
            angular = float(
                np.clip(
                    turn_sign * max(abs(angular), 0.95 * self.angular_speed_max),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )
        elif front_min < self.obstacle_slow_distance:
            linear = min(linear, 0.04)
            # Fast reactive turn when entering a narrow obstacle zone.
            inv_fl = 1.0 / max(front_left_min, 0.08)
            inv_fr = 1.0 / max(front_right_min, 0.08)
            side_bias = float(np.clip(inv_fr - inv_fl, -2.0, 2.0))
            angular += self.avoid_turn_boost * side_bias * self.angular_speed_max
            angular = float(np.clip(angular, -self.angular_speed_max, self.angular_speed_max))
            linear = min(
                linear,
                float(
                    np.clip(
                        0.10 * (front_min / max(self.obstacle_slow_distance, 1e-3)),
                        0.02,
                        0.08,
                    )
                ),
            )

        # Side guard: avoid steering into nearby side wall while squeezing through corridors.
        if front_left_min < self.side_guard_distance and angular > 0.0:
            angular = min(
                angular,
                -0.22
                * self.angular_speed_max
                * float(np.clip((self.side_guard_distance - front_left_min) / self.side_guard_distance, 0.0, 1.0)),
            )
        if front_right_min < self.side_guard_distance and angular < 0.0:
            angular = max(
                angular,
                0.22
                * self.angular_speed_max
                * float(np.clip((self.side_guard_distance - front_right_min) / self.side_guard_distance, 0.0, 1.0)),
            )
        if min(front_left_min, front_right_min) < self.side_guard_distance:
            linear = min(linear, 0.05)

        if (now - self._last_debug_sec) > 1.0:
            self._last_debug_sec = now
            policy_mode = "model" if self.actor is not None else "heuristic"
            self.get_logger().info(
                f"ctrl[{policy_mode}] | dist={relative_dist:.2f} ang={relative_angle:.2f} "
                f"front={front_min:.2f} fl={front_left_min:.2f} fr={front_right_min:.2f} "
                f"lin={linear:.3f} angz={angular:.3f} "
                f"rl_weight={rl_weight:.2f} stuck_local={stuck_local}"
            )

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def _waypoint_in_base_frame(self, waypoint: PoseStamped) -> Optional[tuple[float, float]]:
        """Transform waypoint pose into base_link and return (x, y)."""
        wp_base = self._transform_pose(waypoint, self.base_frame)
        if wp_base is None:
            return None
        return float(wp_base.pose.position.x), float(wp_base.pose.position.y)

    def _transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """TF transform helper."""
        if pose.header.frame_id == target_frame:
            return pose
        # Query latest available TF and apply to the input pose.
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                Time(seconds=0.0),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            now = self._now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"TF transform failed ({pose.header.frame_id} -> {target_frame}): {exc}"
                )
            return None

        req = PoseStamped()
        req.header.frame_id = pose.header.frame_id
        req.header.stamp = tf.header.stamp
        req.pose = pose.pose
        try:
            out = do_transform_pose_stamped(req, tf)
        except Exception as exc:
            now = self._now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"Pose transform error ({pose.header.frame_id} -> {target_frame}): {exc}"
                )
            return None
        out.header.stamp = tf.header.stamp
        out.header.frame_id = target_frame
        return out

    def predict_action(
        self,
        state: np.ndarray,
        relative_dist: float,
        relative_angle: float,
    ) -> np.ndarray:
        """Return normalized action in [-1, 1] via model or heuristic fallback."""
        if self.actor is not None:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(state_t).cpu().numpy()[0]
            return np.clip(action.astype(np.float32), -1.0, 1.0)
        return self._heuristic_action(relative_dist, relative_angle)

    def _heuristic_action(self, relative_dist: float, relative_angle: float) -> np.ndarray:
        """Fallback local planner when model is unavailable."""
        scan_angles = self.latest_scan_angles
        scan_ranges = self.latest_scan_ranges
        if scan_ranges.size == 0:
            return np.array([-1.0, 0.0], dtype=np.float32)

        front_mask = np.abs(scan_angles) < math.radians(25.0)
        fl_mask = (scan_angles > math.radians(20.0)) & (scan_angles < math.radians(100.0))
        fr_mask = (scan_angles < -math.radians(20.0)) & (scan_angles > -math.radians(100.0))
        front_min = float(np.min(scan_ranges[front_mask])) if np.any(front_mask) else self.lidar_max_range
        front_left_min = float(np.min(scan_ranges[fl_mask])) if np.any(fl_mask) else self.lidar_max_range
        front_right_min = float(np.min(scan_ranges[fr_mask])) if np.any(fr_mask) else self.lidar_max_range

        inv_l = 1.0 / max(front_left_min, 0.05)
        inv_r = 1.0 / max(front_right_min, 0.05)
        repulsive = float(np.clip(inv_r - inv_l, -1.5, 1.5))

        k_goal = 1.25
        k_avoid = 0.35
        angular_cmd = float(
            np.clip(
                k_goal * relative_angle + k_avoid * repulsive,
                -self.angular_speed_max,
                self.angular_speed_max,
            )
        )

        if abs(relative_angle) > 1.2:
            angular_cmd = float(
                np.clip(
                    math.copysign(max(1.0, 0.8 * self.angular_speed_max), relative_angle),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )
            linear_cmd = 0.0
            return np.array([-1.0, angular_cmd / max(self.angular_speed_max, 1e-3)], dtype=np.float32)

        blocked_front = front_min < max(self.obstacle_stop_distance + 0.10, 0.32)
        if blocked_front:
            turn_sign = -1.0 if front_left_min < front_right_min else 1.0
            angular_cmd = float(
                np.clip(
                    turn_sign * max(0.75 * self.angular_speed_max, abs(angular_cmd)),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )
            linear_cmd = 0.0
        else:
            heading_factor = float(np.clip(1.0 - abs(angular_cmd) / max(self.angular_speed_max, 1e-3), 0.0, 1.0))
            clearance_factor = float(
                np.clip(
                    (front_min - self.obstacle_stop_distance) / max(0.8 - self.obstacle_stop_distance, 1e-3),
                    0.0,
                    1.0,
                )
            )
            dist_factor = float(np.clip(relative_dist / 0.8, 0.4, 1.0))
            linear_cmd = self.linear_speed_max * (0.25 + 0.75 * heading_factor) * clearance_factor * dist_factor
            if abs(relative_angle) > 1.0:
                linear_cmd = min(linear_cmd, 0.03)
            linear_cmd = float(np.clip(linear_cmd, 0.0, self.linear_speed_max))

        linear_norm = float(np.clip(2.0 * linear_cmd / max(self.linear_speed_max, 1e-3) - 1.0, -1.0, 1.0))
        angular_norm = float(np.clip(angular_cmd / max(self.angular_speed_max, 1e-3), -1.0, 1.0))
        return np.array([linear_norm, angular_norm], dtype=np.float32)

    def _path_follow_action(
        self,
        relative_dist: float,
        relative_angle: float,
        front_min: float,
        left_clear: float,
        right_clear: float,
    ) -> np.ndarray:
        """Deterministic path-follow action used as the primary controller."""
        k_heading = 1.35
        angular_cmd = float(
            np.clip(
                k_heading * relative_angle,
                -self.angular_speed_max,
                self.angular_speed_max,
            )
        )

        inv_l = 1.0 / max(left_clear, 0.05)
        inv_r = 1.0 / max(right_clear, 0.05)
        repulsive = float(np.clip(inv_r - inv_l, -1.8, 1.8))
        avoid_gain = float(np.clip((self.policy_blend_obstacle_distance - front_min) / 0.8, 0.0, 1.0))
        angular_cmd += 0.55 * avoid_gain * repulsive
        angular_cmd = float(np.clip(angular_cmd, -self.angular_speed_max, self.angular_speed_max))

        heading_factor = float(np.clip(1.0 - abs(relative_angle) / 1.2, 0.0, 1.0))
        clearance_factor = float(
            np.clip(
                (front_min - self.obstacle_stop_distance) / max(0.9 - self.obstacle_stop_distance, 1e-3),
                0.0,
                1.0,
            )
        )
        dist_factor = float(np.clip(relative_dist / 1.2, 0.15, 1.0))
        linear_cmd = self.linear_speed_max * heading_factor * clearance_factor * dist_factor
        linear_cmd = float(np.clip(linear_cmd, 0.0, self.linear_speed_max))

        if front_min < (self.obstacle_stop_distance + 0.04):
            linear_cmd = 0.0
            turn_sign = 1.0 if left_clear >= right_clear else -1.0
            angular_cmd = float(
                np.clip(
                    turn_sign * max(0.8 * self.angular_speed_max, abs(angular_cmd)),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )

        linear_norm = float(np.clip(2.0 * linear_cmd / max(self.linear_speed_max, 1e-3) - 1.0, -1.0, 1.0))
        angular_norm = float(np.clip(angular_cmd / max(self.angular_speed_max, 1e-3), -1.0, 1.0))
        return np.array([linear_norm, angular_norm], dtype=np.float32)

    def _publish_stop(self) -> None:
        """Publish zero velocity."""
        self.cmd_pub.publish(Twist())

    def _update_waypoint_progress(self, now_sec: float, dist: float) -> None:
        """Track recent waypoint distance for local stuck detection."""
        self.waypoint_progress.append((now_sec, dist))
        min_t = now_sec - self.progress_window_sec
        while self.waypoint_progress and self.waypoint_progress[0][0] < min_t:
            self.waypoint_progress.popleft()

    def _is_local_stuck(self, now_sec: float, current_dist: float) -> bool:
        """Return True when waypoint distance barely improves during the window."""
        if len(self.waypoint_progress) < 3:
            return False
        dt = self.waypoint_progress[-1][0] - self.waypoint_progress[0][0]
        if dt < (0.8 * self.progress_window_sec):
            return False
        d0 = self.waypoint_progress[0][1]
        best = min(d for _, d in self.waypoint_progress)
        improved = d0 - best
        return (improved < self.progress_min_delta) and (current_dist > 0.35)

    def _now_sec(self) -> float:
        """Return current ROS time in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args: Optional[list[str]] = None) -> None:
    """Entrypoint for ROS 2 execution."""
    rclpy.init(args=args)
    node = RLLocalDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
