"""Hierarchical RL local driver node.

This node acts as the local driver:
1) Read the Realsense-derived LaserScan from `/scan`.
2) Read FAR-style tactical waypoint from `/hrl_local_waypoint`.
3) Convert waypoint into base_link-relative distance/angle.
4) Run RL policy inference for local obstacle avoidance + local planning.
5) Publish cmd_vel.
"""

from __future__ import annotations

from collections import deque
import math
import os
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
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
        self.goal_active_topic = str(self.get_parameter("goal_active_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.scan_samples = int(self.get_parameter("scan_samples").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.linear_speed_max = float(self.get_parameter("linear_speed_max").value)
        self.angular_speed_max = float(self.get_parameter("angular_speed_max").value)
        self.goal_active_timeout_sec = float(self.get_parameter("goal_active_timeout_sec").value)
        self.waypoint_timeout_sec = float(self.get_parameter("waypoint_timeout_sec").value)
        self.obstacle_stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        self.obstacle_slow_distance = float(self.get_parameter("obstacle_slow_distance").value)
        self.obstacle_hard_stop_distance = float(
            self.get_parameter("obstacle_hard_stop_distance").value
        )
        self.emergency_stop_distance = float(
            self.get_parameter("emergency_stop_distance").value
        )
        self.proactive_avoid_distance = float(
            self.get_parameter("proactive_avoid_distance").value
        )
        self.side_guard_distance = float(self.get_parameter("side_guard_distance").value)
        self.avoid_turn_boost = float(self.get_parameter("avoid_turn_boost").value)
        self.progress_window_sec = float(self.get_parameter("progress_window_sec").value)
        self.progress_min_delta = float(self.get_parameter("progress_min_delta").value)
        self.waypoint_close_distance = float(self.get_parameter("waypoint_close_distance").value)
        self.waypoint_stop_distance = float(self.get_parameter("waypoint_stop_distance").value)
        self.turn_in_place_angle = float(self.get_parameter("turn_in_place_angle").value)
        self.orbit_break_angle = float(self.get_parameter("orbit_break_angle").value)
        self.policy_blend_far = float(self.get_parameter("policy_blend_far").value)
        self.policy_blend_near_obstacle = float(
            self.get_parameter("policy_blend_near_obstacle").value
        )
        self.policy_blend_obstacle_distance = float(
            self.get_parameter("policy_blend_obstacle_distance").value
        )
        self.enable_local_escape = bool(self.get_parameter("enable_local_escape").value)
        self.publish_rl_path = bool(self.get_parameter("publish_rl_path").value)
        self.rl_path_topic = str(self.get_parameter("rl_path_topic").value)
        self.rl_path_frame = str(self.get_parameter("rl_path_frame").value)
        self.rl_path_horizon_steps = int(self.get_parameter("rl_path_horizon_steps").value)
        self.rl_path_dt_sec = float(self.get_parameter("rl_path_dt_sec").value)
        self.lookaround_enabled = bool(self.get_parameter("lookaround_enabled").value)
        self.lookaround_front_distance = float(self.get_parameter("lookaround_front_distance").value)
        self.lookaround_clear_distance = float(self.get_parameter("lookaround_clear_distance").value)
        self.lookaround_turn_speed = float(self.get_parameter("lookaround_turn_speed").value)
        self.lookaround_duration_sec = float(self.get_parameter("lookaround_duration_sec").value)
        self.lookaround_min_duration_sec = float(self.get_parameter("lookaround_min_duration_sec").value)
        self.lookaround_cooldown_sec = float(self.get_parameter("lookaround_cooldown_sec").value)
        self.waiting_scan_creep_enabled = bool(
            self.get_parameter("waiting_scan_creep_enabled").value
        )
        self.waiting_scan_creep_speed = float(self.get_parameter("waiting_scan_creep_speed").value)
        self.waiting_scan_turn_speed = float(self.get_parameter("waiting_scan_turn_speed").value)
        self.waiting_scan_front_clearance = float(
            self.get_parameter("waiting_scan_front_clearance").value
        )

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
        self.raw_scan_ranges = np.full(360, self.lidar_max_range, dtype=np.float32)
        self.raw_scan_angles = np.linspace(-math.pi, math.pi, 360, dtype=np.float32)
        self.latest_waypoint: Optional[PoseStamped] = None
        self.latest_waypoint_recv_sec = -1e9
        self.goal_active = False
        self.goal_active_recv_sec = -1e9
        self.has_scan = False
        self._last_debug_sec = -1e9
        self._last_tf_warn_sec = -1e9
        self.prev_action_norm = np.zeros(2, dtype=np.float32)
        self.waypoint_progress: Deque[tuple[float, float]] = deque()
        self.escape_turn_sign = 1.0
        self.lookaround_until_sec = -1e9
        self.lookaround_started_sec = -1e9
        self.lookaround_cooldown_until_sec = -1e9
        self.lookaround_turn_sign = 1.0

        self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")
        self.actor: Optional[torch.nn.Module] = None
        self._init_policy()

        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.waypoint_topic, self._waypoint_callback, 10)
        self.create_subscription(Bool, self.goal_active_topic, self._goal_active_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.rl_path_pub = self.create_publisher(Path, self.rl_path_topic, 1) if self.publish_rl_path else None
        self.create_timer(max(1e-3, 1.0 / self.control_rate_hz), self._control_loop)

        self.get_logger().info(
            "RL local driver ready. Waiting for scan + /hrl_local_waypoint ..."
        )

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by this node."""
        default_workspace = os.environ.get("RL_BASE_WS", "/home/david/Desktop/laiting/rl_base_navigation")
        default_model_dir = os.environ.get("RL_BASE_MODEL_DIR", os.path.join(default_workspace, "navigation_model"))
        default_model_path = os.path.join(default_model_dir, "td3_latest.pth")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("waypoint_topic", "/hrl_local_waypoint")
        self.declare_parameter("goal_active_topic", "/hrl_goal_active")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("scan_samples", 40)
        self.declare_parameter("lidar_max_range", 3.5)
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("linear_speed_max", 0.20)
        self.declare_parameter("angular_speed_max", 2.2)
        self.declare_parameter("goal_active_timeout_sec", 1.2)
        self.declare_parameter("waypoint_timeout_sec", 2.0)
        self.declare_parameter("obstacle_stop_distance", 0.22)
        self.declare_parameter("obstacle_slow_distance", 0.38)
        self.declare_parameter("obstacle_hard_stop_distance", 0.28)
        self.declare_parameter("emergency_stop_distance", 0.24)
        self.declare_parameter("proactive_avoid_distance", 0.85)
        self.declare_parameter("side_guard_distance", 0.24)
        self.declare_parameter("avoid_turn_boost", 0.42)
        self.declare_parameter("progress_window_sec", 4.0)
        self.declare_parameter("progress_min_delta", 0.20)
        self.declare_parameter("waypoint_close_distance", 0.45)
        self.declare_parameter("waypoint_stop_distance", 0.15)
        self.declare_parameter("turn_in_place_angle", 1.25)
        self.declare_parameter("orbit_break_angle", 1.65)
        self.declare_parameter("policy_blend_far", 0.15)
        self.declare_parameter("policy_blend_near_obstacle", 0.40)
        self.declare_parameter("policy_blend_obstacle_distance", 0.70)
        self.declare_parameter("enable_local_escape", False)
        self.declare_parameter("publish_rl_path", True)
        self.declare_parameter("rl_path_topic", "/rl_model_path")
        self.declare_parameter("rl_path_frame", "base_link")
        self.declare_parameter("rl_path_horizon_steps", 28)
        self.declare_parameter("rl_path_dt_sec", 0.20)
        self.declare_parameter("lookaround_enabled", True)
        self.declare_parameter("lookaround_front_distance", 0.75)
        self.declare_parameter("lookaround_clear_distance", 1.10)
        self.declare_parameter("lookaround_turn_speed", 0.75)
        self.declare_parameter("lookaround_duration_sec", 1.30)
        self.declare_parameter("lookaround_min_duration_sec", 0.45)
        self.declare_parameter("lookaround_cooldown_sec", 0.80)
        self.declare_parameter("waiting_scan_creep_enabled", True)
        self.declare_parameter("waiting_scan_creep_speed", 0.06)
        self.declare_parameter("waiting_scan_turn_speed", 0.35)
        self.declare_parameter("waiting_scan_front_clearance", 1.0)

        # RL policy settings.
        self.declare_parameter("policy_source", "td3")
        self.declare_parameter("model_path", default_model_path)
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
        """Downsample scan into fixed-size range feature vector."""
        if not msg.ranges:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        fill = min(float(msg.range_max), self.lidar_max_range)
        ranges = np.nan_to_num(ranges, nan=fill, posinf=fill, neginf=fill)
        ranges = np.clip(ranges, float(msg.range_min), self.lidar_max_range)
        raw_angles = float(msg.angle_min) + np.arange(len(ranges), dtype=np.float32) * float(msg.angle_increment)
        self.raw_scan_ranges = ranges.astype(np.float32)
        self.raw_scan_angles = raw_angles.astype(np.float32)

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

    def _goal_active_callback(self, msg: Bool) -> None:
        """Store whether global planner currently has an active final goal."""
        self.goal_active = bool(msg.data)
        self.goal_active_recv_sec = self._now_sec()
        if not self.goal_active:
            self.latest_waypoint = None
            self.waypoint_progress.clear()
            self.prev_action_norm[:] = 0.0

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
        if (not self.goal_active) or ((now - self.goal_active_recv_sec) > self.goal_active_timeout_sec):
            self._publish_stop()
            return
        if not self.has_scan:
            self._publish_stop()
            return
        if self.latest_waypoint is None:
            self._publish_waiting_lookaround(now)
            return
        if (now - self.latest_waypoint_recv_sec) > self.waypoint_timeout_sec:
            self._publish_waiting_lookaround(now)
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

        front_min = self._sector_min(-20.0, 20.0)
        front_wide_min = self._sector_min(-45.0, 45.0)
        front_left_min = self._sector_min(15.0, 80.0)
        front_right_min = self._sector_min(-80.0, -15.0)
        left_clear = self._sector_mean(40.0, 120.0)
        right_clear = self._sector_mean(-120.0, -40.0)

        if self._publish_lookaround_if_needed(now, front_wide_min, left_clear, right_clear, relative_dist):
            return

        base_action = self._path_follow_action(
            relative_dist=relative_dist,
            relative_angle=relative_angle,
            front_min=front_wide_min,
            left_clear=left_clear,
            right_clear=right_clear,
        )

        if self.actor is not None:
            policy_action = np.asarray(
                self.predict_action(state, relative_dist, relative_angle),
                dtype=np.float32,
            )
            # 強化轉圈時的路徑跟隨權重
            if stuck_local:
                path_assist = 0.50  # 提升至 50%，讓路徑跟隨更主導
            elif front_min < self.policy_blend_obstacle_distance:
                path_assist = self.policy_blend_near_obstacle
            else:
                path_assist = self.policy_blend_far
            path_assist = float(np.clip(path_assist, 0.0, 0.8))

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

        # 強化旋轉限制：大幅降低無謂轉向
        if abs(relative_angle) > self.turn_in_place_angle:
            linear = min(linear, 0.02)
            angular = float(
                np.clip(
                    math.copysign(max(abs(angular), 0.65 * self.angular_speed_max), relative_angle),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )

        # Break local orbiting near waypoint: rotate in place first, then move.
        if relative_dist < self.waypoint_close_distance and abs(relative_angle) > self.orbit_break_angle:
            linear = 0.0
            angular = float(
                np.clip(
                    math.copysign(max(abs(angular), 0.80 * self.angular_speed_max), relative_angle),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )

        # Near waypoint: prioritize passing through instead of orbiting in place.
        if (
            relative_dist < self.waypoint_close_distance
            and front_wide_min > (self.proactive_avoid_distance + 0.10)
            and abs(relative_angle) < 0.70
        ):
            # Keep slow forward motion only when heading is aligned and nearby area is clear.
            linear = max(linear, min(0.18 * self.linear_speed_max, 0.035))

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
            linear = 0.0

        turn_sign = 1.0 if left_clear >= right_clear else -1.0

        # Safety shield + 轉圈逃脫
        if min(front_min, front_wide_min) < self.emergency_stop_distance:
            linear = 0.0
            angular = float(
                np.clip(
                    turn_sign * max(abs(angular), 0.95 * self.angular_speed_max),
                    -self.angular_speed_max,
                    self.angular_speed_max,
                )
            )
        elif front_wide_min < self.proactive_avoid_distance:
            proximity = float(
                np.clip(
                    (self.proactive_avoid_distance - front_wide_min)
                    / max(self.proactive_avoid_distance - self.emergency_stop_distance, 1e-3),
                    0.0,
                    1.0,
                )
            )
            linear_cap = float(
                np.clip(
                    self.linear_speed_max * (0.55 - 0.45 * proximity),
                    0.02,
                    self.linear_speed_max * 0.55,
                )
            )
            linear = min(linear, linear_cap)
            inv_fl = 1.0 / max(front_left_min, 0.08)
            inv_fr = 1.0 / max(front_right_min, 0.08)
            side_bias = float(np.clip(inv_fr - inv_fl, -2.0, 2.0))
            angular += (self.avoid_turn_boost + 0.35 * proximity) * side_bias * self.angular_speed_max
            min_turn = (0.25 + 0.55 * proximity) * self.angular_speed_max
            if abs(angular) < min_turn:
                angular = math.copysign(min_turn, angular if abs(angular) > 1e-4 else turn_sign)
            angular = float(np.clip(angular, -self.angular_speed_max, self.angular_speed_max))

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
            linear = min(linear, 0.03)

        # Speed governor: reduce forward speed when close to obstacles or turning hard.
        if front_wide_min < 1.20:
            clearance_ratio = float(
                np.clip(
                    (front_wide_min - self.emergency_stop_distance)
                    / max(1.20 - self.emergency_stop_distance, 1e-3),
                    0.10,
                    1.0,
                )
            )
            linear = min(linear, self.linear_speed_max * clearance_ratio)
        turn_ratio = float(np.clip(1.0 - abs(angular) / max(0.95 * self.angular_speed_max, 1e-3), 0.15, 1.0))
        linear = min(linear, self.linear_speed_max * turn_ratio)

        if (now - self._last_debug_sec) > 1.0:
            self._last_debug_sec = now
            policy_mode = "model" if self.actor is not None else "heuristic"
            self.get_logger().info(
                f"LOCAL | policy={policy_mode} dist={relative_dist:.2f} ang={relative_angle:.2f} "
                f"front={front_min:.2f} stuck={stuck_local} lin={linear:.3f} ang={angular:.3f}"
            )

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        self._publish_rl_path(linear, angular)

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

    def _sector_min(self, deg_min: float, deg_max: float) -> float:
        """Return minimum LiDAR distance in an angular sector (degrees, base_link frame)."""
        low = math.radians(min(deg_min, deg_max))
        high = math.radians(max(deg_min, deg_max))
        angles = self.raw_scan_angles if self.raw_scan_angles.size > 0 else self.latest_scan_angles
        ranges = self.raw_scan_ranges if self.raw_scan_ranges.size > 0 else self.latest_scan_ranges
        if ranges.size == 0:
            return self.lidar_max_range
        mask = (angles >= low) & (angles <= high)
        if not np.any(mask):
            return self.lidar_max_range
        return float(np.min(ranges[mask]))

    def _sector_mean(self, deg_min: float, deg_max: float) -> float:
        """Return mean LiDAR distance in an angular sector (degrees, base_link frame)."""
        low = math.radians(min(deg_min, deg_max))
        high = math.radians(max(deg_min, deg_max))
        angles = self.raw_scan_angles if self.raw_scan_angles.size > 0 else self.latest_scan_angles
        ranges = self.raw_scan_ranges if self.raw_scan_ranges.size > 0 else self.latest_scan_ranges
        if ranges.size == 0:
            return self.lidar_max_range
        mask = (angles >= low) & (angles <= high)
        if not np.any(mask):
            return self.lidar_max_range
        return float(np.mean(ranges[mask]))

    def _publish_lookaround_if_needed(
        self,
        now_sec: float,
        front_clear: float,
        left_clear: float,
        right_clear: float,
        relative_dist: float,
    ) -> bool:
        """Rotate in place briefly so the narrow Realsense FOV can inspect both sides."""
        if not self.lookaround_enabled or relative_dist < self.waypoint_close_distance:
            return False

        still_active = now_sec < self.lookaround_until_sec
        active_elapsed = now_sec - self.lookaround_started_sec
        if still_active:
            if front_clear >= self.lookaround_clear_distance and active_elapsed >= self.lookaround_min_duration_sec:
                self.lookaround_until_sec = -1e9
                self.lookaround_cooldown_until_sec = now_sec + self.lookaround_cooldown_sec
                return False
            self._publish_lookaround_turn(front_clear)
            return True

        if now_sec < self.lookaround_cooldown_until_sec:
            return False
        if front_clear >= self.lookaround_front_distance:
            return False

        if abs(left_clear - right_clear) > 0.05:
            self.lookaround_turn_sign = 1.0 if left_clear >= right_clear else -1.0
        else:
            self.lookaround_turn_sign *= -1.0
        self.lookaround_started_sec = now_sec
        self.lookaround_until_sec = now_sec + self.lookaround_duration_sec
        self._publish_lookaround_turn(front_clear)
        return True

    def _publish_lookaround_turn(self, front_clear: float) -> None:
        """Publish a short in-place scan turn and blue local arc."""
        angular = float(np.clip(self.lookaround_turn_sign * self.lookaround_turn_speed, -self.angular_speed_max, self.angular_speed_max))
        cmd = Twist()
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        self.prev_action_norm = np.array([-1.0, angular / max(self.angular_speed_max, 1e-3)], dtype=np.float32)
        self._publish_rl_path(0.0, angular)
        if (self._now_sec() - self._last_debug_sec) > 1.0:
            self._last_debug_sec = self._now_sec()
            self.get_logger().info(f"LOCAL look-around | front={front_clear:.2f} ang={angular:.2f}")

    def _publish_waiting_lookaround(self, now_sec: float) -> None:
        """Keep scanning while the global planner is waiting for enough map to publish a waypoint."""
        if not self.lookaround_enabled:
            self._publish_stop()
            return
        if now_sec >= self.lookaround_until_sec:
            self.lookaround_turn_sign *= -1.0
            self.lookaround_started_sec = now_sec
            self.lookaround_until_sec = now_sec + self.lookaround_duration_sec
        front_clear = self._sector_min(-35.0, 35.0)
        self._publish_waiting_scan_motion(front_clear)

    def _publish_waiting_scan_motion(self, front_clear: float) -> None:
        """Bootstrap SLAM with a slow safe arc when no global waypoint exists yet."""
        angular = float(
            np.clip(
                self.lookaround_turn_sign * self.waiting_scan_turn_speed,
                -self.angular_speed_max,
                self.angular_speed_max,
            )
        )
        linear = 0.0
        if self.waiting_scan_creep_enabled and front_clear >= self.waiting_scan_front_clearance:
            linear = float(np.clip(self.waiting_scan_creep_speed, 0.0, self.linear_speed_max))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        self.prev_action_norm = np.array(
            [
                (linear / max(self.linear_speed_max, 1e-3)) * 2.0 - 1.0,
                angular / max(self.angular_speed_max, 1e-3),
            ],
            dtype=np.float32,
        )
        self._publish_rl_path(linear, angular)
        if (self._now_sec() - self._last_debug_sec) > 1.0:
            self._last_debug_sec = self._now_sec()
            self.get_logger().info(
                f"LOCAL map bootstrap | front={front_clear:.2f} lin={linear:.2f} ang={angular:.2f}"
            )

    def _publish_rl_path(self, linear: float, angular: float) -> None:
        """Publish a short base_link-frame local rollout for RViz."""
        if self.rl_path_pub is None:
            return

        steps = max(1, self.rl_path_horizon_steps)
        dt = max(0.02, self.rl_path_dt_sec)
        x = 0.0
        y = 0.0
        yaw = 0.0
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.frame_id = self.rl_path_frame
        path.header.stamp = stamp

        for _ in range(steps + 1):
            pose = PoseStamped()
            pose.header.frame_id = self.rl_path_frame
            pose.header.stamp = stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.03
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path.poses.append(pose)

            next_x = x + linear * math.cos(yaw) * dt
            next_y = y + linear * math.sin(yaw) * dt
            if not self._local_rollout_has_clearance(next_x, next_y):
                break
            x = next_x
            y = next_y
            yaw = math.atan2(math.sin(yaw + angular * dt), math.cos(yaw + angular * dt))

        self.rl_path_pub.publish(path)

    def _local_rollout_has_clearance(self, x: float, y: float) -> bool:
        """Check a base_link-frame rollout point against the current Realsense scan."""
        distance = math.hypot(x, y)
        if distance <= 0.02:
            return True
        angle = math.atan2(y, x)
        ranges = self.raw_scan_ranges if self.raw_scan_ranges.size > 0 else self.latest_scan_ranges
        angles = self.raw_scan_angles if self.raw_scan_angles.size > 0 else self.latest_scan_angles
        if ranges.size == 0 or ranges.size != angles.size:
            return True
        diff = np.abs(np.arctan2(np.sin(angles - angle), np.cos(angles - angle)))
        mask = diff <= math.radians(8.0)
        if not np.any(mask):
            idx = int(np.argmin(diff))
            clearance = float(ranges[idx])
        else:
            clearance = float(np.percentile(ranges[mask], 20.0))
        return (distance + max(0.35, self.obstacle_stop_distance)) <= clearance

    def _publish_stop(self) -> None:
        """Publish zero velocity and clear previous action memory."""
        self.prev_action_norm[:] = 0.0
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
