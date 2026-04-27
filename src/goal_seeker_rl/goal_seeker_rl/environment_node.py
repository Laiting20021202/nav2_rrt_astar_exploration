"""Environment logic for depth-scan-based goal navigation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


@dataclass
class StepResult:
    """Container for one environment evaluation step."""

    state: np.ndarray
    reward: float
    done: bool
    reason: str
    goal_distance: float
    heading_angle: float
    min_obstacle_distance: float


class GoalSeekerEnvironment:
    """Converts ROS sensor streams into RL state and reward signals."""

    def __init__(
        self,
        lidar_samples: int = 24,
        lidar_max_range: float = 3.5,
        max_goal_distance: float = 10.0,
        goal_tolerance: float = 0.20,
        collision_distance: float = 0.13,
        stuck_window_sec: float = 10.0,
        stuck_cell_size: float = 0.10,
        stuck_overlap_threshold: float = 0.70,
        stuck_min_displacement: float = 0.50,
        spin_filter_angular_threshold: float = 0.5,
        spin_filter_min_range: float = 0.15,
        episodic_memory_enabled: bool = True,
        memory_cell_size: float = 0.20,
        memory_novelty_reward: float = 0.40,
        memory_revisit_penalty: float = 0.80,
        memory_revisit_saturation: int = 4,
        dead_end_front_distance: float = 0.45,
        dead_end_side_distance: float = 0.65,
        dead_end_clearance_margin: float = 0.20,
        dead_end_front_angle_deg: float = 30.0,
        dead_end_side_min_angle_deg: float = 45.0,
        dead_end_side_max_angle_deg: float = 140.0,
        dead_end_revisit_gate: float = 0.5,
        goal_reward: float = 100.0,
        collision_penalty: float = -100.0,
        stuck_penalty: float = -30.0,
        progress_reward_scale: float = 10.0,
        forward_reward_scale: float = 0.5,
        angular_penalty_scale: float = 0.5,
        obstacle_penalty_scale: float = 0.5,
        time_penalty: float = 0.01,
    ) -> None:
        self.lidar_samples = lidar_samples
        self.lidar_max_range = lidar_max_range
        self.max_goal_distance = max_goal_distance
        self.goal_tolerance = goal_tolerance
        self.collision_distance = collision_distance

        self.stuck_window_sec = stuck_window_sec
        self.stuck_cell_size = stuck_cell_size
        self.stuck_overlap_threshold = stuck_overlap_threshold
        self.stuck_min_displacement = max(0.01, stuck_min_displacement)

        # FAR Planner-inspired anti-noise scan filtering.
        self.spin_filter_angular_threshold = spin_filter_angular_threshold
        self.spin_filter_min_range = spin_filter_min_range
        self.episodic_memory_enabled = episodic_memory_enabled
        self.memory_cell_size = max(0.05, memory_cell_size)
        self.memory_novelty_reward = memory_novelty_reward
        self.memory_revisit_penalty = memory_revisit_penalty
        self.memory_revisit_saturation = max(1, memory_revisit_saturation)
        self.dead_end_front_distance = dead_end_front_distance
        self.dead_end_side_distance = dead_end_side_distance
        self.dead_end_clearance_margin = dead_end_clearance_margin
        self.dead_end_front_angle_rad = math.radians(dead_end_front_angle_deg)
        self.dead_end_side_min_angle_rad = math.radians(dead_end_side_min_angle_deg)
        self.dead_end_side_max_angle_rad = math.radians(dead_end_side_max_angle_deg)
        self.dead_end_revisit_gate = dead_end_revisit_gate
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.stuck_penalty = stuck_penalty
        self.progress_reward_scale = progress_reward_scale
        self.forward_reward_scale = forward_reward_scale
        self.angular_penalty_scale = angular_penalty_scale
        self.obstacle_penalty_scale = obstacle_penalty_scale
        self.time_penalty = time_penalty

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.angular_velocity_z = 0.0
        self.has_odom = False
        self.has_scan = False

        self.goal_xy: Optional[Tuple[float, float]] = None
        self.previous_goal_distance: Optional[float] = None

        self.scan_norm = np.ones(self.lidar_samples, dtype=np.float32)
        self.scan_sampled = np.full(self.lidar_samples, self.lidar_max_range, dtype=np.float32)
        self.scan_angles = np.linspace(-math.pi, math.pi, self.lidar_samples, dtype=np.float32)
        self.min_obstacle_distance = self.lidar_max_range

        self.position_history: Deque[Tuple[float, float, float]] = deque()
        self.last_overlap_ratio = 0.0
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.last_memory_cell: Optional[Tuple[int, int]] = None
        self.last_revisit_ratio = 0.0
        self.dead_end_detected = False
        self.dead_end_turn_hint = 0.0
        self.dead_end_intensity = 0.0

    @property
    def state_dim(self) -> int:
        """Return size of state vector: scan samples + distance + heading."""
        return self.lidar_samples + 2

    def set_goal(self, x: float, y: float, now_sec: float) -> None:
        """Set a new goal and reset per-episode internal state."""
        self.goal_xy = (x, y)
        self.previous_goal_distance = None
        self.position_history.clear()
        self.position_history.append((now_sec, self.robot_x, self.robot_y))
        self.last_overlap_ratio = 0.0
        self._reset_episodic_memory()

    def clear_goal(self) -> None:
        """Clear active goal."""
        self.goal_xy = None
        self.previous_goal_distance = None
        self._reset_episodic_memory()

    def ready_for_control(self) -> bool:
        """Check whether sensor data and goal are available."""
        return self.has_odom and self.has_scan and self.goal_xy is not None

    def register_reset(self, now_sec: float) -> None:
        """Reset episode-only memory after simulation reset."""
        self.previous_goal_distance = None
        self.position_history.clear()
        self.position_history.append((now_sec, self.robot_x, self.robot_y))
        self.last_overlap_ratio = 0.0
        self._reset_episodic_memory()

    def update_odom(self, msg: Odometry, now_sec: float) -> None:
        """Update robot pose and motion state from odometry."""
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        self.robot_yaw = self._yaw_from_quaternion(
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        )
        self.angular_velocity_z = float(msg.twist.twist.angular.z)
        self.has_odom = True

        self.position_history.append((now_sec, self.robot_x, self.robot_y))
        self._trim_history(now_sec)

    def update_scan(self, msg: LaserScan) -> None:
        """Update normalized scan state and minimum obstacle distance."""
        if not msg.ranges:
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid_fill = min(float(msg.range_max), self.lidar_max_range)
        ranges = np.nan_to_num(ranges, nan=valid_fill, posinf=valid_fill, neginf=valid_fill)
        ranges = np.clip(ranges, float(msg.range_min), self.lidar_max_range)

        if abs(self.angular_velocity_z) > self.spin_filter_angular_threshold:
            # Ignore very short rays while spinning to reduce floor artifacts.
            ranges = np.where(ranges < self.spin_filter_min_range, self.lidar_max_range, ranges)

        if len(ranges) >= self.lidar_samples and (len(ranges) % self.lidar_samples == 0):
            # Match native scanner bins when divisible (e.g. 360 -> 40 uses stride=9).
            stride = len(ranges) // self.lidar_samples
            idx = np.arange(0, stride * self.lidar_samples, stride, dtype=np.int32)
        else:
            idx = np.linspace(0, len(ranges) - 1, self.lidar_samples, dtype=np.int32)
        sampled = ranges[idx]
        sampled_angles = float(msg.angle_min) + idx.astype(np.float32) * float(msg.angle_increment)

        self.scan_norm = np.clip(sampled / self.lidar_max_range, 0.0, 1.0).astype(np.float32)
        self.scan_sampled = sampled.astype(np.float32)
        self.scan_angles = sampled_angles.astype(np.float32)
        self.min_obstacle_distance = float(np.min(sampled))
        self._update_dead_end_signal()
        self.has_scan = True

    def compute_state(self) -> np.ndarray:
        """Build state vector from range scan + relative goal polar coordinates."""
        goal_distance, heading = self._goal_metrics()
        distance_norm = float(np.clip(goal_distance / self.max_goal_distance, 0.0, 1.0))
        heading_norm = float(np.clip(heading / math.pi, -1.0, 1.0))
        return np.concatenate([self.scan_norm, np.array([distance_norm, heading_norm], dtype=np.float32)])

    def evaluate_step(self, now_sec: float, previous_action: Optional[np.ndarray] = None) -> StepResult:
        """Evaluate reward and termination flags for the current observation."""
        self._trim_history(now_sec)
        state = self.compute_state()
        goal_distance, heading = self._goal_metrics()

        if self.goal_xy is None:
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                reason="idle",
                goal_distance=goal_distance,
                heading_angle=heading,
                min_obstacle_distance=self.min_obstacle_distance,
            )

        reward = 0.0
        done = False
        reason = "running"
        cell_visits = self._update_visit_memory()
        if self.episodic_memory_enabled:
            self.last_revisit_ratio = float(np.clip((cell_visits - 1) / self.memory_revisit_saturation, 0.0, 1.0))
        else:
            self.last_revisit_ratio = 0.0

        if self.previous_goal_distance is not None:
            reward += self.progress_reward_scale * (self.previous_goal_distance - goal_distance)

        if goal_distance <= self.goal_tolerance:
            reward += self.goal_reward
            done = True
            reason = "goal"
        elif self.min_obstacle_distance <= self.collision_distance:
            reward += self.collision_penalty
            done = True
            reason = "collision"
        else:
            reward -= self.time_penalty
            if previous_action is not None:
                action = np.asarray(previous_action, dtype=np.float32)
                if action.size >= 2:
                    linear_norm = float(np.clip((float(action[0]) + 1.0) * 0.5, 0.0, 1.0))
                    angular_norm = float(np.clip(abs(float(action[1])), 0.0, 1.0))
                    near_obstacle = max(0.0, 1.0 - self.min_obstacle_distance)
                    reward += self.forward_reward_scale * linear_norm
                    reward -= self.angular_penalty_scale * angular_norm
                    reward -= self.obstacle_penalty_scale * near_obstacle

            self.last_overlap_ratio = self._compute_overlap_ratio()
            if self.last_overlap_ratio >= self.stuck_overlap_threshold:
                reward += self.stuck_penalty
                done = True
                reason = "stuck"
            elif self.episodic_memory_enabled:
                novelty = 1.0 / math.sqrt(float(cell_visits))
                reward += self.memory_novelty_reward * novelty
                reward -= self.memory_revisit_penalty * self.last_revisit_ratio

        self.previous_goal_distance = goal_distance

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            reason=reason,
            goal_distance=goal_distance,
            heading_angle=heading,
            min_obstacle_distance=self.min_obstacle_distance,
        )

    def get_escape_signal(self) -> tuple[bool, float, float]:
        """Return dead-end detection signal for optional action override."""
        return self.dead_end_detected, self.dead_end_turn_hint, self.dead_end_intensity

    def _reset_episodic_memory(self) -> None:
        """Clear per-episode location memory and dead-end state."""
        self.visit_counts.clear()
        self.last_memory_cell = None
        self.last_revisit_ratio = 0.0
        self.dead_end_detected = False
        self.dead_end_turn_hint = 0.0
        self.dead_end_intensity = 0.0

    def _grid_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert world position to discretized memory cell id."""
        return (
            int(round(x / self.memory_cell_size)),
            int(round(y / self.memory_cell_size)),
        )

    def _update_visit_memory(self) -> int:
        """Update visit count for current grid cell and return current cell count."""
        if not self.episodic_memory_enabled:
            self.last_revisit_ratio = 0.0
            return 1

        current_cell = self._grid_cell(self.robot_x, self.robot_y)
        if current_cell != self.last_memory_cell:
            self.visit_counts[current_cell] = self.visit_counts.get(current_cell, 0) + 1
            self.last_memory_cell = current_cell
        return self.visit_counts.get(current_cell, 1)

    def get_visit_count(self, x: float, y: float) -> int:
        """Return episodic visit count for an arbitrary world position."""
        if not self.episodic_memory_enabled:
            return 0
        cell = self._grid_cell(x, y)
        return int(self.visit_counts.get(cell, 0))

    def _update_dead_end_signal(self) -> None:
        """Estimate dead-end presence and suggested turn direction from LiDAR sectors."""
        front_mask = np.abs(self.scan_angles) <= self.dead_end_front_angle_rad
        left_mask = (self.scan_angles >= self.dead_end_side_min_angle_rad) & (
            self.scan_angles <= self.dead_end_side_max_angle_rad
        )
        right_mask = (self.scan_angles <= -self.dead_end_side_min_angle_rad) & (
            self.scan_angles >= -self.dead_end_side_max_angle_rad
        )
        if (not np.any(front_mask)) or (not np.any(left_mask)) or (not np.any(right_mask)):
            self.dead_end_detected = False
            self.dead_end_turn_hint = 0.0
            self.dead_end_intensity = 0.0
            return

        front_clear = float(np.percentile(self.scan_sampled[front_mask], 80))
        left_clear = float(np.percentile(self.scan_sampled[left_mask], 80))
        right_clear = float(np.percentile(self.scan_sampled[right_mask], 80))
        side_best = max(left_clear, right_clear)
        turn_hint = 1.0 if left_clear >= right_clear else -1.0
        revisit_escape = self.last_revisit_ratio >= self.dead_end_revisit_gate

        dead_end = (
            front_clear < self.dead_end_front_distance
            and side_best > (front_clear + self.dead_end_clearance_margin)
            and side_best > self.dead_end_side_distance
        )
        if (not dead_end) and revisit_escape and front_clear < (self.dead_end_front_distance + 0.15):
            dead_end = True

        if dead_end:
            intensity = float(
                np.clip((self.dead_end_front_distance - front_clear) / max(0.01, self.dead_end_front_distance), 0.0, 1.0)
            )
            self.dead_end_detected = True
            self.dead_end_turn_hint = turn_hint
            self.dead_end_intensity = intensity
        else:
            self.dead_end_detected = False
            self.dead_end_turn_hint = 0.0
            self.dead_end_intensity = 0.0

    def _goal_metrics(self) -> tuple[float, float]:
        """Return distance and relative heading to active goal."""
        if self.goal_xy is None:
            return self.max_goal_distance, 0.0

        dx = self.goal_xy[0] - self.robot_x
        dy = self.goal_xy[1] - self.robot_y
        distance = math.sqrt(dx * dx + dy * dy)
        heading = math.atan2(dy, dx) - self.robot_yaw

        while heading > math.pi:
            heading -= 2.0 * math.pi
        while heading < -math.pi:
            heading += 2.0 * math.pi
        return distance, heading

    def _trim_history(self, now_sec: float) -> None:
        """Keep only recent position samples within the configured time window."""
        min_time = now_sec - self.stuck_window_sec
        while self.position_history and self.position_history[0][0] < min_time:
            self.position_history.popleft()

    def _compute_overlap_ratio(self) -> float:
        """Compute overlap ratio of recent trajectory positions."""
        if len(self.position_history) < 10:
            return 0.0

        history_duration = self.position_history[-1][0] - self.position_history[0][0]
        if history_duration < self.stuck_window_sec * 0.7:
            return 0.0

        start = self.position_history[0]
        end = self.position_history[-1]
        net_displacement = math.hypot(end[1] - start[1], end[2] - start[2])
        if net_displacement >= self.stuck_min_displacement:
            # The robot has made meaningful progress in the recent window.
            return 0.0

        cells = [
            (
                int(round(x / self.stuck_cell_size)),
                int(round(y / self.stuck_cell_size)),
            )
            for (_, x, y) in self.position_history
        ]
        # Remove consecutive duplicates so slow motion inside one grid cell does not
        # look like pathological looping.
        compressed_cells = [cells[0]]
        for cell in cells[1:]:
            if cell != compressed_cells[-1]:
                compressed_cells.append(cell)

        if len(compressed_cells) < 6:
            return 0.0

        unique_cells = len(set(compressed_cells))
        return 1.0 - (unique_cells / float(len(compressed_cells)))

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle in radians."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
