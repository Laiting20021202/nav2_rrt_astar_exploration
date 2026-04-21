#!/usr/bin/env python3
from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .exploration_memory import ExplorationMemory
from .exploration_types import FrontierCandidate, NavigationTarget, Pose2D
from .frontier_extractor import FrontierExtractor
from .frontier_scoring import FrontierScorer
from .map_utils import (
    angle_wrap,
    astar_path,
    build_inflated_obstacle_mask,
    euclidean,
    grid_index,
    grid_to_world,
    in_bounds,
    is_unknown,
    known_cell_count,
    line_collision_free,
    nearest_free_cell,
    neighbors4,
    path_length_m,
    world_to_grid,
)
from .optional_learned_ranker import OptionalLearnedRanker


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


@dataclass
class MazeCellMemory:
    visited: bool = False
    walls: Dict[str, bool] = field(
        default_factory=lambda: {"N": False, "E": False, "S": False, "W": False}
    )
    dead_end: bool = False
    has_unexplored_neighbor: bool = False
    last_seen_sec: float = 0.0


class ExplorationCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("exploration_coordinator")

        # Topics / frames
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("legacy_goal_topic", "/mapless_goal")
        self.declare_parameter("clicked_goal_topic", "/clicked_point")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("global_frame", "map")

        self.declare_parameter("frontier_markers_topic", "/exploration/frontier_markers")
        self.declare_parameter("candidate_markers_topic", "/exploration/candidate_markers")
        self.declare_parameter("selected_marker_topic", "/exploration/selected_target")
        self.declare_parameter("final_goal_marker_topic", "/exploration/final_goal_marker")
        self.declare_parameter("robot_pose_marker_topic", "/exploration/robot_pose_marker")
        self.declare_parameter("rrt_tree_topic", "/exploration/rrt_tree")
        self.declare_parameter("selected_path_topic", "/exploration/selected_path")
        self.declare_parameter("state_topic", "/exploration/state")

        # Core timings
        self.declare_parameter("loop_rate_hz", 1.5)
        self.declare_parameter("replan_period_sec", 2.0)
        self.declare_parameter("goal_reachability_recheck_sec", 2.0)
        self.declare_parameter("goal_tolerance_m", 0.35)
        self.declare_parameter("progress_min_delta_m", 0.12)

        # Grid / planning behavior
        self.declare_parameter("free_threshold", 15)
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("inflation_radius_m", 0.20)
        self.declare_parameter("dynamic_inflation_enabled", True)
        self.declare_parameter("dynamic_inflation_goal_dist_m", 2.0)
        self.declare_parameter("dynamic_inflation_scale", 0.6)
        self.declare_parameter("dynamic_inflation_min_radius_m", 0.06)
        self.declare_parameter("slim_inflation_fallback_enabled", True)
        self.declare_parameter("goal_snap_radius_cells", 12)
        self.declare_parameter("astar_revisit_cost_scale", 0.06)
        self.declare_parameter("decision_persistence_sec", 3.0)
        self.declare_parameter("optimistic_unknown_cost_scale", 0.45)

        self.declare_parameter("maze_mode", True)
        self.declare_parameter("maze_cell_size_m", 0.40)
        self.declare_parameter("maze_search_depth_limit", 80)
        self.declare_parameter("maze_dead_end_penalty", 2.0)

        # Sensor noise stabilization (pitch-induced lidar artifacts)
        self.declare_parameter("noise_filter_enabled", True)
        self.declare_parameter("noise_filter_transient_ttl_sec", 1.2)
        self.declare_parameter("noise_filter_confirm_hits", 3)
        self.declare_parameter("noise_filter_isolated_neighbor_max", 1)
        self.declare_parameter("noise_filter_soft_cost_scale", 0.9)

        # Frontier filtering
        self.declare_parameter("frontier_filter_min_distance", 0.6)
        self.declare_parameter("frontier_filter_max_distance", 14.0)
        self.declare_parameter("max_scored_candidates", 30)

        # Recovery / failure
        self.declare_parameter("recovery_cooldown_sec", 3.0)
        self.declare_parameter("max_frontier_failures_total", 80)

        # Scoring weights
        self.declare_parameter("w_info", 1.4)
        self.declare_parameter("w_cost", 1.15)
        self.declare_parameter("w_visit", 0.7)
        self.declare_parameter("w_fail", 1.0)
        self.declare_parameter("w_turn", 0.35)
        self.declare_parameter("w_goal", 0.5)
        self.declare_parameter("w_commit", 0.45)
        self.declare_parameter("w_clearance", 0.45)
        self.declare_parameter("goal_gravity_range_m", 6.0)
        self.declare_parameter("goal_gravity_exp_gain", 1.2)
        self.declare_parameter("goal_gravity_max_multiplier", 3.5)
        self.declare_parameter("info_near_goal_min_scale", 0.35)

        # Optional neural module
        self.declare_parameter("learned_ranker_enabled", False)
        self.declare_parameter("learned_ranker_bias_gain", 0.05)

        # RRT exploration parameters
        self.declare_parameter("frontier_min_cluster_size", 10)
        self.declare_parameter("frontier_info_gain_radius_m", 1.1)
        self.declare_parameter("rrt_iterations", 500)
        self.declare_parameter("rrt_step_m", 0.25)
        self.declare_parameter("rrt_goal_bias", 0.35)
        self.declare_parameter("rrt_frontier_proximity_m", 0.6)
        self.declare_parameter("rrt_goal_snap_radius_cells", 8)
        self.declare_parameter("candidate_min_separation_m", 0.45)
        self.declare_parameter("candidate_max_per_frontier", 3)
        self.declare_parameter("robot_candidate_min_distance_m", 0.7)
        self.declare_parameter("candidate_inward_offset_m", 0.18)
        self.declare_parameter("candidate_snap_radius_cells", 6)
        self.declare_parameter("candidate_clearance_max_m", 1.2)

        # Memory / anti-wandering
        self.declare_parameter("visited_decay_tau", 120.0)
        self.declare_parameter("visited_increment", 1.0)
        self.declare_parameter("visited_prune_threshold", 0.03)
        self.declare_parameter("frontier_cooldown_sec", 12.0)
        self.declare_parameter("frontier_blacklist_base_sec", 25.0)
        self.declare_parameter("frontier_blacklist_max_sec", 240.0)
        self.declare_parameter("frontier_fail_blacklist_threshold", 2)
        self.declare_parameter("commitment_horizon_sec", 10.0)
        self.declare_parameter("stuck_window_sec", 12.0)
        self.declare_parameter("stuck_radius_m", 0.22)
        self.declare_parameter("oscillation_window_sec", 10.0)
        self.declare_parameter("oscillation_cell_size_m", 0.20)
        self.declare_parameter("oscillation_toggle_threshold", 4)
        self.declare_parameter("stagnation_window_sec", 20.0)
        self.declare_parameter("stagnation_min_known_delta", 40)
        self.declare_parameter("stagnation_min_travel_m", 0.60)

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.clicked_goal_topic = str(self.get_parameter("clicked_goal_topic").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.global_frame = str(self.get_parameter("global_frame").value)

        self.frontier_markers_topic = str(self.get_parameter("frontier_markers_topic").value)
        self.candidate_markers_topic = str(self.get_parameter("candidate_markers_topic").value)
        self.selected_marker_topic = str(self.get_parameter("selected_marker_topic").value)
        self.final_goal_marker_topic = str(self.get_parameter("final_goal_marker_topic").value)
        self.robot_pose_marker_topic = str(self.get_parameter("robot_pose_marker_topic").value)
        self.rrt_tree_topic = str(self.get_parameter("rrt_tree_topic").value)
        self.selected_path_topic = str(self.get_parameter("selected_path_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.loop_rate_hz = float(self.get_parameter("loop_rate_hz").value)
        self.replan_period_sec = float(self.get_parameter("replan_period_sec").value)
        self.goal_reachability_recheck_sec = float(self.get_parameter("goal_reachability_recheck_sec").value)
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self.progress_min_delta_m = float(self.get_parameter("progress_min_delta_m").value)

        self.free_threshold = int(self.get_parameter("free_threshold").value)
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.inflation_radius_m = float(self.get_parameter("inflation_radius_m").value)
        self.dynamic_inflation_enabled = bool(self.get_parameter("dynamic_inflation_enabled").value)
        self.dynamic_inflation_goal_dist_m = float(self.get_parameter("dynamic_inflation_goal_dist_m").value)
        self.dynamic_inflation_scale = float(self.get_parameter("dynamic_inflation_scale").value)
        self.dynamic_inflation_min_radius_m = float(self.get_parameter("dynamic_inflation_min_radius_m").value)
        self.slim_inflation_fallback_enabled = bool(self.get_parameter("slim_inflation_fallback_enabled").value)
        self.goal_snap_radius_cells = int(self.get_parameter("goal_snap_radius_cells").value)
        self.astar_revisit_cost_scale = float(self.get_parameter("astar_revisit_cost_scale").value)
        self.decision_persistence_sec = float(self.get_parameter("decision_persistence_sec").value)
        self.optimistic_unknown_cost_scale = float(self.get_parameter("optimistic_unknown_cost_scale").value)

        self.maze_mode = bool(self.get_parameter("maze_mode").value)
        self.maze_cell_size_m = float(self.get_parameter("maze_cell_size_m").value)
        self.maze_search_depth_limit = int(self.get_parameter("maze_search_depth_limit").value)
        self.maze_dead_end_penalty = float(self.get_parameter("maze_dead_end_penalty").value)

        self.noise_filter_enabled = bool(self.get_parameter("noise_filter_enabled").value)
        self.noise_filter_transient_ttl_sec = float(self.get_parameter("noise_filter_transient_ttl_sec").value)
        self.noise_filter_confirm_hits = int(self.get_parameter("noise_filter_confirm_hits").value)
        self.noise_filter_isolated_neighbor_max = int(self.get_parameter("noise_filter_isolated_neighbor_max").value)
        self.noise_filter_soft_cost_scale = float(self.get_parameter("noise_filter_soft_cost_scale").value)

        self.frontier_filter_min_distance = float(self.get_parameter("frontier_filter_min_distance").value)
        self.frontier_filter_max_distance = float(self.get_parameter("frontier_filter_max_distance").value)
        self.max_scored_candidates = int(self.get_parameter("max_scored_candidates").value)
        self.candidate_max_per_frontier = int(self.get_parameter("candidate_max_per_frontier").value)

        self.goal_gravity_range_m = float(self.get_parameter("goal_gravity_range_m").value)
        self.goal_gravity_exp_gain = float(self.get_parameter("goal_gravity_exp_gain").value)
        self.goal_gravity_max_multiplier = float(self.get_parameter("goal_gravity_max_multiplier").value)
        self.info_near_goal_min_scale = float(self.get_parameter("info_near_goal_min_scale").value)

        self.recovery_cooldown_sec = float(self.get_parameter("recovery_cooldown_sec").value)
        self.max_frontier_failures_total = int(self.get_parameter("max_frontier_failures_total").value)

        self.rng = random.Random(42)

        extractor_params = {
            "free_threshold": self.free_threshold,
            "occupied_threshold": self.occupied_threshold,
            "frontier_min_cluster_size": int(self.get_parameter("frontier_min_cluster_size").value),
            "frontier_info_gain_radius_m": float(self.get_parameter("frontier_info_gain_radius_m").value),
            "frontier_filter_min_distance": self.frontier_filter_min_distance,
            "frontier_filter_max_distance": self.frontier_filter_max_distance,
            "rrt_iterations": int(self.get_parameter("rrt_iterations").value),
            "rrt_step_m": float(self.get_parameter("rrt_step_m").value),
            "rrt_goal_bias": float(self.get_parameter("rrt_goal_bias").value),
            "rrt_frontier_proximity_m": float(self.get_parameter("rrt_frontier_proximity_m").value),
            "rrt_goal_snap_radius_cells": int(self.get_parameter("rrt_goal_snap_radius_cells").value),
            "candidate_min_separation_m": float(self.get_parameter("candidate_min_separation_m").value),
            "candidate_max_per_frontier": int(self.get_parameter("candidate_max_per_frontier").value),
            "robot_candidate_min_distance_m": float(self.get_parameter("robot_candidate_min_distance_m").value),
            "candidate_inward_offset_m": float(self.get_parameter("candidate_inward_offset_m").value),
            "candidate_snap_radius_cells": int(self.get_parameter("candidate_snap_radius_cells").value),
            "candidate_clearance_max_m": float(self.get_parameter("candidate_clearance_max_m").value),
        }
        self.extractor = FrontierExtractor(extractor_params)

        memory_params = {
            "visited_decay_tau": float(self.get_parameter("visited_decay_tau").value),
            "visited_increment": float(self.get_parameter("visited_increment").value),
            "visited_prune_threshold": float(self.get_parameter("visited_prune_threshold").value),
            "frontier_cooldown_sec": float(self.get_parameter("frontier_cooldown_sec").value),
            "frontier_blacklist_base_sec": float(self.get_parameter("frontier_blacklist_base_sec").value),
            "frontier_blacklist_max_sec": float(self.get_parameter("frontier_blacklist_max_sec").value),
            "frontier_fail_blacklist_threshold": int(self.get_parameter("frontier_fail_blacklist_threshold").value),
            "commitment_horizon_sec": float(self.get_parameter("commitment_horizon_sec").value),
            "stuck_window_sec": float(self.get_parameter("stuck_window_sec").value),
            "stuck_radius_m": float(self.get_parameter("stuck_radius_m").value),
            "oscillation_window_sec": float(self.get_parameter("oscillation_window_sec").value),
            "oscillation_cell_size_m": float(self.get_parameter("oscillation_cell_size_m").value),
            "oscillation_toggle_threshold": int(self.get_parameter("oscillation_toggle_threshold").value),
            "stagnation_window_sec": float(self.get_parameter("stagnation_window_sec").value),
            "stagnation_min_known_delta": int(self.get_parameter("stagnation_min_known_delta").value),
            "stagnation_min_travel_m": float(self.get_parameter("stagnation_min_travel_m").value),
        }
        self.memory = ExplorationMemory(memory_params)

        scoring_params = {
            "w_info": float(self.get_parameter("w_info").value),
            "w_cost": float(self.get_parameter("w_cost").value),
            "w_visit": float(self.get_parameter("w_visit").value),
            "w_fail": float(self.get_parameter("w_fail").value),
            "w_turn": float(self.get_parameter("w_turn").value),
            "w_goal": float(self.get_parameter("w_goal").value),
            "w_commit": float(self.get_parameter("w_commit").value),
            "w_clearance": float(self.get_parameter("w_clearance").value),
            "goal_gravity_range_m": self.goal_gravity_range_m,
            "goal_gravity_exp_gain": self.goal_gravity_exp_gain,
            "goal_gravity_max_multiplier": self.goal_gravity_max_multiplier,
            "info_near_goal_min_scale": self.info_near_goal_min_scale,
        }
        self.scorer = FrontierScorer(scoring_params)

        learned_params = {
            "enabled": bool(self.get_parameter("learned_ranker_enabled").value),
            "bias_gain": float(self.get_parameter("learned_ranker_bias_gain").value),
        }
        self.learned_ranker = OptionalLearnedRanker(learned_params)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.clear_local_client = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        self.clear_global_client = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        if self.legacy_goal_topic != self.goal_topic:
            self.legacy_goal_sub = self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_callback, 10)
        else:
            self.legacy_goal_sub = None
        if self.clicked_goal_topic not in (self.goal_topic, self.legacy_goal_topic):
            self.clicked_goal_sub = self.create_subscription(
                PointStamped,
                self.clicked_goal_topic,
                self.clicked_point_callback,
                10,
            )
        else:
            self.clicked_goal_sub = None

        self.frontier_pub = self.create_publisher(MarkerArray, self.frontier_markers_topic, 10)
        self.candidate_pub = self.create_publisher(MarkerArray, self.candidate_markers_topic, 10)
        self.selected_pub = self.create_publisher(Marker, self.selected_marker_topic, 10)
        self.final_goal_pub = self.create_publisher(Marker, self.final_goal_marker_topic, 10)
        self.robot_pose_pub = self.create_publisher(Marker, self.robot_pose_marker_topic, 10)
        self.rrt_tree_pub = self.create_publisher(Marker, self.rrt_tree_topic, 10)
        self.selected_path_pub = self.create_publisher(Path, self.selected_path_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.timer = self.create_timer(1.0 / max(0.1, self.loop_rate_hz), self.timer_callback)

        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_inflated_mask: List[bool] = []
        self.latest_relaxed_inflated_mask: List[bool] = []
        self.latest_goal_snap_cell: Optional[Tuple[int, int]] = None
        self.latest_soft_obstacle_cells: Dict[Tuple[int, int], float] = {}
        self.occupied_history: Dict[Tuple[int, int], Tuple[int, float]] = {}
        self.maze_memory: Dict[Tuple[int, int], MazeCellMemory] = {}
        self.map_signature: Optional[Tuple[int, int, float]] = None
        self.latest_known_count = 0

        self.final_goal: Optional[PoseStamped] = None
        self.final_goal_reachable_cache = False

        self.active_target: Optional[NavigationTarget] = None
        self.active_goal_handle = None
        self.current_goal_token = 0
        self.ignored_cancel_tokens = set()

        self.mode = "IDLE"
        self.recovering_until = 0.0
        self.last_plan_stamp = 0.0
        self.last_goal_reachability_stamp = 0.0
        self.last_feedback_remaining = float("inf")
        self.total_frontier_failures = 0
        self.target_lock_until = 0.0

        self.get_logger().info("ExplorationCoordinator started. frontier memory + scoring + Nav2 switching active.")

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self.latest_known_count = known_cell_count(msg)

        width = int(msg.info.width)
        height = int(msg.info.height)
        signature = (width, height, float(msg.info.resolution))
        now_sec = self.now_sec()
        if self.map_signature != signature:
            self.map_signature = signature
            self.occupied_history.clear()
            self.latest_soft_obstacle_cells = {}
            self.maze_memory.clear()

        soft_obstacle_cells: Dict[Tuple[int, int], float] = {}
        if self.noise_filter_enabled:
            current_occupied = set()
            for y in range(height):
                base = y * width
                for x in range(width):
                    idx = base + x
                    if int(msg.data[idx]) >= self.occupied_threshold:
                        current_occupied.add((x, y))

            ttl = max(0.05, self.noise_filter_transient_ttl_sec)
            cutoff = now_sec - ttl
            stale = [cell for cell, (_, stamp) in self.occupied_history.items() if stamp < cutoff]
            for cell in stale:
                del self.occupied_history[cell]

            confirm_hits = max(1, self.noise_filter_confirm_hits)
            for cell in current_occupied:
                prev = self.occupied_history.get(cell)
                if prev is not None and (now_sec - prev[1]) <= ttl:
                    hit_count = min(confirm_hits + 2, prev[0] + 1)
                else:
                    hit_count = 1
                self.occupied_history[cell] = (hit_count, now_sec)

            max_neighbors = max(0, self.noise_filter_isolated_neighbor_max)
            for cell in current_occupied:
                hit_count, _ = self.occupied_history.get(cell, (1, now_sec))
                if hit_count >= confirm_hits:
                    continue

                x, y = cell
                occupied_neighbors = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        if (x + dx, y + dy) in current_occupied:
                            occupied_neighbors += 1
                if occupied_neighbors <= max_neighbors:
                    # Lower confidence -> lower traversal penalty.
                    soft_weight = 1.0 - (hit_count / float(confirm_hits))
                    soft_obstacle_cells[cell] = max(0.15, min(1.0, soft_weight))
        else:
            self.occupied_history.clear()

        self.latest_soft_obstacle_cells = soft_obstacle_cells
        ignored_cells = set(soft_obstacle_cells.keys())
        self.latest_inflated_mask = build_inflated_obstacle_mask(
            msg,
            occupied_threshold=self.occupied_threshold,
            inflation_radius_m=self.inflation_radius_m,
            ignore_occupied_cells=ignored_cells,
        )

        relaxed_radius_m = max(
            self.dynamic_inflation_min_radius_m,
            self.inflation_radius_m * self.dynamic_inflation_scale,
        )
        if (not self.dynamic_inflation_enabled) or abs(relaxed_radius_m - self.inflation_radius_m) < 1e-4:
            self.latest_relaxed_inflated_mask = self.latest_inflated_mask
        else:
            self.latest_relaxed_inflated_mask = build_inflated_obstacle_mask(
                msg,
                occupied_threshold=self.occupied_threshold,
                inflation_radius_m=relaxed_radius_m,
                ignore_occupied_cells=ignored_cells,
            )

    def goal_callback(self, msg: PoseStamped) -> None:
        transformed = self.transform_pose_to_global(msg)
        if transformed is None:
            src = msg.header.frame_id.strip() if msg.header.frame_id else "<empty>"
            self.get_logger().warn(
                "Ignore goal: cannot transform frame '%s' into '%s' for pose (%.2f, %.2f)."
                % (src, self.global_frame, msg.pose.position.x, msg.pose.position.y)
            )
            return
        self.final_goal = transformed
        self.final_goal_reachable_cache = False
        self.latest_goal_snap_cell = None
        self.last_goal_reachability_stamp = 0.0
        self.publish_final_goal_marker()
        self.get_logger().info(
            "Received final goal at (%.2f, %.2f) in %s"
            % (transformed.pose.position.x, transformed.pose.position.y, self.global_frame)
        )

    def clicked_point_callback(self, msg: PointStamped) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position = msg.point
        pose.pose.orientation.w = 1.0

        transformed = self.transform_pose_to_global(pose)
        if transformed is None:
            src = msg.header.frame_id.strip() if msg.header.frame_id else "<empty>"
            self.get_logger().warn(
                "Ignore clicked point: cannot transform frame '%s' into '%s' for point (%.2f, %.2f)."
                % (src, self.global_frame, msg.point.x, msg.point.y)
            )
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is not None:
            yaw = math.atan2(
                transformed.pose.position.y - robot_pose.y,
                transformed.pose.position.x - robot_pose.x,
            )
            transformed.pose.orientation = quaternion_from_yaw(yaw)
        else:
            transformed.pose.orientation.w = 1.0

        self.final_goal = transformed
        self.final_goal_reachable_cache = False
        self.latest_goal_snap_cell = None
        self.last_goal_reachability_stamp = 0.0
        self.publish_final_goal_marker()
        self.get_logger().info(
            "Received clicked final goal at (%.2f, %.2f) in %s"
            % (transformed.pose.position.x, transformed.pose.position.y, self.global_frame)
        )

    def timer_callback(self) -> None:
        now_sec = self.now_sec()
        if self.latest_map is None:
            self.publish_state("WAITING_MAP")
            return

        self.publish_final_goal_marker()

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.publish_state("WAITING_TF")
            return

        self.publish_robot_pose_marker(robot_pose)

        robot_cell = world_to_grid(self.latest_map, robot_pose.x, robot_pose.y)
        if robot_cell is None:
            self.publish_state("ROBOT_OUTSIDE_MAP")
            return

        self.memory.update_pose(robot_pose, now_sec, self.latest_known_count)
        self.memory.mark_cell_visited(robot_cell, now_sec, amount=0.25)
        if self.maze_mode:
            self.update_maze_memory(robot_cell, now_sec)

        dist_to_final_goal = float("inf")
        if self.final_goal is not None:
            dist_to_final_goal = euclidean(
                (robot_pose.x, robot_pose.y),
                (self.final_goal.pose.position.x, self.final_goal.pose.position.y),
            )
            if dist_to_final_goal <= self.goal_tolerance_m:
                self.on_final_goal_reached()
                return

        if self.active_target is not None and self.active_target.target_type == "frontier":
            stuck = self.memory.is_stuck(now_sec)
            oscillating = self.memory.is_oscillating(now_sec)
            stagnating = self.memory.is_stagnating(now_sec)
            if stuck or oscillating or stagnating:
                reason = []
                if stuck:
                    reason.append("stuck")
                if oscillating:
                    reason.append("oscillation")
                if stagnating:
                    reason.append("stagnation")
                self.register_frontier_failure(self.active_target.frontier_id, now_sec, "+".join(reason))
                self.trigger_recovery("detector=" + "+".join(reason), now_sec)
                self.publish_state("RECOVERY")
                return

        if now_sec < self.recovering_until:
            self.publish_state("RECOVERY")
            return

        # Check if final goal already reachable from known free space.
        if self.final_goal is not None and (
            (now_sec - self.last_goal_reachability_stamp) > self.goal_reachability_recheck_sec
            or not self.final_goal_reachable_cache
        ):
            reachable, path_cells, _ = self.check_goal_reachable(robot_cell, dist_to_final_goal)
            self.final_goal_reachable_cache = reachable
            self.last_goal_reachability_stamp = now_sec
            if reachable:
                target = self.make_final_goal_target(path_cells)
                self.dispatch_target_if_needed(target, now_sec)
                self.publish_state("FINAL_GOAL_NAV")
                return

        # If final goal was reachable in cache and we still have it, keep navigating toward it.
        if self.final_goal is not None and self.final_goal_reachable_cache:
            target = self.make_final_goal_target([])
            self.dispatch_target_if_needed(target, now_sec)
            self.publish_state("FINAL_GOAL_NAV")
            return

        # Explore mode
        if (now_sec - self.last_plan_stamp) < self.replan_period_sec and self.active_target is not None:
            self.publish_state("EXPLORE_COMMIT")
            return

        selected = self.select_frontier_target(robot_pose, robot_cell, now_sec, dist_to_final_goal)
        if selected is None:
            self.publish_state("NO_FRONTIER")
            if self.final_goal is not None and self.memory.is_stagnating(now_sec):
                self.trigger_recovery("no_frontier_and_stagnating", now_sec)
            return

        self.dispatch_target_if_needed(selected, now_sec)
        self.publish_state("EXPLORING")

    def select_frontier_target(
        self,
        robot_pose: Pose2D,
        robot_cell: Tuple[int, int],
        now_sec: float,
        dist_to_final_goal: float,
    ) -> Optional[NavigationTarget]:
        if self.latest_map is None:
            return None

        if self.maze_mode:
            maze_target = self.select_maze_target(robot_pose, robot_cell, now_sec, dist_to_final_goal)
            if maze_target is not None:
                return maze_target

        clusters = self.extractor.extract_frontier_clusters(self.latest_map, self.latest_inflated_mask)
        self.publish_frontier_markers(clusters)

        if not clusters:
            self.clear_rrt_tree_marker()
            self.clear_candidate_markers()
            return None

        candidates, tree_edges = self.extractor.generate_rrt_candidates(
            self.latest_map,
            robot_cell,
            (robot_pose.x, robot_pose.y),
            clusters,
            self.latest_inflated_mask,
            self.rng,
        )
        self.publish_rrt_tree(tree_edges)

        if not candidates:
            self.clear_candidate_markers()
            return None

        # Respect commitment horizon if active frontier remains reachable.
        if self.active_target is not None and self.active_target.frontier_id is not None:
            af = self.active_target.frontier_id
            if self.memory.commitment_active(af, now_sec):
                for cand in candidates:
                    if cand.frontier_id != af:
                        continue
                    path_cells, path_cost = self.plan_on_known_free(
                        robot_cell,
                        cand.cell,
                        dist_to_goal_m=dist_to_final_goal,
                    )
                    if path_cells:
                        cand.path_cells = path_cells
                        cand.path_cost = path_cost
                        target_yaw = self.compute_target_yaw(robot_pose, cand.world)
                        return NavigationTarget(
                            target_type="frontier",
                            target_id=cand.candidate_id,
                            frontier_id=cand.frontier_id,
                            candidate_id=cand.candidate_id,
                            pose_world=(cand.world[0], cand.world[1], target_yaw),
                            path_cells=path_cells,
                        )

        final_goal_xy = None
        if self.final_goal is not None:
            final_goal_xy = (self.final_goal.pose.position.x, self.final_goal.pose.position.y)

        scored: List[FrontierCandidate] = []
        available_count = 0
        path_ok_count = 0
        for cand in candidates:
            if not self.memory.frontier_available(cand.frontier_id, now_sec):
                continue
            available_count += 1

            path_cells, path_cost = self.plan_on_known_free(
                robot_cell,
                cand.cell,
                dist_to_goal_m=dist_to_final_goal,
            )
            if not path_cells:
                continue
            path_ok_count += 1

            cand.path_cells = path_cells
            cand.path_cost = path_cost
            cand.revisit_penalty = self.memory.revisit_penalty(cand.cell, now_sec)
            cand.failed_penalty = self.memory.frontier_failed_penalty(cand.frontier_id, now_sec)
            cand.heading_penalty = self.scorer.heading_change_penalty(robot_pose, cand.world)
            cand.goal_alignment_bonus = self.scorer.goal_alignment_bonus(robot_pose, cand.world, final_goal_xy)
            cand.commitment_bonus = self.memory.commitment_bonus(cand.frontier_id, now_sec)
            scored.append(cand)

        if not scored:
            fallback_scored = self.score_frontier_boundary_fallback(
                clusters,
                robot_pose,
                robot_cell,
                final_goal_xy,
                now_sec,
                dist_to_final_goal,
            )
            if fallback_scored:
                self.publish_candidate_markers(fallback_scored)
                best = fallback_scored[0]
                target_yaw = self.compute_target_yaw(robot_pose, best.world)
                return NavigationTarget(
                    target_type="frontier",
                    target_id=best.candidate_id,
                    frontier_id=best.frontier_id,
                    candidate_id=best.candidate_id,
                    pose_world=(best.world[0], best.world[1], target_yaw),
                    path_cells=best.path_cells,
                )
            self.get_logger().warn(
                "No scored frontier candidate: total=%d available=%d path_ok=%d"
                % (len(candidates), available_count, path_ok_count)
            )
            self.clear_candidate_markers()
            return None

        scored = self.scorer.score_candidates(scored, robot_pose, final_goal_xy, self.memory, now_sec)
        scored = self.learned_ranker.rerank(scored)
        scored = scored[: max(1, self.max_scored_candidates)]
        self.publish_candidate_markers(scored)

        best = scored[0]
        target_yaw = self.compute_target_yaw(robot_pose, best.world)
        return NavigationTarget(
            target_type="frontier",
            target_id=best.candidate_id,
            frontier_id=best.frontier_id,
            candidate_id=best.candidate_id,
            pose_world=(best.world[0], best.world[1], target_yaw),
            path_cells=best.path_cells,
        )

    def select_maze_target(
        self,
        robot_pose: Pose2D,
        robot_cell: Tuple[int, int],
        now_sec: float,
        dist_to_final_goal: float,
    ) -> Optional[NavigationTarget]:
        current_maze = self.grid_to_maze_cell(robot_cell)
        current_state = self.maze_memory.get(current_maze)
        if current_state is None:
            return None

        choices: List[Tuple[float, NavigationTarget]] = []
        for direction, delta in (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0))):
            if current_state.walls.get(direction, False):
                continue
            maze_cell = (current_maze[0] + delta[0], current_maze[1] + delta[1])
            maze_id = "maze_%d_%d" % maze_cell
            if not self.memory.frontier_available(maze_id, now_sec):
                continue
            neighbor_state = self.maze_memory.get(maze_cell)
            if neighbor_state is not None and neighbor_state.dead_end and not neighbor_state.has_unexplored_neighbor:
                continue
            if neighbor_state is not None and neighbor_state.visited and not neighbor_state.has_unexplored_neighbor:
                continue

            target = self.build_maze_navigation_target(
                maze_cell,
                maze_id,
                robot_pose,
                robot_cell,
                dist_to_final_goal,
            )
            if target is None:
                continue

            score = 0.0
            if neighbor_state is None or not neighbor_state.visited:
                score += 2.0
            if neighbor_state is None or neighbor_state.has_unexplored_neighbor:
                score += 2.0
            if neighbor_state is not None and neighbor_state.dead_end:
                score -= self.maze_dead_end_penalty
            score -= 0.25 * path_length_m(target.path_cells, self.latest_map.info.resolution)
            choices.append((score, target))

        if choices:
            choices.sort(key=lambda item: item[0], reverse=True)
            return choices[0][1]

        bfs_path = self.bfs_to_maze_branch(current_maze)
        if bfs_path:
            maze_cell = bfs_path[-1]
            maze_id = "maze_%d_%d" % maze_cell
            return self.build_maze_navigation_target(
                maze_cell,
                maze_id,
                robot_pose,
                robot_cell,
                dist_to_final_goal,
            )
        return None

    def build_maze_navigation_target(
        self,
        maze_cell: Tuple[int, int],
        maze_id: str,
        robot_pose: Pose2D,
        robot_cell: Tuple[int, int],
        dist_to_final_goal: float,
    ) -> Optional[NavigationTarget]:
        if self.latest_map is None:
            return None

        center_cell = self.maze_cell_center_grid(maze_cell)
        staged = self.extractor.stage_candidate(
            self.latest_map,
            center_cell,
            (robot_pose.x, robot_pose.y),
            self.latest_inflated_mask,
        )
        if staged is None:
            return None
        target_cell, target_world = staged

        strict_path, strict_cost = self.plan_on_known_free(
            robot_cell,
            target_cell,
            dist_to_goal_m=dist_to_final_goal,
            allow_unknown_fallback=False,
        )
        if not strict_path:
            self.plan_on_known_free(
                robot_cell,
                target_cell,
                dist_to_goal_m=dist_to_final_goal,
                allow_unknown_fallback=True,
            )
            return None

        target_yaw = self.compute_target_yaw(robot_pose, target_world)
        return NavigationTarget(
            target_type="frontier",
            target_id=maze_id,
            frontier_id=maze_id,
            candidate_id=maze_id,
            pose_world=(target_world[0], target_world[1], target_yaw),
            path_cells=strict_path,
        )

    def bfs_to_maze_branch(self, start_cell: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        queue = deque([(start_cell, [start_cell])])
        visited = {start_cell}

        while queue:
            cell, path = queue.popleft()
            if len(path) > self.maze_search_depth_limit:
                continue
            state = self.maze_memory.get(cell)
            if state is None:
                continue
            if cell != start_cell and state.has_unexplored_neighbor and not state.dead_end:
                return path

            for direction, delta in (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0))):
                if state.walls.get(direction, False):
                    continue
                nb = (cell[0] + delta[0], cell[1] + delta[1])
                if nb in visited:
                    continue
                nb_state = self.maze_memory.get(nb)
                if nb_state is None:
                    continue
                if nb_state.dead_end and not nb_state.has_unexplored_neighbor:
                    continue
                if not nb_state.visited and not nb_state.has_unexplored_neighbor:
                    continue
                visited.add(nb)
                queue.append((nb, path + [nb]))
        return None

    def update_maze_memory(self, robot_cell: Tuple[int, int], now_sec: float) -> None:
        if self.latest_map is None:
            return
        current_maze = self.grid_to_maze_cell(robot_cell)
        cells_to_update = [current_maze]
        for delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            cells_to_update.append((current_maze[0] + delta[0], current_maze[1] + delta[1]))

        for maze_cell in cells_to_update:
            state = self.maze_memory.setdefault(maze_cell, MazeCellMemory())
            state.last_seen_sec = now_sec
            if maze_cell == current_maze:
                state.visited = True
            state.has_unexplored_neighbor = self.maze_cell_has_unexplored_neighbor(maze_cell)
            state.walls = {
                "N": self.maze_edge_blocked(maze_cell, (0, 1)),
                "E": self.maze_edge_blocked(maze_cell, (1, 0)),
                "S": self.maze_edge_blocked(maze_cell, (0, -1)),
                "W": self.maze_edge_blocked(maze_cell, (-1, 0)),
            }
            open_count = sum(0 if blocked else 1 for blocked in state.walls.values())
            if state.visited and not state.has_unexplored_neighbor and open_count <= 1:
                state.dead_end = True
            elif state.has_unexplored_neighbor:
                state.dead_end = False

    def grid_to_maze_cell(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        stride = self.maze_stride_cells()
        return (cell[0] // stride, cell[1] // stride)

    def maze_cell_center_grid(self, maze_cell: Tuple[int, int]) -> Tuple[int, int]:
        assert self.latest_map is not None
        stride = self.maze_stride_cells()
        width = int(self.latest_map.info.width)
        height = int(self.latest_map.info.height)
        cx = min(width - 1, max(0, maze_cell[0] * stride + stride // 2))
        cy = min(height - 1, max(0, maze_cell[1] * stride + stride // 2))
        return (cx, cy)

    def maze_stride_cells(self) -> int:
        if self.latest_map is None:
            return 1
        return max(1, int(round(self.maze_cell_size_m / max(self.latest_map.info.resolution, 1e-6))))

    def maze_cell_has_unexplored_neighbor(self, maze_cell: Tuple[int, int]) -> bool:
        if self.latest_map is None:
            return False
        width = int(self.latest_map.info.width)
        height = int(self.latest_map.info.height)
        stride = self.maze_stride_cells()
        center = self.maze_cell_center_grid(maze_cell)
        radius = max(1, stride // 2 + 1)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell = (center[0] + dx, center[1] + dy)
                if not in_bounds(width, height, cell):
                    continue
                if is_unknown(int(self.latest_map.data[grid_index(width, cell)])):
                    return True
        return False

    def maze_edge_blocked(self, maze_cell: Tuple[int, int], delta: Tuple[int, int]) -> bool:
        if self.latest_map is None:
            return True
        width = int(self.latest_map.info.width)
        height = int(self.latest_map.info.height)
        stride = self.maze_stride_cells()
        start = self.maze_cell_center_grid(maze_cell)
        target_cell = (maze_cell[0] + delta[0], maze_cell[1] + delta[1])
        end = (target_cell[0] * stride + stride // 2, target_cell[1] * stride + stride // 2)
        if not in_bounds(width, height, start) or not in_bounds(width, height, end):
            return True
        return not line_collision_free(
            width,
            height,
            self.latest_inflated_mask,
            self.latest_map.data,
            start,
            end,
            self.free_threshold,
            allow_unknown=True,
        )

    def score_frontier_boundary_fallback(
        self,
        clusters,
        robot_pose: Pose2D,
        robot_cell: Tuple[int, int],
        final_goal_xy: Optional[Tuple[float, float]],
        now_sec: float,
        dist_to_final_goal: float,
    ) -> List[FrontierCandidate]:
        if self.latest_map is None:
            return []

        scored: List[FrontierCandidate] = []
        max_per_frontier = max(1, min(4, self.candidate_max_per_frontier))
        min_dist = max(0.2, 0.5 * self.frontier_filter_min_distance)

        for cluster in clusters:
            if not self.memory.frontier_available(cluster.cluster_id, now_sec):
                continue
            boundary = list(cluster.boundary_cells)
            if not boundary:
                continue
            boundary.sort(
                key=lambda c: euclidean(
                    (robot_pose.x, robot_pose.y),
                    grid_to_world(self.latest_map, c),
                ),
                reverse=True,
            )

            accepted = 0
            for idx, cell in enumerate(boundary):
                if accepted >= max_per_frontier:
                    break
                staged = self.extractor.stage_candidate(
                    self.latest_map,
                    cell,
                    (robot_pose.x, robot_pose.y),
                    self.latest_inflated_mask,
                )
                if staged is None:
                    continue
                cell, world = staged
                if euclidean((robot_pose.x, robot_pose.y), world) < min_dist:
                    continue
                path_cells, path_cost = self.plan_on_known_free(
                    robot_cell,
                    cell,
                    dist_to_goal_m=dist_to_final_goal,
                )
                if not path_cells:
                    continue

                cand = FrontierCandidate(
                    candidate_id=f"fb_{cluster.cluster_id}_{idx}",
                    frontier_id=cluster.cluster_id,
                    world=world,
                    cell=cell,
                    information_gain=cluster.information_gain,
                )
                cand.path_cells = path_cells
                cand.path_cost = path_cost
                cand.revisit_penalty = self.memory.revisit_penalty(cand.cell, now_sec)
                cand.failed_penalty = self.memory.frontier_failed_penalty(cand.frontier_id, now_sec)
                cand.heading_penalty = self.scorer.heading_change_penalty(robot_pose, cand.world)
                cand.goal_alignment_bonus = self.scorer.goal_alignment_bonus(robot_pose, cand.world, final_goal_xy)
                cand.commitment_bonus = self.memory.commitment_bonus(cand.frontier_id, now_sec)
                cand.clearance_bonus = self.extractor.estimate_clearance(self.latest_map, cand.cell)
                scored.append(cand)
                accepted += 1

        if not scored:
            return []
        scored = self.scorer.score_candidates(scored, robot_pose, final_goal_xy, self.memory, now_sec)
        scored = self.learned_ranker.rerank(scored)
        scored = scored[: max(1, self.max_scored_candidates)]
        self.get_logger().info("Using boundary fallback candidates: %d" % len(scored))
        return scored

    def check_goal_reachable(
        self,
        robot_cell: Tuple[int, int],
        dist_to_final_goal: float,
    ) -> Tuple[bool, List[Tuple[int, int]], float]:
        if self.latest_map is None or self.final_goal is None:
            self.latest_goal_snap_cell = None
            return (False, [], float("inf"))

        goal_xy = (self.final_goal.pose.position.x, self.final_goal.pose.position.y)
        goal_cell = world_to_grid(self.latest_map, goal_xy[0], goal_xy[1])
        if goal_cell is None:
            self.latest_goal_snap_cell = None
            return (False, [], float("inf"))

        goal_cell = nearest_free_cell(
            self.latest_map,
            goal_cell,
            self.latest_inflated_mask,
            self.free_threshold,
            self.goal_snap_radius_cells,
        )
        if goal_cell is None:
            self.latest_goal_snap_cell = None
            return (False, [], float("inf"))

        path_cells, path_cost = self.plan_on_known_free(
            robot_cell,
            goal_cell,
            dist_to_goal_m=dist_to_final_goal,
        )
        if not path_cells:
            self.latest_goal_snap_cell = None
            return (False, [], float("inf"))

        # Do not accept meaningless tiny path to near-obstacle snap point.
        path_m = path_length_m(path_cells, self.latest_map.info.resolution)
        if path_m < max(0.2, self.goal_tolerance_m * 0.5):
            self.latest_goal_snap_cell = None
            return (False, [], float("inf"))
        self.latest_goal_snap_cell = goal_cell
        return (True, path_cells, path_cost)

    def plan_on_known_free(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        dist_to_goal_m: float = float("inf"),
        allow_unknown_fallback: bool = False,
    ) -> Tuple[List[Tuple[int, int]], float]:
        if self.latest_map is None:
            return ([], float("inf"))

        def _plan_with_mask(mask: List[bool], allow_unknown: bool) -> Tuple[List[Tuple[int, int]], float]:
            start_free = nearest_free_cell(
                self.latest_map,
                start,
                mask,
                self.free_threshold,
                self.goal_snap_radius_cells,
            )
            if start_free is None:
                return ([], float("inf"))
            goal_free = nearest_free_cell(
                self.latest_map,
                goal,
                mask,
                self.free_threshold,
                self.goal_snap_radius_cells,
            )
            if goal_free is None:
                return ([], float("inf"))
            return astar_path(
                self.latest_map,
                mask,
                start_free,
                goal_free,
                free_threshold=self.free_threshold,
                allow_unknown=allow_unknown,
                revisit_heat=self.memory.visited_heat,
                revisit_cost_scale=self.astar_revisit_cost_scale,
                soft_obstacle_cells=self.latest_soft_obstacle_cells,
                soft_obstacle_cost_scale=self.noise_filter_soft_cost_scale,
                unknown_cost_scale=self.optimistic_unknown_cost_scale if allow_unknown else 0.0,
            )

        path_cells, path_cost = _plan_with_mask(self.latest_inflated_mask, allow_unknown=False)
        if path_cells:
            return (path_cells, path_cost)

        can_relax = bool(self.latest_relaxed_inflated_mask) and (
            self.latest_relaxed_inflated_mask is not self.latest_inflated_mask
        )
        if not can_relax:
            return ([], float("inf"))

        relaxed_path, relaxed_cost = _plan_with_mask(self.latest_relaxed_inflated_mask, allow_unknown=False)
        if relaxed_path:
            if self.slim_inflation_fallback_enabled or (
                self.dynamic_inflation_enabled
                and math.isfinite(dist_to_goal_m)
                and dist_to_goal_m < self.dynamic_inflation_goal_dist_m
            ):
                self.get_logger().debug(
                    "Inflation fallback succeeded (dist_to_goal=%.2f m)." % dist_to_goal_m
                )
            return (relaxed_path, relaxed_cost)
        if allow_unknown_fallback:
            optimistic_path, optimistic_cost = _plan_with_mask(self.latest_relaxed_inflated_mask, allow_unknown=True)
            if optimistic_path:
                return (optimistic_path, optimistic_cost)
        return ([], float("inf"))

    def make_final_goal_target(self, path_cells: List[Tuple[int, int]]) -> NavigationTarget:
        assert self.final_goal is not None
        x = float(self.final_goal.pose.position.x)
        y = float(self.final_goal.pose.position.y)
        if self.latest_map is not None and self.latest_goal_snap_cell is not None:
            x, y = grid_to_world(self.latest_map, self.latest_goal_snap_cell)
        yaw = yaw_from_quaternion(self.final_goal.pose.orientation)
        return NavigationTarget(
            target_type="final",
            target_id="final_goal",
            pose_world=(x, y, yaw),
            path_cells=path_cells,
            frontier_id=None,
            candidate_id=None,
        )

    def dispatch_target_if_needed(self, target: NavigationTarget, now_sec: float) -> None:
        if self.active_target is not None and self.is_same_target(self.active_target, target):
            # keep current command
            self.last_plan_stamp = now_sec
            self.publish_selected_target(target)
            self.publish_selected_path(target.path_cells)
            return

        if self.active_target is not None and now_sec < self.target_lock_until:
            if not self.active_target_has_fatal_conflict(now_sec):
                # Decision persistence: keep current target for a short horizon.
                self.publish_selected_target(self.active_target)
                self.publish_selected_path(self.active_target.path_cells)
                self.last_plan_stamp = now_sec
                return
            self.get_logger().warn("Decision lock overridden due to fatal path conflict.")

        self.cancel_active_navigation()

        if not self.nav_client.server_is_ready():
            if not self.nav_client.wait_for_server(timeout_sec=0.25):
                self.get_logger().warn("Waiting for navigate_to_pose action server...")
                return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.make_pose(target.pose_world[0], target.pose_world[1], target.pose_world[2])

        self.get_logger().info(
            "Dispatching target: type=%s id=%s pose=(%.2f, %.2f, %.2f) frame=%s"
            % (
                target.target_type,
                target.target_id,
                target.pose_world[0],
                target.pose_world[1],
                target.pose_world[2],
                self.global_frame,
            )
        )

        self.current_goal_token += 1
        token = self.current_goal_token
        send_future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(lambda fut, t=token, tar=target: self.goal_response_callback(fut, t, tar))

        self.active_target = target
        self.last_plan_stamp = now_sec
        self.target_lock_until = now_sec + max(0.0, self.decision_persistence_sec)
        if target.frontier_id is not None:
            self.memory.register_frontier_selected(target.frontier_id, now_sec)

        self.publish_selected_target(target)
        self.publish_selected_path(target.path_cells)

        if target.target_type == "final":
            self.mode = "FINAL_GOAL_NAV"
        else:
            self.mode = "EXPLORING"

        self.get_logger().info(
            "Dispatch %s target id=%s @ (%.2f, %.2f)"
            % (target.target_type, target.target_id, target.pose_world[0], target.pose_world[1])
        )

    def goal_response_callback(self, future, token: int, target: NavigationTarget) -> None:
        if token != self.current_goal_token:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(
                "Nav2 rejected target: type=%s id=%s pose=(%.2f, %.2f, %.2f) frame=%s"
                % (
                    target.target_type,
                    target.target_id,
                    target.pose_world[0],
                    target.pose_world[1],
                    target.pose_world[2],
                    self.global_frame,
                )
            )
            if self.active_target is not None:
                self.get_logger().warn(
                    "Active target at reject time: type=%s id=%s pose=(%.2f, %.2f, %.2f)"
                    % (
                        self.active_target.target_type,
                        self.active_target.target_id,
                        self.active_target.pose_world[0],
                        self.active_target.pose_world[1],
                        self.active_target.pose_world[2],
                    )
                )
            if target.frontier_id is not None:
                self.register_frontier_failure(target.frontier_id, self.now_sec(), "goal_rejected")
                self.memory.frontier_cooldown_until[target.frontier_id] = max(
                    self.memory.frontier_cooldown_until.get(target.frontier_id, 0.0),
                    self.now_sec() + max(self.replan_period_sec * 2.0, 5.0),
                )
            self.active_goal_handle = None
            self.active_target = None
            self.last_plan_stamp = self.now_sec()
            self.target_lock_until = self.now_sec() + self.replan_period_sec
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut, t=token, tar=target: self.result_callback(fut, t, tar))

    def feedback_callback(self, feedback_msg) -> None:
        remaining = float(feedback_msg.feedback.distance_remaining)
        if remaining + self.progress_min_delta_m < self.last_feedback_remaining:
            self.last_feedback_remaining = remaining
            self.memory.last_motion_stamp = self.now_sec()

    def result_callback(self, future, token: int, target: NavigationTarget) -> None:
        if token != self.current_goal_token:
            return

        self.active_goal_handle = None
        result = future.result()
        if result is None:
            return

        status = result.status
        if status == GoalStatus.STATUS_CANCELED and token in self.ignored_cancel_tokens:
            self.ignored_cancel_tokens.discard(token)
            return

        self.ignored_cancel_tokens.discard(token)

        now_sec = self.now_sec()
        if status == GoalStatus.STATUS_SUCCEEDED:
            if target.target_type == "final":
                self.on_final_goal_reached()
                return
            if target.frontier_id is not None:
                self.memory.register_frontier_success(target.frontier_id)
                self.memory.mark_path_visited(target.path_cells, now_sec)
            self.active_target = None
            self.target_lock_until = now_sec
            self.last_plan_stamp = 0.0
            self.mode = "EXPLORING"
            return

        if target.target_type == "frontier" and target.frontier_id is not None:
            self.register_frontier_failure(target.frontier_id, now_sec, "nav2_status_%d" % int(status))
            self.trigger_recovery("frontier_nav_failed", now_sec)
        elif target.target_type == "final":
            self.final_goal_reachable_cache = False
            self.latest_goal_snap_cell = None
            self.trigger_recovery("final_goal_nav_failed", now_sec)

        self.active_target = None
        self.target_lock_until = now_sec

    def register_frontier_failure(self, frontier_id: Optional[str], now_sec: float, reason: str) -> None:
        if frontier_id is None:
            return
        self.memory.register_frontier_failure(frontier_id, now_sec)
        self.total_frontier_failures += 1
        self.get_logger().warn(
            "Frontier failure: id=%s total=%d reason=%s"
            % (frontier_id, self.total_frontier_failures, reason)
        )
        if self.total_frontier_failures >= self.max_frontier_failures_total:
            self.mode = "FAILED"
            self.get_logger().error("Exceeded max_frontier_failures_total=%d" % self.max_frontier_failures_total)

    def trigger_recovery(self, reason: str, now_sec: float) -> None:
        self.cancel_active_navigation()
        self.memory.clear_commitment()
        self.target_lock_until = now_sec

        req = ClearEntireCostmap.Request()
        if self.clear_local_client.service_is_ready():
            self.clear_local_client.call_async(req)
        if self.clear_global_client.service_is_ready():
            self.clear_global_client.call_async(req)

        self.recovering_until = now_sec + self.recovery_cooldown_sec
        self.mode = "RECOVERY"
        self.get_logger().warn("Recovery triggered: %s" % reason)

    def cancel_active_navigation(self) -> None:
        if self.active_goal_handle is None:
            return
        self.ignored_cancel_tokens.add(self.current_goal_token)
        try:
            self.active_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self.active_goal_handle = None

    def on_final_goal_reached(self) -> None:
        self.cancel_active_navigation()
        self.active_target = None
        self.target_lock_until = self.now_sec()
        self.final_goal = None
        self.final_goal_reachable_cache = False
        self.latest_goal_snap_cell = None
        self.memory.clear_commitment()
        self.clear_final_goal_marker()
        self.mode = "FINISHED"
        self.publish_state("FINISHED")
        self.get_logger().info("Final goal reached.")

    def transform_pose_to_global(self, pose: PoseStamped) -> Optional[PoseStamped]:
        src = pose.header.frame_id.strip() if pose.header.frame_id else self.global_frame
        if src == self.global_frame:
            out = PoseStamped()
            out.header.frame_id = self.global_frame
            out.header.stamp = self.get_clock().now().to_msg()
            out.pose = pose.pose
            return out

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                src,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as ex:
            # Fallback when RViz publishes map/odom but one transform is temporarily absent.
            if src in ("map", "/map", "odom", "/odom"):
                out = PoseStamped()
                out.header.frame_id = self.global_frame
                out.header.stamp = self.get_clock().now().to_msg()
                out.pose = pose.pose
                self.get_logger().warn(
                    "Goal frame fallback: src=%s unavailable (%s), treated as %s"
                    % (src, str(ex), self.global_frame)
                )
                return out
            return None

        tx = tf_msg.transform.translation.x
        ty = tf_msg.transform.translation.y
        tq = tf_msg.transform.rotation
        tyaw = yaw_from_quaternion(tq)

        px = pose.pose.position.x
        py = pose.pose.position.y
        pyaw = yaw_from_quaternion(pose.pose.orientation)

        gx = tx + math.cos(tyaw) * px - math.sin(tyaw) * py
        gy = ty + math.sin(tyaw) * px + math.cos(tyaw) * py
        gyaw = angle_wrap(tyaw + pyaw)

        return self.make_pose(gx, gy, gyaw)

    def get_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException:
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return Pose2D(x=float(t.x), y=float(t.y), yaw=yaw_from_quaternion(q))

    def compute_target_yaw(self, robot_pose: Pose2D, target_xy: Tuple[float, float]) -> float:
        return math.atan2(target_xy[1] - robot_pose.y, target_xy[0] - robot_pose.x)

    def active_target_has_fatal_conflict(self, now_sec: float) -> bool:
        if self.active_target is None or self.latest_map is None:
            return False

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False
        robot_cell = world_to_grid(self.latest_map, robot_pose.x, robot_pose.y)
        if robot_cell is None:
            return True

        target_cell = world_to_grid(
            self.latest_map,
            self.active_target.pose_world[0],
            self.active_target.pose_world[1],
        )
        if target_cell is None:
            return True

        dist_to_final_goal = float("inf")
        if self.final_goal is not None:
            dist_to_final_goal = euclidean(
                (robot_pose.x, robot_pose.y),
                (self.final_goal.pose.position.x, self.final_goal.pose.position.y),
            )
        path_cells, _ = self.plan_on_known_free(
            robot_cell,
            target_cell,
            dist_to_goal_m=dist_to_final_goal,
        )
        if path_cells:
            return False

        if (now_sec - self.last_plan_stamp) > 0.5:
            self.get_logger().warn("Active target became unreachable, unlocking persistence.")
        return True

    def is_same_target(self, a: NavigationTarget, b: NavigationTarget) -> bool:
        if a.target_type != b.target_type:
            return False
        if a.frontier_id is not None and b.frontier_id is not None and a.frontier_id != b.frontier_id:
            return False
        return euclidean((a.pose_world[0], a.pose_world[1]), (b.pose_world[0], b.pose_world[1])) < 0.20

    def publish_state(self, state: str) -> None:
        self.mode = state
        self.state_pub.publish(String(data=state))

    def publish_frontier_markers(self, clusters) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.header.frame_id = self.global_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        max_info = max(1.0, max((c.information_gain for c in clusters), default=1.0))
        for i, c in enumerate(clusters):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns = "frontiers"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(c.centroid_world[0])
            m.pose.position.y = float(c.centroid_world[1])
            m.pose.position.z = 0.03
            m.pose.orientation.w = 1.0
            m.scale.x = 0.14
            m.scale.y = 0.14
            m.scale.z = 0.14
            alpha = min(1.0, c.information_gain / max_info)
            m.color.r = 0.1
            m.color.g = 0.6
            m.color.b = 1.0
            m.color.a = 0.4 + 0.5 * alpha
            markers.markers.append(m)

        self.frontier_pub.publish(markers)

    def clear_candidate_markers(self) -> None:
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self.candidate_pub.publish(arr)

    def publish_candidate_markers(self, candidates: List[FrontierCandidate]) -> None:
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.header.frame_id = self.global_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        best = max((c.score for c in candidates), default=1.0)
        worst = min((c.score for c in candidates), default=0.0)
        span = max(1e-6, best - worst)

        for i, c in enumerate(candidates):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns = "candidates"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(c.world[0])
            m.pose.position.y = float(c.world[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.12
            m.scale.y = 0.12
            m.scale.z = 0.12
            t = (c.score - worst) / span
            m.color.r = 1.0 - t
            m.color.g = t
            m.color.b = 0.15
            m.color.a = 0.9
            arr.markers.append(m)

        self.candidate_pub.publish(arr)

    def publish_rrt_tree(self, edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color.r = 0.0
        marker.color.g = 0.9
        marker.color.b = 1.0
        marker.color.a = 0.35

        for a, b in edges:
            marker.points.append(Point(x=float(a[0]), y=float(a[1]), z=0.02))
            marker.points.append(Point(x=float(b[0]), y=float(b[1]), z=0.02))

        self.rrt_tree_pub.publish(marker)

    def clear_rrt_tree_marker(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.action = Marker.DELETE
        self.rrt_tree_pub.publish(marker)

    def publish_selected_target(self, target: NavigationTarget) -> None:
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "selected_target"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = float(target.pose_world[0])
        m.pose.position.y = float(target.pose_world[1])
        m.pose.position.z = 0.08
        m.pose.orientation = quaternion_from_yaw(float(target.pose_world[2]))
        m.scale.x = 0.35
        m.scale.y = 0.07
        m.scale.z = 0.07
        if target.target_type == "final":
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
        else:
            m.color.r = 1.0
            m.color.g = 0.85
            m.color.b = 0.1
        m.color.a = 0.95
        self.selected_pub.publish(m)

    def publish_final_goal_marker(self) -> None:
        if self.final_goal is None:
            self.clear_final_goal_marker()
            return

        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "final_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.final_goal.pose.position.x)
        m.pose.position.y = float(self.final_goal.pose.position.y)
        m.pose.position.z = 0.10
        m.pose.orientation.w = 1.0
        m.scale.x = 0.18
        m.scale.y = 0.18
        m.scale.z = 0.18
        m.color.r = 1.0
        m.color.g = 0.15
        m.color.b = 0.15
        m.color.a = 0.95
        self.final_goal_pub.publish(m)

    def clear_final_goal_marker(self) -> None:
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "final_goal"
        m.id = 0
        m.action = Marker.DELETE
        self.final_goal_pub.publish(m)

    def publish_robot_pose_marker(self, robot_pose: Pose2D) -> None:
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "robot_pose"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = float(robot_pose.x)
        m.pose.position.y = float(robot_pose.y)
        m.pose.position.z = 0.09
        m.pose.orientation = quaternion_from_yaw(robot_pose.yaw)
        m.scale.x = 0.28
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color.r = 0.1
        m.color.g = 0.55
        m.color.b = 1.0
        m.color.a = 0.95
        self.robot_pose_pub.publish(m)

    def publish_selected_path(self, path_cells: List[Tuple[int, int]]) -> None:
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.latest_map is None:
            self.selected_path_pub.publish(msg)
            return
        for cell in path_cells:
            x, y = grid_to_world(self.latest_map, cell)
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.selected_path_pub.publish(msg)

    def make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(yaw)
        return pose

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationCoordinator()
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
