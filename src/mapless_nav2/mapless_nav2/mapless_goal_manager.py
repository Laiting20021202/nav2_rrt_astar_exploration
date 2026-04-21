#!/usr/bin/env python3
import heapq
import math
import random
import struct
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class RRTNode:
    x: float
    y: float
    parent: int
    cost: float = 0.0


@dataclass
class DeadEndZone:
    x: float
    y: float
    radius: float
    expire_sec: float


@dataclass
class RRTPlan:
    path: List[Tuple[float, float]]
    tree_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


class MaplessGoalManager(Node):
    def __init__(self) -> None:
        super().__init__("mapless_goal_manager")

        self.declare_parameter("goal_topic", "/mapless_goal")
        self.declare_parameter("rviz_goal_topic", "/goal_pose")
        self.declare_parameter("active_subgoal_topic", "/mapless_active_subgoal")
        self.declare_parameter("goal_reached_topic", "/mapless_goal_reached")
        self.declare_parameter("safety_status_topic", "/mapless_safety_active")
        self.declare_parameter("tilt_status_topic", "/scan_tilt_exceeded")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("global_frame", "odom")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("planner_profile", "baseline")

        self.declare_parameter("rrt_path_topic", "/mapless_rrt_path")
        self.declare_parameter("rrt_tree_topic", "/mapless_rrt_tree")
        self.declare_parameter("rrt_goal_marker_topic", "/mapless_rrt_goal")
        self.declare_parameter("breadcrumb_path_topic", "/mapless_breadcrumb_path")
        self.declare_parameter("mission_memory_cloud_topic", "/mapless_memory_points")

        self.declare_parameter("update_rate_hz", 4.0)
        self.declare_parameter("goal_tolerance", 0.30)
        self.declare_parameter("subgoal_lookahead", 0.70)
        self.declare_parameter("subgoal_min_spacing", 0.45)
        self.declare_parameter("subgoal_min_distance", 0.20)
        self.declare_parameter("min_goal_send_interval", 0.8)
        self.declare_parameter("min_subgoal_hold_time", 0.9)
        self.declare_parameter("progress_min_delta", 0.08)
        self.declare_parameter("progress_timeout", 4.0)
        self.declare_parameter("replan_interval", 0.35)
        self.declare_parameter("replan_lock_sec", 0.7)
        self.declare_parameter("blocked_front_clearance", 0.24)
        self.declare_parameter("blocked_persist_sec", 1.4)
        self.declare_parameter("stuck_window_sec", 8.0)
        self.declare_parameter("stuck_radius", 0.22)
        self.declare_parameter("escape_commit_sec", 2.2)
        self.declare_parameter("escape_distance", 0.75)
        self.declare_parameter("escape_heading_range_deg", 100.0)
        self.declare_parameter("escape_heading_step_deg", 8.0)
        self.declare_parameter("escape_min_clearance", 0.35)
        self.declare_parameter("escape_goal_bias", 0.45)
        self.declare_parameter("escape_open_bias", 1.0)
        self.declare_parameter("dead_end_memory_sec", 90.0)
        self.declare_parameter("dead_end_radius", 1.05)
        self.declare_parameter("dead_end_merge_dist", 0.45)
        self.declare_parameter("dead_end_max_zones", 20)
        self.declare_parameter("dead_end_escape_allowance", 0.55)
        self.declare_parameter("dead_end_penalty_weight", 2.0)
        self.declare_parameter("dead_end_replan_cost", 0.35)
        self.declare_parameter("dead_end_sample_reject_prob", 0.55)
        self.declare_parameter("goal_progress_penalty_weight", 1.0)

        self.declare_parameter("planning_horizon", 5.5)
        self.declare_parameter("grid_planner_enabled", False)
        self.declare_parameter("grid_heuristic_weight", 1.35)
        self.declare_parameter("grid_cost_scale", 2.0)
        self.declare_parameter("grid_dead_end_scale", 1.8)
        self.declare_parameter("grid_max_expansions", 5000)
        self.declare_parameter("grid_goal_tolerance_cells", 2)
        self.declare_parameter("rrt_step_size", 0.18)
        self.declare_parameter("rrt_goal_sample_rate", 0.22)
        self.declare_parameter("rrt_max_iterations", 1200)
        self.declare_parameter("rrt_goal_connect_dist", 0.35)
        self.declare_parameter("rrt_partial_min_length", 0.15)
        self.declare_parameter("collision_clearance", 0.22)
        self.declare_parameter("collision_check_resolution", 0.08)
        self.declare_parameter("shortcut_iterations", 30)
        self.declare_parameter("rrt_random_seed", 42)

        self.declare_parameter("scan_keep_time", 6.5)
        self.declare_parameter("scan_max_range", 3.5)
        self.declare_parameter("scan_decimation", 1)
        self.declare_parameter("costmap_collision_threshold", 45)
        self.declare_parameter("costmap_unknown_is_obstacle", True)
        self.declare_parameter("costmap_clearance_padding", 0.04)
        self.declare_parameter("mission_obstacle_resolution", 0.08)
        self.declare_parameter("mission_obstacle_max_cells", 80000)
        self.declare_parameter("mission_obstacle_hit_increment", 1.2)
        self.declare_parameter("mission_obstacle_clear_decrement", 0.25)
        self.declare_parameter("mission_obstacle_block_threshold", 1.0)
        self.declare_parameter("mission_obstacle_block_radius", 0.10)
        self.declare_parameter("mission_obstacle_ray_step", 0.06)
        self.declare_parameter("mission_memory_cloud_max_points", 4500)
        self.declare_parameter("mission_memory_publish_interval", 0.25)
        self.declare_parameter("experience_resolution", 0.18)
        self.declare_parameter("experience_max_cells", 40000)
        self.declare_parameter("experience_skip_start_distance", 0.40)
        self.declare_parameter("experience_revisit_penalty_weight", 0.0)
        self.declare_parameter("experience_fail_penalty_weight", 0.8)
        self.declare_parameter("experience_penalty_cap", 2.5)
        self.declare_parameter("breadcrumb_spacing", 0.20)
        self.declare_parameter("breadcrumb_max_points", 2400)
        self.declare_parameter("breadcrumb_retreat_distance", 1.20)
        self.declare_parameter("breadcrumb_min_goal_distance", 0.45)
        self.declare_parameter("breadcrumb_path_reuse_weight", 0.0)
        self.declare_parameter("blocked_branch_memory_sec", 140.0)
        self.declare_parameter("blocked_branch_memory_distance", 2.8)
        self.declare_parameter("blocked_branch_zone_radius", 0.42)
        self.declare_parameter("blocked_branch_skip_distance", 0.65)
        self.declare_parameter("blocked_branch_zone_step", 0.35)

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.rviz_goal_topic = str(self.get_parameter("rviz_goal_topic").value)
        self.active_subgoal_topic = str(self.get_parameter("active_subgoal_topic").value)
        self.goal_reached_topic = str(self.get_parameter("goal_reached_topic").value)
        self.safety_status_topic = str(self.get_parameter("safety_status_topic").value)
        self.tilt_status_topic = str(self.get_parameter("tilt_status_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.local_costmap_topic = str(self.get_parameter("local_costmap_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.planner_profile = str(self.get_parameter("planner_profile").value).strip().lower()
        if self.planner_profile not in ("baseline", "advanced"):
            self.get_logger().warn(
                "Unknown planner_profile '%s', fallback to 'baseline'." % self.planner_profile
            )
            self.planner_profile = "baseline"
        self.advanced_mode = self.planner_profile == "advanced"

        self.rrt_path_topic = str(self.get_parameter("rrt_path_topic").value)
        self.rrt_tree_topic = str(self.get_parameter("rrt_tree_topic").value)
        self.rrt_goal_marker_topic = str(self.get_parameter("rrt_goal_marker_topic").value)
        self.breadcrumb_path_topic = str(self.get_parameter("breadcrumb_path_topic").value)
        self.mission_memory_cloud_topic = str(self.get_parameter("mission_memory_cloud_topic").value)

        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.subgoal_lookahead = float(self.get_parameter("subgoal_lookahead").value)
        self.subgoal_min_spacing = float(self.get_parameter("subgoal_min_spacing").value)
        self.subgoal_min_distance = float(self.get_parameter("subgoal_min_distance").value)
        self.min_goal_send_interval = float(self.get_parameter("min_goal_send_interval").value)
        self.min_subgoal_hold_time = float(self.get_parameter("min_subgoal_hold_time").value)
        self.progress_min_delta = float(self.get_parameter("progress_min_delta").value)
        self.progress_timeout = float(self.get_parameter("progress_timeout").value)
        self.replan_interval = float(self.get_parameter("replan_interval").value)
        self.replan_lock_sec = float(self.get_parameter("replan_lock_sec").value)
        self.blocked_front_clearance = float(self.get_parameter("blocked_front_clearance").value)
        self.blocked_persist_sec = float(self.get_parameter("blocked_persist_sec").value)
        self.stuck_window_sec = float(self.get_parameter("stuck_window_sec").value)
        self.stuck_radius = float(self.get_parameter("stuck_radius").value)
        self.escape_commit_sec = float(self.get_parameter("escape_commit_sec").value)
        self.escape_distance = float(self.get_parameter("escape_distance").value)
        self.escape_heading_range = math.radians(float(self.get_parameter("escape_heading_range_deg").value))
        self.escape_heading_step = math.radians(float(self.get_parameter("escape_heading_step_deg").value))
        self.escape_min_clearance = float(self.get_parameter("escape_min_clearance").value)
        self.escape_goal_bias = float(self.get_parameter("escape_goal_bias").value)
        self.escape_open_bias = float(self.get_parameter("escape_open_bias").value)
        self.dead_end_memory_sec = float(self.get_parameter("dead_end_memory_sec").value)
        self.dead_end_radius = float(self.get_parameter("dead_end_radius").value)
        self.dead_end_merge_dist = float(self.get_parameter("dead_end_merge_dist").value)
        self.dead_end_max_zones = max(4, int(self.get_parameter("dead_end_max_zones").value))
        self.dead_end_escape_allowance = float(self.get_parameter("dead_end_escape_allowance").value)
        self.dead_end_penalty_weight = float(self.get_parameter("dead_end_penalty_weight").value)
        self.dead_end_replan_cost = float(self.get_parameter("dead_end_replan_cost").value)
        self.dead_end_sample_reject_prob = float(self.get_parameter("dead_end_sample_reject_prob").value)
        self.goal_progress_penalty_weight = float(self.get_parameter("goal_progress_penalty_weight").value)

        self.planning_horizon = float(self.get_parameter("planning_horizon").value)
        self.grid_planner_enabled = bool(self.get_parameter("grid_planner_enabled").value)
        self.grid_heuristic_weight = float(self.get_parameter("grid_heuristic_weight").value)
        self.grid_cost_scale = float(self.get_parameter("grid_cost_scale").value)
        self.grid_dead_end_scale = float(self.get_parameter("grid_dead_end_scale").value)
        self.grid_max_expansions = max(500, int(self.get_parameter("grid_max_expansions").value))
        self.grid_goal_tolerance_cells = max(0, int(self.get_parameter("grid_goal_tolerance_cells").value))
        self.rrt_step_size = float(self.get_parameter("rrt_step_size").value)
        self.rrt_goal_sample_rate = float(self.get_parameter("rrt_goal_sample_rate").value)
        self.rrt_max_iterations = int(self.get_parameter("rrt_max_iterations").value)
        self.rrt_goal_connect_dist = float(self.get_parameter("rrt_goal_connect_dist").value)
        self.rrt_partial_min_length = max(0.05, float(self.get_parameter("rrt_partial_min_length").value))
        self.collision_clearance = float(self.get_parameter("collision_clearance").value)
        self.collision_check_resolution = float(self.get_parameter("collision_check_resolution").value)
        self.shortcut_iterations = int(self.get_parameter("shortcut_iterations").value)
        self.rrt_random_seed = int(self.get_parameter("rrt_random_seed").value)

        self.scan_keep_time = float(self.get_parameter("scan_keep_time").value)
        self.scan_max_range = float(self.get_parameter("scan_max_range").value)
        self.scan_decimation = max(1, int(self.get_parameter("scan_decimation").value))
        self.costmap_collision_threshold = int(self.get_parameter("costmap_collision_threshold").value)
        self.costmap_unknown_is_obstacle = bool(self.get_parameter("costmap_unknown_is_obstacle").value)
        self.costmap_clearance_padding = max(0.0, float(self.get_parameter("costmap_clearance_padding").value))
        self.mission_obstacle_resolution = max(0.03, float(self.get_parameter("mission_obstacle_resolution").value))
        self.mission_obstacle_max_cells = max(1000, int(self.get_parameter("mission_obstacle_max_cells").value))
        self.mission_obstacle_hit_increment = max(0.2, float(self.get_parameter("mission_obstacle_hit_increment").value))
        self.mission_obstacle_clear_decrement = max(0.05, float(self.get_parameter("mission_obstacle_clear_decrement").value))
        self.mission_obstacle_block_threshold = max(0.5, float(self.get_parameter("mission_obstacle_block_threshold").value))
        self.mission_obstacle_block_radius = max(0.05, float(self.get_parameter("mission_obstacle_block_radius").value))
        self.mission_obstacle_ray_step = max(0.03, float(self.get_parameter("mission_obstacle_ray_step").value))
        self.mission_memory_cloud_max_points = max(200, int(self.get_parameter("mission_memory_cloud_max_points").value))
        self.mission_memory_publish_interval = max(0.05, float(self.get_parameter("mission_memory_publish_interval").value))
        self.experience_resolution = max(0.05, float(self.get_parameter("experience_resolution").value))
        self.experience_max_cells = max(1000, int(self.get_parameter("experience_max_cells").value))
        self.experience_skip_start_distance = max(0.0, float(self.get_parameter("experience_skip_start_distance").value))
        self.experience_revisit_penalty_weight = max(0.0, float(self.get_parameter("experience_revisit_penalty_weight").value))
        self.experience_fail_penalty_weight = max(0.0, float(self.get_parameter("experience_fail_penalty_weight").value))
        self.experience_penalty_cap = max(0.5, float(self.get_parameter("experience_penalty_cap").value))
        self.breadcrumb_spacing = max(0.05, float(self.get_parameter("breadcrumb_spacing").value))
        self.breadcrumb_max_points = max(200, int(self.get_parameter("breadcrumb_max_points").value))
        self.breadcrumb_retreat_distance = max(0.4, float(self.get_parameter("breadcrumb_retreat_distance").value))
        self.breadcrumb_min_goal_distance = max(0.2, float(self.get_parameter("breadcrumb_min_goal_distance").value))
        self.breadcrumb_path_reuse_weight = max(0.0, float(self.get_parameter("breadcrumb_path_reuse_weight").value))
        self.blocked_branch_memory_sec = max(20.0, float(self.get_parameter("blocked_branch_memory_sec").value))
        self.blocked_branch_memory_distance = max(0.8, float(self.get_parameter("blocked_branch_memory_distance").value))
        self.blocked_branch_zone_radius = max(0.2, float(self.get_parameter("blocked_branch_zone_radius").value))
        self.blocked_branch_skip_distance = max(0.2, float(self.get_parameter("blocked_branch_skip_distance").value))
        self.blocked_branch_zone_step = max(0.1, float(self.get_parameter("blocked_branch_zone_step").value))

        if not self.advanced_mode:
            # Baseline profile keeps planner behavior close to the initial simple RRT setup.
            self.dead_end_penalty_weight = 0.0
            self.dead_end_sample_reject_prob = 0.0
            self.goal_progress_penalty_weight = 0.0
            self.grid_dead_end_scale = 0.0
            self.experience_revisit_penalty_weight = 0.0
            self.experience_fail_penalty_weight = 0.0
            # Disable dead-end memory side effects in baseline mode.
            self.dead_end_replan_cost = 1.0e9
            self.dead_end_memory_sec = 0.0

        self.rng = random.Random(self.rrt_random_seed)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        if self.rviz_goal_topic != self.goal_topic:
            self.rviz_goal_sub = self.create_subscription(PoseStamped, self.rviz_goal_topic, self.goal_callback, 10)
        else:
            self.rviz_goal_sub = None
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.safety_sub = self.create_subscription(Bool, self.safety_status_topic, self.safety_callback, 10)
        self.tilt_sub = self.create_subscription(Bool, self.tilt_status_topic, self.tilt_callback, 10)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            10,
        )

        viz_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.active_subgoal_pub = self.create_publisher(PoseStamped, self.active_subgoal_topic, 10)
        self.goal_reached_pub = self.create_publisher(Bool, self.goal_reached_topic, 10)
        self.rrt_path_pub = self.create_publisher(Path, self.rrt_path_topic, viz_qos)
        self.rrt_tree_pub = self.create_publisher(Marker, self.rrt_tree_topic, viz_qos)
        self.rrt_goal_marker_pub = self.create_publisher(Marker, self.rrt_goal_marker_topic, viz_qos)
        self.breadcrumb_path_pub = self.create_publisher(Path, self.breadcrumb_path_topic, viz_qos)
        self.mission_memory_cloud_pub = self.create_publisher(PointCloud2, self.mission_memory_cloud_topic, viz_qos)

        self.timer = self.create_timer(1.0 / max(self.update_rate_hz, 0.1), self.timer_callback)

        self.final_goal: Optional[PoseStamped] = None
        self.active_subgoal: Optional[PoseStamped] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_local_costmap: Optional[OccupancyGrid] = None
        self.safety_active = False
        self.obstacle_points: Deque[Tuple[float, float, float]] = deque()
        self.mission_obstacle_cells: Dict[Tuple[int, int], float] = {}
        self.path_visit_cells: Dict[Tuple[int, int], int] = {}
        self.failed_branch_cells: Dict[Tuple[int, int], float] = {}
        self.pose_history: Deque[Tuple[float, float, float]] = deque()
        self.dead_end_zones: Deque[DeadEndZone] = deque()
        self.breadcrumb_points: Deque[Tuple[float, float]] = deque()
        self.current_plan: Optional[RRTPlan] = None
        self.escape_goal: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.ignored_cancel_tokens = set()

        self.navigation_active = False
        self.current_goal_token = 0

        now = self.get_clock().now()
        self.escape_until = now
        self.escape_attempts = 0
        self.last_send_time = None
        self.last_subgoal_time = now
        self.last_plan_time = now
        self.plan_lock_until = now
        self.last_progress_time = now
        self.best_distance = float("inf")
        self.last_feedback_remaining = float("inf")
        self.last_tf_warn_time = None
        self.last_plan_warn_time = None
        self.front_blocked_since = None
        self.consecutive_plan_failures = 0
        self.mission_memory_dirty = False
        self.last_memory_cloud_pub_time = now
        self.last_tilt_purge_time = now

        self.get_logger().info(
            "Mapless planner ready: profile=%s grid=%s horizon=%.2f rrt_iters=%d dead_end_mem=%.0fs breadcrumb=%.2fm"
            % (
                self.planner_profile,
                "on" if self.grid_planner_enabled else "off",
                self.planning_horizon,
                self.rrt_max_iterations,
                self.dead_end_memory_sec,
                self.breadcrumb_spacing,
            )
        )

    def goal_callback(self, msg: PoseStamped) -> None:
        src_frame = msg.header.frame_id.strip() if msg.header.frame_id else self.global_frame
        goal = self.transform_pose_to_global(msg)
        if goal is None:
            self.get_logger().warn(
                "Drop goal: cannot transform frame '%s' to global frame '%s'."
                % (src_frame, self.global_frame)
            )
            return
        self.cancel_active_navigation()

        self.final_goal = goal
        self.active_subgoal = None
        self.current_plan = None
        self.escape_goal = None
        self.escape_attempts = 0
        self.escape_until = self.get_clock().now()
        self.plan_lock_until = self.get_clock().now()
        self.navigation_active = False
        self.front_blocked_since = None
        self.consecutive_plan_failures = 0
        self.best_distance = float("inf")
        self.last_feedback_remaining = float("inf")
        self.last_progress_time = self.get_clock().now()
        self.pose_history.clear()
        self.breadcrumb_points.clear()
        self.mission_obstacle_cells.clear()
        self.mission_memory_dirty = True
        self.publish_mission_memory_cloud(force=True)
        self.path_visit_cells.clear()
        self.failed_branch_cells.clear()
        robot_pose = self.get_robot_pose(log_warning=False)
        if robot_pose is not None:
            self.breadcrumb_points.append((robot_pose.x, robot_pose.y))
            self.publish_breadcrumb_visualization()
        else:
            self.clear_breadcrumb_visualization()
        self.goal_reached_pub.publish(Bool(data=False))

        self.publish_goal_marker(goal)
        self.clear_rrt_visualization(clear_goal=False)

        self.get_logger().info(
            "Received final goal at (%.2f, %.2f), src_frame='%s' planned_frame='%s'."
            % (goal.pose.position.x, goal.pose.position.y, src_frame, goal.header.frame_id)
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

        robot_pose = self.get_robot_pose(log_warning=False)
        if robot_pose is None:
            return

        now_sec = self.now_seconds()
        max_range = min(float(msg.range_max), self.scan_max_range)

        angle = float(msg.angle_min)
        angle_inc = float(msg.angle_increment)
        for idx, r in enumerate(msg.ranges):
            if idx % self.scan_decimation != 0:
                angle += angle_inc
                continue

            mark_hit = False
            if math.isfinite(r):
                if r < float(msg.range_min):
                    angle += angle_inc
                    continue
                if r <= max_range:
                    beam_range = float(r)
                    mark_hit = True
                else:
                    # No valid hit in bounded range: clear memory along the beam.
                    beam_range = max_range
            elif math.isinf(r) and r > 0.0:
                # Infinite return means free space up to sensor max range.
                beam_range = max_range
            else:
                angle += angle_inc
                continue

            g_angle = robot_pose.yaw + angle
            px = robot_pose.x + beam_range * math.cos(g_angle)
            py = robot_pose.y + beam_range * math.sin(g_angle)
            if mark_hit:
                self.obstacle_points.append((px, py, now_sec))
            self.update_mission_obstacle_ray(robot_pose.x, robot_pose.y, px, py, mark_hit=mark_hit)
            angle += angle_inc

        self.prune_obstacles(now_sec)

    def local_costmap_callback(self, msg: OccupancyGrid) -> None:
        self.latest_local_costmap = msg

    def safety_callback(self, msg: Bool) -> None:
        self.safety_active = bool(msg.data)

    def tilt_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        now = self.get_clock().now()
        if (now - self.last_tilt_purge_time) < Duration(seconds=1.0):
            return
        self.last_tilt_purge_time = now
        if self.mission_obstacle_cells:
            self.mission_obstacle_cells.clear()
            self.mission_memory_dirty = True
            self.publish_mission_memory_cloud(force=True)

    def timer_callback(self) -> None:
        if self.final_goal is None:
            return

        robot_pose = self.get_robot_pose(log_warning=True)
        if robot_pose is None:
            return

        now = self.get_clock().now()
        now_sec = self.now_seconds()
        self.prune_obstacles(now_sec)
        self.prune_dead_end_zones(now_sec)
        self.update_pose_history(now_sec, robot_pose)
        self.update_breadcrumbs(robot_pose)
        self.publish_mission_memory_cloud()
        obstacles = [(x, y) for (x, y, _) in self.obstacle_points]

        fx = self.final_goal.pose.position.x
        fy = self.final_goal.pose.position.y
        dist_to_final = distance_xy(robot_pose.x, robot_pose.y, fx, fy)

        if dist_to_final + self.progress_min_delta < self.best_distance:
            self.best_distance = dist_to_final
            self.last_progress_time = now
            self.escape_attempts = 0

        if dist_to_final <= self.goal_tolerance:
            self.get_logger().info("Reached final goal within tolerance %.2f m." % self.goal_tolerance)
            self.final_goal = None
            self.active_subgoal = None
            self.current_plan = None
            self.escape_goal = None
            self.escape_attempts = 0
            self.pose_history.clear()
            self.breadcrumb_points.clear()
            self.mission_obstacle_cells.clear()
            self.mission_memory_dirty = True
            self.publish_mission_memory_cloud(force=True)
            self.path_visit_cells.clear()
            self.failed_branch_cells.clear()
            self.navigation_active = False
            self.front_blocked_since = None
            self.consecutive_plan_failures = 0
            self.plan_lock_until = now
            self.goal_reached_pub.publish(Bool(data=True))
            self.clear_rrt_visualization(clear_goal=True)
            self.clear_breadcrumb_visualization()
            return

        front_clearance = self.get_scan_clearance(0.0)
        trapped = self.is_robot_trapped()
        in_escape = now < self.escape_until and self.escape_goal is not None
        if (not in_escape) and self.escape_goal is not None and now >= self.escape_until:
            self.escape_goal = None
        reached_active = False
        active_blocked = False
        if self.active_subgoal is not None:
            ax = self.active_subgoal.pose.position.x
            ay = self.active_subgoal.pose.position.y
            reached_active = distance_xy(robot_pose.x, robot_pose.y, ax, ay) < max(0.25, 0.5 * self.subgoal_lookahead)
            active_blocked = not self.segment_collision_free((robot_pose.x, robot_pose.y), (ax, ay), obstacles)

        no_progress = (now - self.last_progress_time) > Duration(seconds=self.progress_timeout)
        stale_plan = (now - self.last_plan_time) > Duration(seconds=self.replan_interval)
        front_blocked = 0.0 < front_clearance < self.blocked_front_clearance
        if self.safety_active:
            front_blocked = True
        if front_blocked:
            if self.front_blocked_since is None:
                self.front_blocked_since = now
        else:
            self.front_blocked_since = None
        blocked_persistent = (
            self.front_blocked_since is not None
            and (now - self.front_blocked_since) > Duration(seconds=self.blocked_persist_sec)
        )
        path_invalid = self.current_plan is not None and not self.path_still_valid(robot_pose, self.current_plan.path, obstacles)
        dead_end_invalid = self.current_plan is not None and self.path_runs_into_dead_end(robot_pose, self.current_plan.path)
        in_plan_lock = now < self.plan_lock_until

        if (trapped or blocked_persistent or (front_blocked and active_blocked)) and not in_escape:
            self.start_escape_mode(
                robot_pose,
                front_blocked,
                obstacles,
                mark_blocked_branch=(blocked_persistent or (front_blocked and no_progress)),
            )
            in_escape = now < self.escape_until and self.escape_goal is not None

        if in_escape:
            escape_subgoal = self.escape_goal
            if escape_subgoal is None:
                return

            if self.is_subgoal_too_close(escape_subgoal, robot_pose):
                self.start_escape_mode(
                    robot_pose,
                    front_blocked,
                    obstacles,
                    mark_blocked_branch=front_blocked,
                )
                escape_subgoal = self.escape_goal
                if escape_subgoal is None or self.is_subgoal_too_close(escape_subgoal, robot_pose):
                    return

            force_send = True
            if self.should_send_subgoal(escape_subgoal, now, robot_pose, force_send):
                self.send_subgoal(escape_subgoal)
            return

        need_replan = (
            self.current_plan is None
            or no_progress
            or active_blocked
            or blocked_persistent
            or path_invalid
            or dead_end_invalid
            or (stale_plan and reached_active)
        )
        if in_plan_lock and not (no_progress or blocked_persistent or path_invalid or dead_end_invalid):
            need_replan = self.current_plan is None

        if need_replan:
            previous_plan = self.current_plan
            if self.current_plan is not None and (no_progress or active_blocked or blocked_persistent or path_invalid or dead_end_invalid):
                self.record_failed_plan_memory(robot_pose, self.current_plan)
            plan = self.plan_rrt_path(robot_pose, obstacles)
            if plan is not None:
                self.current_plan = plan
                self.last_plan_time = now
                self.consecutive_plan_failures = 0
                self.plan_lock_until = now + Duration(seconds=self.replan_lock_sec)
                self.publish_rrt_visualization(plan)
            else:
                keep_previous = (
                    previous_plan is not None
                    and not reached_active
                    and not no_progress
                    and not active_blocked
                    and not blocked_persistent
                    and not path_invalid
                    and not dead_end_invalid
                    and self.path_still_valid(robot_pose, previous_plan.path, obstacles)
                )
                if keep_previous:
                    self.current_plan = previous_plan
                    self.last_plan_time = now
                    self.consecutive_plan_failures = max(0, self.consecutive_plan_failures - 1)
                    self.publish_rrt_visualization(previous_plan)
                    return
                self.current_plan = None
                self.active_subgoal = None
                self.cancel_active_navigation()
                self.consecutive_plan_failures += 1
                if blocked_persistent or no_progress or self.consecutive_plan_failures >= 2:
                    self.start_escape_mode(
                        robot_pose,
                        front_blocked,
                        obstacles,
                        mark_blocked_branch=True,
                    )
                in_escape = now < self.escape_until and self.escape_goal is not None
                if in_escape and self.escape_goal is not None:
                    if self.should_send_subgoal(self.escape_goal, now, robot_pose, True):
                        self.send_subgoal(self.escape_goal)
                    return
                if self.should_log_plan_warning(now):
                    self.get_logger().warn("Planner could not find a collision-free path this cycle.")
                self.clear_rrt_visualization(clear_goal=False)
                return

        if self.current_plan is None:
            return

        subgoal = self.select_subgoal_from_path(self.current_plan.path, robot_pose, front_clearance)
        if self.is_subgoal_too_close(subgoal, robot_pose):
            self.current_plan = None
            if front_blocked or no_progress:
                self.start_escape_mode(
                    robot_pose,
                    front_blocked,
                    obstacles,
                    mark_blocked_branch=True,
                )
            return
        force_send = no_progress or active_blocked or blocked_persistent

        should_send = self.should_send_subgoal(subgoal, now, robot_pose, force_send)
        if (not should_send) and (not self.navigation_active):
            # Keep re-dispatching when Nav2 is idle to avoid "plan exists but robot does not move".
            if (now - self.last_subgoal_time) > Duration(seconds=1.2):
                should_send = True

        if should_send:
            self.send_subgoal(subgoal)

    def should_log_plan_warning(self, now) -> bool:
        if self.last_plan_warn_time is None:
            self.last_plan_warn_time = now
            return True
        if (now - self.last_plan_warn_time) > Duration(seconds=2.0):
            self.last_plan_warn_time = now
            return True
        return False

    def should_send_subgoal(self, subgoal: PoseStamped, now, robot_pose: Pose2D, force_send: bool) -> bool:
        if self.is_subgoal_too_close(subgoal, robot_pose):
            return False

        if self.last_send_time is not None and (now - self.last_send_time) < Duration(seconds=self.min_goal_send_interval):
            return False

        if self.active_subgoal is None:
            return True

        ax = self.active_subgoal.pose.position.x
        ay = self.active_subgoal.pose.position.y
        reached_active = distance_xy(robot_pose.x, robot_pose.y, ax, ay) < max(0.25, 0.5 * self.subgoal_lookahead)

        new = subgoal.pose.position
        delta = distance_xy(ax, ay, new.x, new.y)
        changed = delta > self.subgoal_min_spacing

        if force_send:
            return True
        if reached_active:
            return True

        if (now - self.last_subgoal_time) < Duration(seconds=self.min_subgoal_hold_time) and self.navigation_active:
            return False

        return (not self.navigation_active) or changed

    def is_subgoal_too_close(self, subgoal: PoseStamped, robot_pose: Pose2D) -> bool:
        gx = subgoal.pose.position.x
        gy = subgoal.pose.position.y
        d = distance_xy(robot_pose.x, robot_pose.y, gx, gy)
        if d >= self.subgoal_min_distance:
            return False

        if self.final_goal is None:
            return True

        fx = self.final_goal.pose.position.x
        fy = self.final_goal.pose.position.y
        dist_to_final = distance_xy(robot_pose.x, robot_pose.y, fx, fy)
        return dist_to_final > self.goal_tolerance * 1.6

    def cancel_active_navigation(self) -> None:
        if self.active_goal_handle is None:
            return
        self.ignored_cancel_tokens.add(self.current_goal_token)
        try:
            self.active_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self.active_goal_handle = None
        self.navigation_active = False

    def send_subgoal(self, subgoal: PoseStamped) -> None:
        if not self.nav_to_pose_client.server_is_ready():
            if not self.nav_to_pose_client.wait_for_server(timeout_sec=0.2):
                self.get_logger().warn("Waiting for nav2 action server '/navigate_to_pose'...")
                return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = subgoal

        self.current_goal_token += 1
        current_token = self.current_goal_token

        send_future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(lambda fut, token=current_token: self.goal_response_callback(fut, token))

        self.active_subgoal = subgoal
        self.last_send_time = self.get_clock().now()
        self.last_subgoal_time = self.last_send_time
        self.active_subgoal_pub.publish(subgoal)

        self.get_logger().info("Dispatch RRT subgoal (%.2f, %.2f)." % (subgoal.pose.position.x, subgoal.pose.position.y))

    def goal_response_callback(self, future, token: int) -> None:
        if token != self.current_goal_token:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.navigation_active = False
            self.active_goal_handle = None
            self.get_logger().warn("Subgoal rejected by nav2.")
            return

        self.navigation_active = True
        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut, result_token=token: self.result_callback(fut, result_token))

    def feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        remaining = feedback.distance_remaining
        if remaining > 0.0 and remaining + self.progress_min_delta < self.last_feedback_remaining:
            self.last_feedback_remaining = remaining
            self.last_progress_time = self.get_clock().now()

    def result_callback(self, future, token: int) -> None:
        if token != self.current_goal_token:
            return

        result = future.result()
        self.navigation_active = False
        self.active_goal_handle = None

        if result is None:
            self.get_logger().warn("No result returned from nav2 action.")
            return

        status = result.status
        if status == GoalStatus.STATUS_CANCELED and token in self.ignored_cancel_tokens:
            self.ignored_cancel_tokens.discard(token)
            return

        self.ignored_cancel_tokens.discard(token)
        if status == GoalStatus.STATUS_SUCCEEDED:
            return

        if status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
            robot_pose = self.get_robot_pose(log_warning=False)
            if robot_pose is not None:
                self.register_dead_end_zone(robot_pose, self.now_seconds())
                if self.current_plan is not None:
                    self.record_failed_plan_memory(robot_pose, self.current_plan)
            self.last_progress_time = self.get_clock().now() - Duration(seconds=self.progress_timeout + 1.0)

    def get_robot_pose(self, log_warning: bool) -> Optional[Pose2D]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.15),
            )
        except TransformException as ex:
            if log_warning:
                now = self.get_clock().now()
                if self.last_tf_warn_time is None or (now - self.last_tf_warn_time) > Duration(seconds=2.0):
                    self.get_logger().warn("TF lookup failed: %s" % ex)
                    self.last_tf_warn_time = now
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        return Pose2D(x=t.x, y=t.y, yaw=yaw_from_quaternion(q))

    def get_scan_clearance(self, heading: float) -> float:
        if self.latest_scan is None:
            return 0.0

        scan = self.latest_scan
        angle_min = float(scan.angle_min)
        angle_max = float(scan.angle_max)
        angle_inc = float(scan.angle_increment)
        ranges = scan.ranges

        if angle_inc <= 0.0 or len(ranges) == 0:
            return 0.0
        if heading < angle_min or heading > angle_max:
            return 0.0

        center = int((heading - angle_min) / angle_inc)
        values = []
        for idx in range(center - 2, center + 3):
            if 0 <= idx < len(ranges):
                r = ranges[idx]
                if math.isfinite(r):
                    values.append(float(r))
                elif math.isinf(r) and r > 0.0:
                    values.append(float(scan.range_max))

        if not values:
            return 0.0
        return min(values)

    def plan_rrt_path(self, robot_pose: Pose2D, obstacles: List[Tuple[float, float]]) -> Optional[RRTPlan]:
        if self.final_goal is None:
            return None

        guided_plan = self.plan_grid_path(robot_pose, obstacles)
        if guided_plan is not None:
            return guided_plan

        start = (robot_pose.x, robot_pose.y)
        fx = self.final_goal.pose.position.x
        fy = self.final_goal.pose.position.y
        goal = self.clip_goal_to_horizon(start, (fx, fy), self.planning_horizon)
        start_goal_dist = distance_xy(start[0], start[1], goal[0], goal[1])
        start_ignore_radius = max(self.subgoal_min_distance, 0.9 * self.collision_clearance)

        direct_penalty = self.segment_dead_end_cost(start, goal, start)
        if self.segment_collision_free(
            start,
            goal,
            obstacles,
            ignore_center=start,
            ignore_radius=start_ignore_radius,
        ) and direct_penalty <= self.dead_end_replan_cost:
            return RRTPlan(path=[start, goal], tree_edges=[(start, goal)])

        tree: List[RRTNode] = [RRTNode(x=start[0], y=start[1], parent=-1, cost=0.0)]
        tree_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        goal_index = -1
        best_goal_score = float("inf")
        best_frontier_idx = -1
        best_frontier_score = -float("inf")

        for _ in range(self.rrt_max_iterations):
            sample = self.sample_point(start, goal)
            nearest_idx = self.find_nearest(tree, sample)
            nearest = tree[nearest_idx]
            new_point = self.steer((nearest.x, nearest.y), sample, self.rrt_step_size)

            if distance_xy(nearest.x, nearest.y, new_point[0], new_point[1]) < 1e-3:
                continue
            if not self.segment_collision_free(
                (nearest.x, nearest.y),
                new_point,
                obstacles,
                ignore_center=start,
                ignore_radius=start_ignore_radius,
            ):
                continue

            edge_len = distance_xy(nearest.x, nearest.y, new_point[0], new_point[1])
            edge_dead_end_cost = self.segment_dead_end_cost((nearest.x, nearest.y), new_point, start)
            edge_breadcrumb_bonus = self.segment_breadcrumb_bonus((nearest.x, nearest.y), new_point)
            edge_experience_penalty = self.segment_experience_cost((nearest.x, nearest.y), new_point, start)
            goal_dist_parent = distance_xy(nearest.x, nearest.y, goal[0], goal[1])
            goal_dist_new = distance_xy(new_point[0], new_point[1], goal[0], goal[1])
            progress_penalty = max(0.0, goal_dist_new - goal_dist_parent) * self.goal_progress_penalty_weight
            new_cost = (
                nearest.cost
                + edge_len
                + self.dead_end_penalty_weight * edge_dead_end_cost
                + edge_experience_penalty
                + progress_penalty
                - self.breadcrumb_path_reuse_weight * edge_breadcrumb_bonus
            )

            tree.append(RRTNode(x=new_point[0], y=new_point[1], parent=nearest_idx, cost=new_cost))
            tree_edges.append(((nearest.x, nearest.y), new_point))
            new_idx = len(tree) - 1

            progress_gain = start_goal_dist - goal_dist_new
            dist_from_start = distance_xy(start[0], start[1], new_point[0], new_point[1])
            frontier_score = (
                1.7 * progress_gain
                + 0.35 * dist_from_start
                - 1.0 * self.point_dead_end_penalty(new_point[0], new_point[1], start)
                - self.point_experience_penalty(new_point[0], new_point[1], start)
            )
            if dist_from_start >= self.rrt_partial_min_length and frontier_score > best_frontier_score:
                best_frontier_score = frontier_score
                best_frontier_idx = new_idx

            if distance_xy(new_point[0], new_point[1], goal[0], goal[1]) <= self.rrt_goal_connect_dist:
                if self.segment_collision_free(
                    new_point,
                    goal,
                    obstacles,
                    ignore_center=start,
                    ignore_radius=start_ignore_radius,
                ):
                    goal_edge_len = distance_xy(new_point[0], new_point[1], goal[0], goal[1])
                    goal_edge_dead_end_cost = self.segment_dead_end_cost(new_point, goal, start)
                    goal_edge_breadcrumb_bonus = self.segment_breadcrumb_bonus(new_point, goal)
                    goal_edge_experience_penalty = self.segment_experience_cost(new_point, goal, start)
                    goal_cost = (
                        tree[new_idx].cost
                        + goal_edge_len
                        + self.dead_end_penalty_weight * goal_edge_dead_end_cost
                        + goal_edge_experience_penalty
                        - self.breadcrumb_path_reuse_weight * goal_edge_breadcrumb_bonus
                    )

                    tree.append(RRTNode(x=goal[0], y=goal[1], parent=new_idx, cost=goal_cost))
                    tree_edges.append((new_point, goal))
                    candidate_idx = len(tree) - 1
                    candidate_score = tree[candidate_idx].cost
                    if candidate_score < best_goal_score:
                        best_goal_score = candidate_score
                        goal_index = candidate_idx

        if goal_index < 0:
            if best_frontier_idx < 0:
                if len(tree) <= 1:
                    return None
                best_frontier_idx = max(
                    range(1, len(tree)),
                    key=lambda i: distance_xy(start[0], start[1], tree[i].x, tree[i].y),
                )

            best_dist = distance_xy(tree[best_frontier_idx].x, tree[best_frontier_idx].y, goal[0], goal[1])
            progress_gain = start_goal_dist - best_dist
            dist_from_start = distance_xy(start[0], start[1], tree[best_frontier_idx].x, tree[best_frontier_idx].y)
            if dist_from_start < self.rrt_partial_min_length and progress_gain < 0.02:
                return None
            goal_index = best_frontier_idx

        path = self.extract_path(tree, goal_index)
        if len(path) >= 3:
            path = self.shortcut_path(path, obstacles)

        return RRTPlan(path=path, tree_edges=tree_edges)

    def plan_grid_path(self, robot_pose: Pose2D, obstacles: List[Tuple[float, float]]) -> Optional[RRTPlan]:
        if not self.grid_planner_enabled or self.final_goal is None or self.latest_local_costmap is None:
            return None

        costmap = self.latest_local_costmap
        start_world = (robot_pose.x, robot_pose.y)
        goal_world = self.clip_goal_to_horizon(
            start_world,
            (self.final_goal.pose.position.x, self.final_goal.pose.position.y),
            self.planning_horizon,
        )

        start_cell = self.world_to_grid(costmap, start_world[0], start_world[1])
        goal_cell = self.world_to_grid(costmap, goal_world[0], goal_world[1])
        if start_cell is None or goal_cell is None:
            return None

        start_cell = self.find_nearest_free_cell(costmap, start_cell, max_radius=2)
        goal_cell = self.find_nearest_free_cell(costmap, goal_cell, max_radius=8)
        if start_cell is None or goal_cell is None:
            return None

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        expanded_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        closed = set()

        start_h = self.grid_heuristic(start_cell, goal_cell)
        best_cell = start_cell
        best_rank = start_h
        heapq.heappush(open_heap, (start_h, 0.0, start_cell))
        expansions = 0

        while open_heap and expansions < self.grid_max_expansions:
            _, g_curr, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1

            rank = self.grid_heuristic(current, goal_cell)
            if rank < best_rank:
                best_rank = rank
                best_cell = current

            if self.grid_reached_goal(current, goal_cell):
                best_cell = current
                break

            current_world = self.grid_to_world(costmap, current)
            for neighbor, step_cost in self.iter_grid_neighbors(costmap, current):
                if neighbor in closed:
                    continue

                neighbor_world = self.grid_to_world(costmap, neighbor)
                if not self.segment_collision_free(current_world, neighbor_world, obstacles):
                    continue

                occupancy_cost = self.grid_cell_cost(costmap, neighbor)
                dead_end_penalty = self.point_dead_end_penalty(neighbor_world[0], neighbor_world[1], start_world)
                experience_penalty = self.point_experience_penalty(neighbor_world[0], neighbor_world[1], start_world)
                tentative_g = (
                    g_curr
                    + step_cost
                    + self.grid_cost_scale * occupancy_cost
                    + self.grid_dead_end_scale * dead_end_penalty
                    + experience_penalty
                )

                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                expanded_edges.append((current_world, neighbor_world))
                heuristic = self.grid_heuristic(neighbor, goal_cell)
                f_score = tentative_g + self.grid_heuristic_weight * heuristic
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        if best_cell == start_cell and len(parent) == 0:
            return None

        path = self.reconstruct_grid_path(costmap, parent, start_cell, best_cell)
        if len(path) < 2:
            return None
        if len(path) >= 3:
            path = self.shortcut_path(path, obstacles)

        return RRTPlan(path=path, tree_edges=expanded_edges)

    def sample_point(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[float, float]:
        candidate = goal
        for _ in range(6):
            if self.rng.random() < self.rrt_goal_sample_rate:
                candidate = goal
            else:
                radius = self.planning_horizon * math.sqrt(self.rng.random())
                angle = self.rng.uniform(-math.pi, math.pi)
                candidate = (start[0] + radius * math.cos(angle), start[1] + radius * math.sin(angle))
            if self.sample_allowed(candidate, start):
                return candidate
        return candidate

    def sample_allowed(self, sample: Tuple[float, float], start: Tuple[float, float]) -> bool:
        if not self.dead_end_zones:
            return True

        if distance_xy(sample[0], sample[1], start[0], start[1]) <= self.dead_end_escape_allowance:
            return True

        penalty = self.point_dead_end_penalty(sample[0], sample[1], start)
        if penalty <= 0.0:
            return True

        reject_prob = min(0.85, self.dead_end_sample_reject_prob * min(1.0, penalty))
        return self.rng.random() >= reject_prob

    @staticmethod
    def find_nearest(tree: List[RRTNode], point: Tuple[float, float]) -> int:
        return min(range(len(tree)), key=lambda i: distance_xy(tree[i].x, tree[i].y, point[0], point[1]))

    @staticmethod
    def steer(src: Tuple[float, float], dst: Tuple[float, float], step: float) -> Tuple[float, float]:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        dist = math.hypot(dx, dy)
        if dist <= step:
            return dst
        scale = step / max(dist, 1e-6)
        return (src[0] + dx * scale, src[1] + dy * scale)

    def segment_collision_free(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        obstacles: List[Tuple[float, float]],
        ignore_center: Optional[Tuple[float, float]] = None,
        ignore_radius: float = 0.0,
    ) -> bool:
        if not obstacles and self.latest_local_costmap is None:
            return True

        seg_len = distance_xy(p0[0], p0[1], p1[0], p1[1])
        steps = max(2, int(seg_len / max(self.collision_check_resolution, 1e-3)) + 1)

        for i in range(steps):
            t = i / float(steps - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            if ignore_center is not None and ignore_radius > 1e-6:
                if distance_xy(x, y, ignore_center[0], ignore_center[1]) <= ignore_radius:
                    continue
            if self.point_in_collision(x, y, obstacles):
                return False

        return True

    def segment_dead_end_cost(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        start: Tuple[float, float],
    ) -> float:
        if not self.dead_end_zones:
            return 0.0

        seg_len = distance_xy(p0[0], p0[1], p1[0], p1[1])
        if seg_len <= 1e-6:
            return self.point_dead_end_penalty(p0[0], p0[1], start)

        step = max(0.12, self.collision_check_resolution * 1.5)
        steps = max(2, int(seg_len / step) + 1)
        acc = 0.0
        for i in range(steps):
            t = i / float(steps - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            acc += self.point_dead_end_penalty(x, y, start)

        return seg_len * (acc / float(steps))

    def segment_breadcrumb_bonus(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        if self.breadcrumb_path_reuse_weight <= 0.0 or len(self.breadcrumb_points) < 5:
            return 0.0

        seg_len = distance_xy(p0[0], p0[1], p1[0], p1[1])
        if seg_len <= 1e-6:
            return 0.0

        trail = list(self.breadcrumb_points)
        stride = max(1, len(trail) // 300)
        sampled = trail[::stride]
        if sampled[-1] != trail[-1]:
            sampled.append(trail[-1])

        step = max(0.14, self.collision_check_resolution * 2.0)
        steps = max(2, int(seg_len / step) + 1)
        acc = 0.0
        for i in range(steps):
            t = i / float(steps - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            nearest = min(distance_xy(x, y, bx, by) for bx, by in sampled)
            if nearest < 1.0:
                # Do not reward breadcrumb reuse inside known dead-end memory zones.
                dead_pen = self.point_dead_end_penalty(x, y, p0)
                reuse_scale = max(0.0, 1.0 - min(1.0, dead_pen))
                acc += (1.0 - nearest) * reuse_scale

        return seg_len * (acc / float(steps))

    def segment_experience_cost(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        start: Tuple[float, float],
    ) -> float:
        if (not self.path_visit_cells) and (not self.failed_branch_cells):
            return 0.0

        seg_len = distance_xy(p0[0], p0[1], p1[0], p1[1])
        if seg_len <= 1e-6:
            return self.point_experience_penalty(p0[0], p0[1], start)

        step = max(0.12, self.experience_resolution)
        steps = max(2, int(seg_len / step) + 1)
        acc = 0.0
        for i in range(steps):
            t = i / float(steps - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            acc += self.point_experience_penalty(x, y, start)
        return seg_len * (acc / float(steps))

    def point_experience_penalty(self, x: float, y: float, start: Tuple[float, float]) -> float:
        if not self.advanced_mode:
            return 0.0
        key = self.memory_key(x, y, self.experience_resolution)
        revisit_count = float(self.path_visit_cells.get(key, 0))
        failed_count = float(self.failed_branch_cells.get(key, 0.0))

        if revisit_count <= 0.0 and failed_count <= 0.0:
            return 0.0

        start_dist = distance_xy(x, y, start[0], start[1])
        if start_dist < self.experience_skip_start_distance:
            revisit_count *= 0.2
            failed_count *= 0.1

        penalty = (
            self.experience_revisit_penalty_weight * min(revisit_count, 8.0)
            + self.experience_fail_penalty_weight * min(failed_count, 8.0)
        )
        return min(self.experience_penalty_cap, penalty)

    def point_dead_end_penalty(self, x: float, y: float, start: Tuple[float, float]) -> float:
        if not self.advanced_mode:
            return 0.0
        if not self.dead_end_zones:
            return 0.0

        total = 0.0
        for zone in self.dead_end_zones:
            dist = distance_xy(x, y, zone.x, zone.y)
            if dist >= zone.radius:
                continue

            start_in_zone = distance_xy(start[0], start[1], zone.x, zone.y) < zone.radius
            if start_in_zone and distance_xy(x, y, start[0], start[1]) <= self.dead_end_escape_allowance:
                continue

            ratio = 1.0 - dist / max(zone.radius, 1e-6)
            total += ratio

        return total

    def point_in_collision(self, x: float, y: float, obstacles: List[Tuple[float, float]]) -> bool:
        if self.point_in_costmap_collision(x, y):
            return True
        if self.point_in_mission_obstacle_memory(x, y):
            return True

        r = self.collision_clearance
        r2 = r * r
        for ox, oy in obstacles:
            dx = abs(ox - x)
            if dx > r:
                continue
            dy = abs(oy - y)
            if dy > r:
                continue
            if dx * dx + dy * dy <= r2:
                return True
        return False

    def point_in_costmap_collision(self, x: float, y: float) -> bool:
        msg = self.latest_local_costmap
        if msg is None:
            return False

        info = msg.info
        if info.resolution <= 0.0 or info.width <= 0 or info.height <= 0:
            return False

        res = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        mx = int((x - origin_x) / res)
        my = int((y - origin_y) / res)
        width = int(info.width)
        height = int(info.height)

        check_radius = max(0.0, self.collision_clearance + self.costmap_clearance_padding)
        if check_radius <= 1e-6:
            if mx < 0 or my < 0 or mx >= width or my >= height:
                return self.costmap_unknown_is_obstacle
            idx = my * width + mx
            if idx < 0 or idx >= len(msg.data):
                return self.costmap_unknown_is_obstacle
            cost = int(msg.data[idx])
            if cost < 0:
                return self.costmap_unknown_is_obstacle
            return cost >= self.costmap_collision_threshold

        radius_cells = max(1, int(math.ceil(check_radius / max(res, 1e-6))))
        for ix in range(mx - radius_cells, mx + radius_cells + 1):
            for iy in range(my - radius_cells, my + radius_cells + 1):
                cx = origin_x + (ix + 0.5) * res
                cy = origin_y + (iy + 0.5) * res
                if distance_xy(x, y, cx, cy) > check_radius:
                    continue

                if ix < 0 or iy < 0 or ix >= width or iy >= height:
                    if self.costmap_unknown_is_obstacle:
                        return True
                    continue

                idx = iy * width + ix
                if idx < 0 or idx >= len(msg.data):
                    if self.costmap_unknown_is_obstacle:
                        return True
                    continue

                cost = int(msg.data[idx])
                blocked = self.costmap_unknown_is_obstacle if cost < 0 else (cost >= self.costmap_collision_threshold)
                if blocked:
                    return True

        return False

    def point_in_mission_obstacle_memory(self, x: float, y: float) -> bool:
        if not self.mission_obstacle_cells:
            return False

        res = self.mission_obstacle_resolution
        radius = int(math.ceil(self.mission_obstacle_block_radius / max(res, 1e-6)))
        key = self.memory_key(x, y, res)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                neighbor = (key[0] + dx, key[1] + dy)
                confidence = self.mission_obstacle_cells.get(neighbor, 0.0)
                if confidence < self.mission_obstacle_block_threshold:
                    continue
                mx = neighbor[0] * res
                my = neighbor[1] * res
                if distance_xy(x, y, mx, my) <= self.mission_obstacle_block_radius:
                    return True
        return False

    @staticmethod
    def world_to_grid(msg: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        info = msg.info
        if info.resolution <= 0.0:
            return None

        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        mx = int((x - origin_x) / float(info.resolution))
        my = int((y - origin_y) / float(info.resolution))
        if mx < 0 or my < 0 or mx >= int(info.width) or my >= int(info.height):
            return None
        return (mx, my)

    @staticmethod
    def grid_to_world(msg: OccupancyGrid, cell: Tuple[int, int]) -> Tuple[float, float]:
        info = msg.info
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        res = float(info.resolution)
        return (origin_x + (cell[0] + 0.5) * res, origin_y + (cell[1] + 0.5) * res)

    def grid_cell_blocked(self, msg: OccupancyGrid, cell: Tuple[int, int]) -> bool:
        idx = cell[1] * int(msg.info.width) + cell[0]
        if idx < 0 or idx >= len(msg.data):
            return True
        cost = int(msg.data[idx])
        if cost < 0:
            return self.costmap_unknown_is_obstacle
        return cost >= self.costmap_collision_threshold

    def grid_cell_cost(self, msg: OccupancyGrid, cell: Tuple[int, int]) -> float:
        idx = cell[1] * int(msg.info.width) + cell[0]
        if idx < 0 or idx >= len(msg.data):
            return 1.0
        cost = int(msg.data[idx])
        if cost < 0:
            return 1.0 if self.costmap_unknown_is_obstacle else 0.35
        return min(1.0, max(0.0, float(cost) / 100.0))

    def find_nearest_free_cell(
        self,
        msg: OccupancyGrid,
        cell: Tuple[int, int],
        max_radius: int,
    ) -> Optional[Tuple[int, int]]:
        if not self.grid_cell_blocked(msg, cell):
            return cell

        width = int(msg.info.width)
        height = int(msg.info.height)
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx = cell[0] + dx
                    ny = cell[1] + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    candidate = (nx, ny)
                    if not self.grid_cell_blocked(msg, candidate):
                        return candidate
        return None

    @staticmethod
    def grid_heuristic(cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    def grid_reached_goal(self, cell: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        return max(abs(goal[0] - cell[0]), abs(goal[1] - cell[1])) <= self.grid_goal_tolerance_cells

    def iter_grid_neighbors(
        self,
        msg: OccupancyGrid,
        cell: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], float]]:
        width = int(msg.info.width)
        height = int(msg.info.height)
        neighbors: List[Tuple[Tuple[int, int], float]] = []
        for dx, dy in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            nx = cell[0] + dx
            ny = cell[1] + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            candidate = (nx, ny)
            if self.grid_cell_blocked(msg, candidate):
                continue
            if dx != 0 and dy != 0:
                side_a = (cell[0] + dx, cell[1])
                side_b = (cell[0], cell[1] + dy)
                if self.grid_cell_blocked(msg, side_a) or self.grid_cell_blocked(msg, side_b):
                    continue
                step_cost = math.sqrt(2.0)
            else:
                step_cost = 1.0
            neighbors.append((candidate, step_cost))
        return neighbors

    def reconstruct_grid_path(
        self,
        msg: OccupancyGrid,
        parent: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        path_cells = [goal]
        current = goal
        while current != start:
            current = parent.get(current)
            if current is None:
                break
            path_cells.append(current)
        path_cells.reverse()
        return [self.grid_to_world(msg, cell) for cell in path_cells]

    def extract_path(self, tree: List[RRTNode], node_idx: int) -> List[Tuple[float, float]]:
        path: List[Tuple[float, float]] = []
        idx = node_idx
        while idx >= 0:
            node = tree[idx]
            path.append((node.x, node.y))
            idx = node.parent
        path.reverse()
        return path

    def shortcut_path(self, path: List[Tuple[float, float]], obstacles: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return path

        out = list(path)
        for _ in range(self.shortcut_iterations):
            if len(out) < 3:
                break
            i = self.rng.randint(0, len(out) - 3)
            j = self.rng.randint(i + 2, len(out) - 1)
            if self.segment_collision_free(out[i], out[j], obstacles):
                out = out[: i + 1] + out[j:]

        return out

    def align_path_to_robot(self, path: List[Tuple[float, float]], robot_pose: Pose2D) -> List[Tuple[float, float]]:
        if not path:
            return []
        if len(path) == 1:
            return [(robot_pose.x, robot_pose.y), path[0]]

        best_idx = min(range(len(path)), key=lambda i: distance_xy(robot_pose.x, robot_pose.y, path[i][0], path[i][1]))
        if best_idx >= len(path) - 1:
            return [(robot_pose.x, robot_pose.y), path[-1]]

        aligned = [(robot_pose.x, robot_pose.y)]
        aligned.extend(path[best_idx + 1 :])
        if len(aligned) == 1:
            aligned.append(path[-1])
        return aligned

    def path_still_valid(self, robot_pose: Pose2D, path: List[Tuple[float, float]], obstacles: List[Tuple[float, float]]) -> bool:
        aligned = self.align_path_to_robot(path, robot_pose)
        if len(aligned) < 2:
            return False

        check_segments = min(3, len(aligned) - 1)
        for i in range(check_segments):
            if not self.segment_collision_free(aligned[i], aligned[i + 1], obstacles):
                return False

        return True

    def path_runs_into_dead_end(self, robot_pose: Pose2D, path: List[Tuple[float, float]]) -> bool:
        if not self.dead_end_zones:
            return False

        aligned = self.align_path_to_robot(path, robot_pose)
        if len(aligned) < 2:
            return False

        start = (robot_pose.x, robot_pose.y)
        check_segments = min(4, len(aligned) - 1)
        cost = 0.0
        for i in range(check_segments):
            cost += self.segment_dead_end_cost(aligned[i], aligned[i + 1], start)
        return cost > self.dead_end_replan_cost

    def select_subgoal_from_path(
        self,
        path: List[Tuple[float, float]],
        robot_pose: Pose2D,
        front_clearance: float = 0.0,
    ) -> PoseStamped:
        aligned = self.align_path_to_robot(path, robot_pose)
        if len(aligned) < 2:
            return self.make_pose(robot_pose.x, robot_pose.y, robot_pose.yaw)

        lookahead = max(self.subgoal_lookahead, 0.35)
        if front_clearance > 0.0:
            lookahead = min(lookahead, max(0.35, 0.9 * front_clearance))
        acc = 0.0

        for i in range(1, len(aligned)):
            x0, y0 = aligned[i - 1]
            x1, y1 = aligned[i]
            seg = distance_xy(x0, y0, x1, y1)
            if acc + seg >= lookahead and seg > 1e-6:
                ratio = (lookahead - acc) / seg
                gx = x0 + ratio * (x1 - x0)
                gy = y0 + ratio * (y1 - y0)

                if i < len(aligned) - 1:
                    nx, ny = aligned[i + 1]
                else:
                    nx, ny = aligned[i]
                gyaw = math.atan2(ny - gy, nx - gx)
                return self.make_pose(gx, gy, gyaw)
            acc += seg

        gx, gy = aligned[-1]
        if self.final_goal is not None:
            fx = self.final_goal.pose.position.x
            fy = self.final_goal.pose.position.y
            gyaw = math.atan2(fy - gy, fx - gx)
        else:
            gyaw = robot_pose.yaw
        return self.make_pose(gx, gy, gyaw)

    @staticmethod
    def clip_goal_to_horizon(
        start: Tuple[float, float],
        goal: Tuple[float, float],
        horizon: float,
    ) -> Tuple[float, float]:
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dist = math.hypot(dx, dy)
        if dist <= horizon:
            return goal
        scale = horizon / max(dist, 1e-6)
        return (start[0] + dx * scale, start[1] + dy * scale)

    @staticmethod
    def memory_key(x: float, y: float, resolution: float) -> Tuple[int, int]:
        return (int(round(x / resolution)), int(round(y / resolution)))

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def record_mission_obstacle(self, x: float, y: float, increment: float) -> None:
        key = self.memory_key(x, y, self.mission_obstacle_resolution)
        if key not in self.mission_obstacle_cells and len(self.mission_obstacle_cells) >= self.mission_obstacle_max_cells:
            weakest = min(self.mission_obstacle_cells, key=self.mission_obstacle_cells.get)
            del self.mission_obstacle_cells[weakest]
        prev = self.mission_obstacle_cells.get(key, 0.0)
        new_val = max(0.0, prev + increment)
        if new_val <= 0.05:
            if key in self.mission_obstacle_cells:
                del self.mission_obstacle_cells[key]
                self.mission_memory_dirty = True
            return
        self.mission_obstacle_cells[key] = new_val
        if abs(new_val - prev) > 1e-6:
            self.mission_memory_dirty = True

    def update_mission_obstacle_ray(
        self,
        rx: float,
        ry: float,
        ox: float,
        oy: float,
        mark_hit: bool = True,
    ) -> None:
        ray_len = distance_xy(rx, ry, ox, oy)
        if ray_len <= 1e-4:
            return

        step = max(self.mission_obstacle_ray_step, self.mission_obstacle_resolution)
        steps = max(2, int(ray_len / step) + 1)

        for i in range(1, steps - 1):
            t = i / float(steps - 1)
            x = rx + t * (ox - rx)
            y = ry + t * (oy - ry)
            self.record_mission_obstacle(x, y, -self.mission_obstacle_clear_decrement)

        if mark_hit:
            self.record_mission_obstacle(ox, oy, self.mission_obstacle_hit_increment)

    def publish_mission_memory_cloud(self, force: bool = False) -> None:
        now = self.get_clock().now()
        if not force:
            if (now - self.last_memory_cloud_pub_time) < Duration(seconds=self.mission_memory_publish_interval):
                return
            if not self.mission_memory_dirty:
                return

        msg = PointCloud2()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = now.to_msg()
        msg.height = 1
        msg.is_bigendian = False
        msg.is_dense = True
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12

        strong_cells = [
            (cell, conf)
            for cell, conf in self.mission_obstacle_cells.items()
            if conf >= self.mission_obstacle_block_threshold
        ]
        if strong_cells:
            if len(strong_cells) > self.mission_memory_cloud_max_points:
                strong_cells = heapq.nlargest(self.mission_memory_cloud_max_points, strong_cells, key=lambda item: item[1])

            msg.width = len(strong_cells)
            msg.row_step = msg.point_step * msg.width
            data = bytearray(msg.row_step)
            res = self.mission_obstacle_resolution
            for i, (cell, _) in enumerate(strong_cells):
                px = float(cell[0] * res)
                py = float(cell[1] * res)
                struct.pack_into("<fff", data, i * msg.point_step, px, py, 0.06)
            msg.data = bytes(data)
        else:
            msg.width = 0
            msg.row_step = 0
            msg.data = b""

        self.mission_memory_cloud_pub.publish(msg)
        self.last_memory_cloud_pub_time = now
        self.mission_memory_dirty = False

    def record_plan_visit_memory(self, robot_pose: Pose2D, plan: RRTPlan) -> None:
        self.record_path_memory(robot_pose, plan.path, self.path_visit_cells, increment=1.0)

    def record_failed_plan_memory(self, robot_pose: Pose2D, plan: RRTPlan) -> None:
        if not self.advanced_mode:
            return
        self.record_path_memory(robot_pose, plan.path, self.failed_branch_cells, increment=1.0)

    def record_path_memory(
        self,
        robot_pose: Pose2D,
        path: List[Tuple[float, float]],
        memory: Dict[Tuple[int, int], float],
        increment: float,
    ) -> None:
        aligned = self.align_path_to_robot(path, robot_pose)
        if len(aligned) < 2:
            return

        total = 0.0
        step = max(0.10, self.experience_resolution * 0.8)
        max_cells = self.experience_max_cells

        for i in range(1, len(aligned)):
            p0 = aligned[i - 1]
            p1 = aligned[i]
            seg_len = distance_xy(p0[0], p0[1], p1[0], p1[1])
            if seg_len <= 1e-6:
                continue

            steps = max(2, int(seg_len / step) + 1)
            for j in range(1, steps):
                t = j / float(steps - 1)
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                total += seg_len / float(steps - 1)
                if total < self.experience_skip_start_distance:
                    continue

                key = self.memory_key(x, y, self.experience_resolution)
                if key not in memory and len(memory) >= max_cells:
                    return
                memory[key] = memory.get(key, 0.0) + increment

    def prune_obstacles(self, now_sec: float) -> None:
        cutoff = now_sec - self.scan_keep_time
        while self.obstacle_points and self.obstacle_points[0][2] < cutoff:
            self.obstacle_points.popleft()

    def prune_dead_end_zones(self, now_sec: float) -> None:
        if not self.advanced_mode:
            return
        if not self.dead_end_zones:
            return
        self.dead_end_zones = deque(zone for zone in self.dead_end_zones if zone.expire_sec > now_sec)

    def update_breadcrumbs(self, robot_pose: Pose2D) -> None:
        point = (robot_pose.x, robot_pose.y)
        if not self.breadcrumb_points:
            self.breadcrumb_points.append(point)
            self.publish_breadcrumb_visualization()
            return

        last_x, last_y = self.breadcrumb_points[-1]
        if distance_xy(last_x, last_y, point[0], point[1]) < self.breadcrumb_spacing:
            return

        self.breadcrumb_points.append(point)
        while len(self.breadcrumb_points) > self.breadcrumb_max_points:
            self.breadcrumb_points.popleft()

        self.publish_breadcrumb_visualization()

    def publish_breadcrumb_visualization(self) -> None:
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.breadcrumb_points:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.01
            p.pose.orientation = quaternion_from_yaw(0.0)
            path_msg.poses.append(p)
        self.breadcrumb_path_pub.publish(path_msg)

    def clear_breadcrumb_visualization(self) -> None:
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.breadcrumb_path_pub.publish(path_msg)

    def update_pose_history(self, now_sec: float, robot_pose: Pose2D) -> None:
        self.pose_history.append((robot_pose.x, robot_pose.y, now_sec))
        cutoff = now_sec - self.stuck_window_sec
        while self.pose_history and self.pose_history[0][2] < cutoff:
            self.pose_history.popleft()

    def is_robot_trapped(self) -> bool:
        if len(self.pose_history) < 6:
            return False

        xs = [p[0] for p in self.pose_history]
        ys = [p[1] for p in self.pose_history]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        max_radius = max(distance_xy(x, y, cx, cy) for x, y, _ in self.pose_history)
        return max_radius < self.stuck_radius

    def start_escape_mode(
        self,
        robot_pose: Pose2D,
        front_blocked: bool,
        obstacles: List[Tuple[float, float]],
        mark_blocked_branch: bool = False,
    ) -> None:
        self.register_dead_end_zone(robot_pose, self.now_seconds())
        if mark_blocked_branch:
            self.register_blocked_branch_zones(robot_pose, self.now_seconds())
        mode = "breadcrumb"
        subgoal = self.compute_memory_escape_subgoal(robot_pose, obstacles)
        if subgoal is None:
            mode = "scan"
            subgoal = self.compute_escape_subgoal(robot_pose, obstacles)
        if subgoal is None:
            return

        now = self.get_clock().now()
        self.escape_goal = subgoal
        self.escape_until = now + Duration(seconds=self.escape_commit_sec)
        self.escape_attempts += 1
        self.current_plan = None
        self.active_subgoal = None
        self.cancel_active_navigation()
        self.navigation_active = False
        self.front_blocked_since = None
        self.last_send_time = None
        self.last_progress_time = now
        self.plan_lock_until = now + Duration(seconds=self.escape_commit_sec)

        self.get_logger().warn(
            "Trap detected%s, start %s escape #%d to (%.2f, %.2f)."
            % (
                " with front blocked" if front_blocked else "",
                mode,
                self.escape_attempts,
                subgoal.pose.position.x,
                subgoal.pose.position.y,
            )
        )

    def register_dead_end_zone(self, robot_pose: Pose2D, now_sec: float) -> None:
        if not self.advanced_mode:
            return
        center_x = robot_pose.x
        center_y = robot_pose.y
        radius = self.dead_end_radius + 0.10 * min(self.escape_attempts, 4)
        expire_sec = now_sec + self.dead_end_memory_sec
        self.add_or_merge_dead_end_zone(center_x, center_y, radius, expire_sec)

    def add_or_merge_dead_end_zone(self, center_x: float, center_y: float, radius: float, expire_sec: float) -> None:
        kept: Deque[DeadEndZone] = deque()
        for zone in self.dead_end_zones:
            merge_dist = max(self.dead_end_merge_dist, 0.45 * (zone.radius + radius))
            if distance_xy(zone.x, zone.y, center_x, center_y) <= merge_dist:
                total_w = max(zone.radius, 1e-3) + max(radius, 1e-3)
                center_x = (center_x * radius + zone.x * zone.radius) / total_w
                center_y = (center_y * radius + zone.y * zone.radius) / total_w
                radius = max(radius, zone.radius)
                expire_sec = max(expire_sec, zone.expire_sec)
            else:
                kept.append(zone)

        kept.append(DeadEndZone(x=center_x, y=center_y, radius=radius, expire_sec=expire_sec))
        while len(kept) > self.dead_end_max_zones:
            kept.popleft()
        self.dead_end_zones = kept

    def register_blocked_branch_zones(self, robot_pose: Pose2D, now_sec: float) -> None:
        if not self.advanced_mode:
            return
        if len(self.breadcrumb_points) < 8:
            return

        trail = list(self.breadcrumb_points)
        nearest_idx = min(
            range(len(trail)),
            key=lambda i: distance_xy(robot_pose.x, robot_pose.y, trail[i][0], trail[i][1]),
        )
        if nearest_idx < 2:
            return

        expire_sec = now_sec + self.blocked_branch_memory_sec
        last_mark = (robot_pose.x, robot_pose.y)
        travelled = 0.0
        prev = trail[nearest_idx]
        added = 0
        for i in range(nearest_idx - 1, -1, -1):
            pt = trail[i]
            travelled += distance_xy(prev[0], prev[1], pt[0], pt[1])
            if travelled > self.blocked_branch_memory_distance:
                break
            prev = pt

            if distance_xy(robot_pose.x, robot_pose.y, pt[0], pt[1]) < self.blocked_branch_skip_distance:
                continue
            if distance_xy(last_mark[0], last_mark[1], pt[0], pt[1]) < self.blocked_branch_zone_step:
                continue

            self.add_or_merge_dead_end_zone(
                center_x=pt[0],
                center_y=pt[1],
                radius=self.blocked_branch_zone_radius,
                expire_sec=expire_sec,
            )
            last_mark = pt
            added += 1

        if added > 0:
            self.get_logger().info("Marked blocked branch memory zones: %d" % added)

    def compute_escape_subgoal(self, robot_pose: Pose2D, obstacles: List[Tuple[float, float]]) -> Optional[PoseStamped]:
        if self.latest_scan is None or self.final_goal is None:
            return None

        scan = self.latest_scan
        angle_min = float(scan.angle_min)
        angle_max = float(scan.angle_max)
        step = max(self.escape_heading_step, 1e-3)
        hmin = max(angle_min, -self.escape_heading_range)
        hmax = min(angle_max, self.escape_heading_range)
        if hmax <= hmin:
            return None

        fx = self.final_goal.pose.position.x
        fy = self.final_goal.pose.position.y
        goal_heading = wrap_to_pi(math.atan2(fy - robot_pose.y, fx - robot_pose.x) - robot_pose.yaw)

        best_heading = None
        best_clearance = 0.0
        best_score = -float("inf")

        h = hmin
        while h <= hmax + 1e-6:
            clearance = self.get_scan_clearance(h)
            if clearance >= self.escape_min_clearance:
                open_term = min(clearance / max(self.scan_max_range, 1e-3), 1.0)
                goal_term = math.cos(wrap_to_pi(h - goal_heading))
                score = self.escape_open_bias * open_term + self.escape_goal_bias * goal_term
                if score > best_score:
                    best_score = score
                    best_heading = h
                    best_clearance = clearance
            h += step

        if best_heading is None:
            return None

        travel = min(
            self.escape_distance + 0.12 * min(self.escape_attempts, 4),
            max(0.35, best_clearance - self.collision_clearance),
        )
        global_heading = wrap_to_pi(robot_pose.yaw + best_heading)
        gx = robot_pose.x + travel * math.cos(global_heading)
        gy = robot_pose.y + travel * math.sin(global_heading)

        if self.point_in_collision(gx, gy, obstacles):
            return None

        gyaw = math.atan2(fy - gy, fx - gx)
        return self.make_pose(gx, gy, gyaw)

    def compute_memory_escape_subgoal(self, robot_pose: Pose2D, obstacles: List[Tuple[float, float]]) -> Optional[PoseStamped]:
        if not self.advanced_mode:
            return None
        if len(self.breadcrumb_points) < 6:
            return None

        trail = list(self.breadcrumb_points)
        nearest_idx = min(
            range(len(trail)),
            key=lambda i: distance_xy(robot_pose.x, robot_pose.y, trail[i][0], trail[i][1]),
        )
        if nearest_idx < 2:
            return None

        start = (robot_pose.x, robot_pose.y)
        fx = self.final_goal.pose.position.x if self.final_goal is not None else robot_pose.x
        fy = self.final_goal.pose.position.y if self.final_goal is not None else robot_pose.y

        acc = 0.0
        best_idx = -1
        best_score = float("inf")
        retreat_limit = self.breadcrumb_retreat_distance * 2.5
        for i in range(nearest_idx - 1, -1, -1):
            acc += distance_xy(trail[i][0], trail[i][1], trail[i + 1][0], trail[i + 1][1])
            if acc < self.breadcrumb_min_goal_distance:
                continue
            if acc > retreat_limit:
                break

            gx, gy = trail[i]
            if self.point_in_collision(gx, gy, obstacles):
                continue
            if not self.segment_collision_free(start, (gx, gy), obstacles):
                continue
            if distance_xy(start[0], start[1], gx, gy) < self.subgoal_min_distance:
                continue

            retreat_error = abs(acc - self.breadcrumb_retreat_distance)
            dead_pen = self.point_dead_end_penalty(gx, gy, start)
            goal_dist = distance_xy(gx, gy, fx, fy)
            score = retreat_error + 0.7 * dead_pen + 0.08 * goal_dist
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            return None

        gx, gy = trail[best_idx]
        if best_idx > 0:
            nx, ny = trail[best_idx - 1]
        else:
            nx = gx + math.cos(robot_pose.yaw)
            ny = gy + math.sin(robot_pose.yaw)
        gyaw = math.atan2(ny - gy, nx - gx)
        return self.make_pose(gx, gy, gyaw)

    def publish_rrt_visualization(self, plan: RRTPlan) -> None:
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in plan.path:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.02
            p.pose.orientation = quaternion_from_yaw(0.0)
            path_msg.poses.append(p)
        self.rrt_path_pub.publish(path_msg)

        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 0.05
        marker.color.g = 0.75
        marker.color.b = 1.0
        marker.color.a = 0.55

        for (x0, y0), (x1, y1) in plan.tree_edges:
            p0 = Point(x=float(x0), y=float(y0), z=0.03)
            p1 = Point(x=float(x1), y=float(y1), z=0.03)
            marker.points.append(p0)
            marker.points.append(p1)

        self.rrt_tree_pub.publish(marker)

    def clear_rrt_visualization(self, clear_goal: bool) -> None:
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.rrt_path_pub.publish(path_msg)

        tree_delete = Marker()
        tree_delete.header.frame_id = self.global_frame
        tree_delete.header.stamp = self.get_clock().now().to_msg()
        tree_delete.ns = "rrt_tree"
        tree_delete.id = 0
        tree_delete.action = Marker.DELETE
        self.rrt_tree_pub.publish(tree_delete)

        if clear_goal:
            goal_delete = Marker()
            goal_delete.header.frame_id = self.global_frame
            goal_delete.header.stamp = self.get_clock().now().to_msg()
            goal_delete.ns = "rrt_goal"
            goal_delete.id = 0
            goal_delete.action = Marker.DELETE
            self.rrt_goal_marker_pub.publish(goal_delete)

    def publish_goal_marker(self, goal: PoseStamped) -> None:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = goal.pose
        marker.pose.position.z = 0.08
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.95
        self.rrt_goal_marker_pub.publish(marker)

    def make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(yaw)
        return pose

    def transform_pose_to_global(self, pose: PoseStamped) -> Optional[PoseStamped]:
        src_frame = pose.header.frame_id.strip() if pose.header.frame_id else self.global_frame
        if src_frame == self.global_frame:
            out = PoseStamped()
            out.header.frame_id = self.global_frame
            out.header.stamp = self.get_clock().now().to_msg()
            out.pose = pose.pose
            return out

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame,
                src_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as ex:
            # In mapless runs, RViz may still publish goals in "map" while no map->odom TF exists.
            # Fallback to treating the goal as already expressed in global_frame so RRT can still run.
            if src_frame in ("map", "/map", "odom", "/odom"):
                out = PoseStamped()
                out.header.frame_id = self.global_frame
                out.header.stamp = self.get_clock().now().to_msg()
                out.pose = pose.pose
                self.get_logger().warn(
                    "Goal frame '%s' unavailable (%s). Fallback: treat goal as '%s'."
                    % (src_frame, str(ex), self.global_frame)
                )
                return out
            return None

        tf_t = tf_msg.transform.translation
        tf_q = tf_msg.transform.rotation
        tf_yaw = yaw_from_quaternion(tf_q)

        px = pose.pose.position.x
        py = pose.pose.position.y
        pyaw = yaw_from_quaternion(pose.pose.orientation)

        gx = tf_t.x + math.cos(tf_yaw) * px - math.sin(tf_yaw) * py
        gy = tf_t.y + math.sin(tf_yaw) * px + math.cos(tf_yaw) * py
        gyaw = wrap_to_pi(tf_yaw + pyaw)

        return self.make_pose(gx, gy, gyaw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaplessGoalManager()
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
