"""Hierarchical RL global planner node (FAR-inspired global search).

This tactical node follows FAR-like ideas:
1) Try known-space global route first.
2) If unavailable, try attemptable route through unknown with penalty.
3) If still unavailable, select a frontier target and route there.
4) Keep memory of failed frontiers to avoid endless revisits.
5) Publish rolling local waypoint and a waypoint marker for RViz.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


Cell = tuple[int, int]


class HRLGlobalPlanner(Node):
    """Global planner + waypoint extractor for hierarchical DRL navigation."""

    def __init__(self) -> None:
        super().__init__("hrl_global_planner")
        self._declare_parameters()

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.local_waypoint_topic = str(self.get_parameter("local_waypoint_topic").value)
        self.global_path_topic = str(self.get_parameter("global_path_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.waypoint_publish_frame = str(self.get_parameter("waypoint_publish_frame").value)

        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.timer_period_sec = float(self.get_parameter("timer_period_sec").value)
        self.stuck_window_sec = float(self.get_parameter("stuck_window_sec").value)
        self.stuck_distance_threshold = float(self.get_parameter("stuck_distance_threshold").value)
        self.replan_cooldown_sec = float(self.get_parameter("replan_cooldown_sec").value)
        self.periodic_replan_sec = float(self.get_parameter("periodic_replan_sec").value)

        self.obstacle_threshold = int(self.get_parameter("occupancy_obstacle_threshold").value)
        self.obstacle_inflation_radius_m = float(
            self.get_parameter("obstacle_inflation_radius_m").value
        )
        self.allow_unknown = bool(self.get_parameter("allow_unknown").value)
        self.unknown_penalty = float(self.get_parameter("unknown_penalty").value)

        self.frontier_search_enabled = bool(self.get_parameter("frontier_search_enabled").value)
        self.frontier_sample_limit = int(self.get_parameter("frontier_sample_limit").value)
        self.frontier_top_k = int(self.get_parameter("frontier_top_k").value)
        self.frontier_goal_weight = float(self.get_parameter("frontier_goal_weight").value)
        self.frontier_start_weight = float(self.get_parameter("frontier_start_weight").value)
        self.frontier_min_distance = float(self.get_parameter("frontier_min_distance").value)
        self.frontier_gain_weight = float(self.get_parameter("frontier_gain_weight").value)
        self.frontier_min_separation = float(self.get_parameter("frontier_min_separation").value)
        self.frontier_revisit_weight = float(self.get_parameter("frontier_revisit_weight").value)
        self.frontier_goal_heading_weight = float(
            self.get_parameter("frontier_goal_heading_weight").value
        )
        self.frontier_goal_heading_min_cos = float(
            self.get_parameter("frontier_goal_heading_min_cos").value
        )

        self.frontier_fail_cooldown_sec = float(self.get_parameter("frontier_fail_cooldown_sec").value)
        self.frontier_fail_radius_cells = int(self.get_parameter("frontier_fail_radius_cells").value)
        self.frontier_stagnation_sec = float(self.get_parameter("frontier_stagnation_sec").value)
        self.frontier_progress_epsilon = float(self.get_parameter("frontier_progress_epsilon").value)
        self.frontier_fail_hard_threshold = int(
            self.get_parameter("frontier_fail_hard_threshold").value
        )
        self.frontier_fail_hard_radius_cells = int(
            self.get_parameter("frontier_fail_hard_radius_cells").value
        )

        self.path_smoothing_enabled = bool(self.get_parameter("path_smoothing_enabled").value)
        self.path_max_skip_cells = int(self.get_parameter("path_max_skip_cells").value)
        self.waypoint_reached_distance = float(self.get_parameter("waypoint_reached_distance").value)
        self.waypoint_hold_timeout_sec = float(
            self.get_parameter("waypoint_hold_timeout_sec").value
        )
        self.waypoint_min_distance = float(self.get_parameter("waypoint_min_distance").value)
        self.waypoint_max_distance = float(self.get_parameter("waypoint_max_distance").value)
        self.waypoint_goal_weight = float(self.get_parameter("waypoint_goal_weight").value)
        self.waypoint_heading_weight = float(self.get_parameter("waypoint_heading_weight").value)
        self.waypoint_revisit_weight = float(self.get_parameter("waypoint_revisit_weight").value)
        self.waypoint_clearance_weight = float(
            self.get_parameter("waypoint_clearance_weight").value
        )
        self.waypoint_clearance_radius_m = float(
            self.get_parameter("waypoint_clearance_radius_m").value
        )

        self.publish_waypoint_marker = bool(self.get_parameter("publish_waypoint_marker").value)
        self.waypoint_marker_topic = str(self.get_parameter("waypoint_marker_topic").value)
        self.waypoint_marker_scale = float(self.get_parameter("waypoint_marker_scale").value)
        self.publish_goal_direction_marker = bool(
            self.get_parameter("publish_goal_direction_marker").value
        )
        self.goal_direction_marker_topic = str(
            self.get_parameter("goal_direction_marker_topic").value
        )
        self.goal_direction_width = float(self.get_parameter("goal_direction_width").value)
        self.publish_deadzone_marker = bool(self.get_parameter("publish_deadzone_marker").value)
        self.deadzone_marker_topic = str(self.get_parameter("deadzone_marker_topic").value)
        self.deadzone_marker_scale = float(self.get_parameter("deadzone_marker_scale").value)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_grid: Optional[np.ndarray] = None
        self.obstacle_mask: Optional[np.ndarray] = None
        self.inflated_obstacle_mask: Optional[np.ndarray] = None
        self.unknown_mask: Optional[np.ndarray] = None
        self.visit_heatmap: Optional[np.ndarray] = None
        self.last_odom: Optional[Odometry] = None
        self.goal_in_map: Optional[PoseStamped] = None
        self.map_update_seq = 0
        self.last_plan_map_seq = -1

        self.path_world: list[tuple[float, float]] = []
        self.path_cells: list[Cell] = []
        self.last_plan_mode = "idle"
        self.path_progress_idx = 0
        self.current_waypoint_idx = 0
        self.current_waypoint_set_sec = -1e9

        self.active_frontier_cell: Optional[Cell] = None
        self.frontier_start_dist_cells = 0.0
        self.frontier_best_dist_cells = float("inf")
        self.frontier_track_start_sec = 0.0
        self.frontier_failures: dict[Cell, tuple[int, float]] = {}

        self.last_replan_sec = -1e9
        self.pose_history: Deque[tuple[float, float, float]] = deque()
        self._last_tf_warn_sec = -1e9

        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.waypoint_pub = self.create_publisher(PoseStamped, self.local_waypoint_topic, 10)
        self.path_pub = self.create_publisher(Path, self.global_path_topic, 1)
        self.waypoint_marker_pub = (
            self.create_publisher(Marker, self.waypoint_marker_topic, 1)
            if self.publish_waypoint_marker
            else None
        )
        self.goal_direction_marker_pub = (
            self.create_publisher(Marker, self.goal_direction_marker_topic, 1)
            if self.publish_goal_direction_marker
            else None
        )
        self.deadzone_marker_pub = (
            self.create_publisher(MarkerArray, self.deadzone_marker_topic, 1)
            if self.publish_deadzone_marker
            else None
        )
        self.create_timer(self.timer_period_sec, self._timer_callback)

        self.get_logger().info(
            "HRL global planner ready (FAR-inspired). Waiting for /goal_pose and /map ..."
        )

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by this node."""
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("local_waypoint_topic", "/hrl_local_waypoint")
        self.declare_parameter("global_path_topic", "/hrl_global_path")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("waypoint_publish_frame", "odom")

        self.declare_parameter("lookahead_distance", 1.0)
        self.declare_parameter("timer_period_sec", 0.5)
        self.declare_parameter("stuck_window_sec", 6.0)
        self.declare_parameter("stuck_distance_threshold", 0.08)
        self.declare_parameter("replan_cooldown_sec", 6.0)
        self.declare_parameter("periodic_replan_sec", 1.0)

        self.declare_parameter("occupancy_obstacle_threshold", 65)
        self.declare_parameter("obstacle_inflation_radius_m", 0.22)
        self.declare_parameter("allow_unknown", True)
        self.declare_parameter("unknown_penalty", 2.5)

        self.declare_parameter("frontier_search_enabled", True)
        self.declare_parameter("frontier_sample_limit", 1500)
        self.declare_parameter("frontier_top_k", 40)
        self.declare_parameter("frontier_goal_weight", 1.0)
        self.declare_parameter("frontier_start_weight", 0.55)
        self.declare_parameter("frontier_min_distance", 1.0)
        self.declare_parameter("frontier_gain_weight", 0.7)
        self.declare_parameter("frontier_min_separation", 0.9)
        self.declare_parameter("frontier_revisit_weight", 1.8)
        self.declare_parameter("frontier_goal_heading_weight", 1.2)
        self.declare_parameter("frontier_goal_heading_min_cos", -0.25)

        self.declare_parameter("frontier_fail_cooldown_sec", 90.0)
        self.declare_parameter("frontier_fail_radius_cells", 8)
        self.declare_parameter("frontier_stagnation_sec", 6.0)
        self.declare_parameter("frontier_progress_epsilon", 0.45)
        self.declare_parameter("frontier_fail_hard_threshold", 3)
        self.declare_parameter("frontier_fail_hard_radius_cells", 10)

        self.declare_parameter("path_smoothing_enabled", True)
        self.declare_parameter("path_max_skip_cells", 24)
        self.declare_parameter("waypoint_reached_distance", 0.45)
        self.declare_parameter("waypoint_hold_timeout_sec", 2.0)
        self.declare_parameter("waypoint_min_distance", 0.7)
        self.declare_parameter("waypoint_max_distance", 3.0)
        self.declare_parameter("waypoint_goal_weight", 2.8)
        self.declare_parameter("waypoint_heading_weight", 0.8)
        self.declare_parameter("waypoint_revisit_weight", 1.5)
        self.declare_parameter("waypoint_clearance_weight", 1.3)
        self.declare_parameter("waypoint_clearance_radius_m", 0.8)

        self.declare_parameter("publish_waypoint_marker", True)
        self.declare_parameter("waypoint_marker_topic", "/hrl_waypoint_marker")
        self.declare_parameter("waypoint_marker_scale", 0.22)
        self.declare_parameter("publish_goal_direction_marker", True)
        self.declare_parameter("goal_direction_marker_topic", "/hrl_goal_direction_marker")
        self.declare_parameter("goal_direction_width", 0.06)
        self.declare_parameter("publish_deadzone_marker", True)
        self.declare_parameter("deadzone_marker_topic", "/hrl_deadzone_markers")
        self.declare_parameter("deadzone_marker_scale", 0.28)

    def _map_callback(self, msg: OccupancyGrid) -> None:
        """Cache latest occupancy grid."""
        self.map_msg = msg
        self.map_update_seq += 1
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width > 0 and height > 0:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
            obstacle_mask = grid >= self.obstacle_threshold
            inflation_cells = self._meter_to_cells(self.obstacle_inflation_radius_m)
            inflation_cells_i = int(max(0, round(inflation_cells)))
            inflated = self._inflate_mask(obstacle_mask, inflation_cells_i)

            self.map_grid = grid
            self.obstacle_mask = obstacle_mask
            self.inflated_obstacle_mask = inflated
            self.unknown_mask = grid < 0

            if (self.visit_heatmap is None) or (self.visit_heatmap.shape != grid.shape):
                self.visit_heatmap = np.zeros(grid.shape, dtype=np.float32)
        if not self.map_frame:
            self.map_frame = msg.header.frame_id or "map"

    def _odom_callback(self, msg: Odometry) -> None:
        """Cache latest odometry."""
        self.last_odom = msg

    def _goal_callback(self, msg: PoseStamped) -> None:
        """Receive final goal and trigger immediate global replan."""
        goal_map = self._transform_pose(msg, target_frame=self.map_frame)
        if goal_map is None:
            self.get_logger().warn(
                f"Cannot transform goal from {msg.header.frame_id} to {self.map_frame}."
            )
            return
        self.goal_in_map = goal_map
        self.path_world = []
        self.path_cells = []
        self.path_progress_idx = 0
        self.current_waypoint_idx = 0
        self.current_waypoint_set_sec = -1e9
        self.active_frontier_cell = None
        self.pose_history.clear()
        self._cleanup_frontier_failures(self._now_sec())
        self.get_logger().info(
            f"Final goal received: x={goal_map.pose.position.x:.2f}, y={goal_map.pose.position.y:.2f}. Replanning ..."
        )
        self._replan(force=True)

    def _timer_callback(self) -> None:
        """Periodic update: monitor progress, replan if needed, publish lookahead waypoint."""
        if self.map_msg is None or self.goal_in_map is None or self.last_odom is None:
            return

        robot_pose_map = self._robot_pose_in_map()
        if robot_pose_map is None:
            return

        now = self._now_sec()
        rx = float(robot_pose_map.pose.position.x)
        ry = float(robot_pose_map.pose.position.y)
        self._update_pose_history(now, rx, ry)
        self._cleanup_frontier_failures(now)

        robot_cell = self._world_to_map_clamped(rx, ry)
        if robot_cell is not None:
            self._record_visit(robot_cell)
        self._publish_goal_direction_marker(rx, ry)
        self._publish_deadzone_markers()

        should_replan = False
        force_replan = False
        if not self.path_world:
            should_replan = True
            force_replan = True
        elif (now - self.last_replan_sec) >= self.periodic_replan_sec:
            # Refresh only when map changed and route mode really needs it.
            map_changed = self.map_update_seq > self.last_plan_map_seq
            if map_changed and ("attemptable" in self.last_plan_mode):
                should_replan = True
            elif (
                map_changed
                and ("frontier" in self.last_plan_mode)
                and (robot_cell is not None)
                and (self.active_frontier_cell is not None)
            ):
                # Keep current frontier stable; only switch after reaching it.
                frontier_dist = self._heuristic(robot_cell, self.active_frontier_cell)
                reached_cells = max(
                    2.0,
                    self._meter_to_cells(max(0.8, 1.5 * self.waypoint_reached_distance)),
                )
                if frontier_dist <= reached_cells:
                    should_replan = True
                    force_replan = True
        elif (robot_cell is not None) and self._frontier_should_abort(now, robot_cell):
            should_replan = True
            force_replan = True
            if self.active_frontier_cell is not None:
                self._mark_frontier_failed(self.active_frontier_cell, now, reason="stagnation")
            self.get_logger().warn("Frontier stagnation detected. Switching to new frontier.")
        elif self._is_stuck(now) and (now - self.last_replan_sec) >= self.replan_cooldown_sec:
            should_replan = True
            force_replan = True
            if self.active_frontier_cell is not None:
                self._mark_frontier_failed(self.active_frontier_cell, now, reason="stuck")
            self.get_logger().warn("Recovery triggered: robot appears stuck, forcing global replan.")

        if should_replan:
            replanned = self._replan(force=force_replan)
            if (not replanned) and (not self.path_world):
                return

        waypoint = self._extract_best_waypoint(robot_pose_map)
        if waypoint is None:
            if not self._replan(force=True):
                return
            waypoint = self._extract_best_waypoint(robot_pose_map)
            if waypoint is None:
                return

        publish_wp = waypoint
        if self.waypoint_publish_frame and (waypoint.header.frame_id != self.waypoint_publish_frame):
            tf_wp = self._transform_pose(waypoint, self.waypoint_publish_frame)
            if tf_wp is None:
                # Skip this cycle rather than sending a stale-frame waypoint.
                return
            publish_wp = tf_wp
        self.waypoint_pub.publish(publish_wp)
        self._publish_waypoint_marker(publish_wp)

    def _replan(self, force: bool = False) -> bool:
        """Compute global path with FAR-inspired known/attemptable/frontier logic."""
        if self.map_msg is None or self.goal_in_map is None or self.last_odom is None:
            return False

        now = self._now_sec()
        if (not force) and (now - self.last_replan_sec) < self.replan_cooldown_sec:
            return bool(self.path_world)

        robot_pose_map = self._robot_pose_in_map()
        if robot_pose_map is None:
            return False

        start_xy = (
            float(robot_pose_map.pose.position.x),
            float(robot_pose_map.pose.position.y),
        )
        goal_xy = (
            float(self.goal_in_map.pose.position.x),
            float(self.goal_in_map.pose.position.y),
        )
        start_cell = self._world_to_map(*start_xy)
        raw_goal_cell = self._world_to_map(*goal_xy)
        goal_cell = raw_goal_cell
        goal_outside_map = raw_goal_cell is None
        if start_cell is None:
            start_cell = self._world_to_map_clamped(*start_xy)
        if goal_cell is None:
            goal_cell = self._world_to_map_clamped(*goal_xy)
        if start_cell is None or goal_cell is None:
            self.get_logger().warn("Start/goal is outside map bounds. Cannot plan.")
            return False

        start_cell = self._nearest_navigable_cell(start_cell, allow_unknown=False, max_radius=8)
        goal_cell = self._nearest_navigable_cell(goal_cell, allow_unknown=False, max_radius=20)
        if start_cell is None and self.allow_unknown:
            start_clamped = self._world_to_map_clamped(*start_xy)
            if start_clamped is not None:
                start_cell = self._nearest_navigable_cell(start_clamped, allow_unknown=True, max_radius=8)
        if goal_cell is None and self.allow_unknown:
            goal_clamped = self._world_to_map_clamped(*goal_xy)
            if goal_clamped is not None:
                goal_cell = self._nearest_navigable_cell(goal_clamped, allow_unknown=True, max_radius=20)
        if start_cell is None or goal_cell is None:
            self.get_logger().warn("Cannot find valid start/goal cell for planning.")
            return False

        path_cells, mode, frontier_cell = self._plan_far_like(
            start_cell,
            goal_cell,
            frontier_first=goal_outside_map,
            now=now,
        )
        if not path_cells:
            self.path_cells = []
            self.path_world = []
            self.last_plan_mode = "failed"
            self.active_frontier_cell = None
            self.get_logger().warn("Global planner failed: no known/attemptable/frontier route found.")
            return False

        if self.path_smoothing_enabled:
            allow_unknown_in_mode = ("attemptable" in mode)
            path_cells = self._smooth_path_cells(path_cells, allow_unknown=allow_unknown_in_mode)

        self.path_cells = path_cells
        self.path_world = [self._map_to_world(mx, my) for (mx, my) in path_cells]
        self.last_plan_mode = mode
        self.path_progress_idx = 0
        self.current_waypoint_idx = 0
        self.current_waypoint_set_sec = -1e9
        self.last_replan_sec = now
        self.last_plan_map_seq = self.map_update_seq

        self.active_frontier_cell = frontier_cell
        if frontier_cell is not None:
            self.frontier_start_dist_cells = self._heuristic(start_cell, frontier_cell)
            self.frontier_best_dist_cells = self.frontier_start_dist_cells
            self.frontier_track_start_sec = now
        else:
            self.frontier_start_dist_cells = 0.0
            self.frontier_best_dist_cells = float("inf")
            self.frontier_track_start_sec = 0.0

        self._publish_debug_path()
        self.get_logger().info(
            f"Global path planned with {len(self.path_world)} points. mode={mode}"
        )
        return True

    def _plan_far_like(
        self,
        start_cell: Cell,
        goal_cell: Cell,
        frontier_first: bool = False,
        now: float = 0.0,
    ) -> tuple[list[Cell], str, Optional[Cell]]:
        """FAR-inspired planning policy: known -> attemptable -> frontier."""

        def _frontier_stage() -> tuple[list[Cell], str, Optional[Cell]]:
            if not self.frontier_search_enabled:
                return [], "failed", None
            frontier_cells = self._extract_frontier_cells(start_cell, goal_cell, now)
            if not frontier_cells:
                return [], "failed", None
            for frontier_cell in frontier_cells[: self.frontier_top_k]:
                path_known = self._astar_cells(start_cell, frontier_cell, allow_unknown=False)
                if path_known:
                    self.get_logger().info(
                        f"Frontier target selected (known): cell={frontier_cell} len={len(path_known)}"
                    )
                    return path_known, "frontier_known", frontier_cell
            if self.allow_unknown:
                for frontier_cell in frontier_cells[: self.frontier_top_k]:
                    path_attempt = self._astar_cells(start_cell, frontier_cell, allow_unknown=True)
                    if path_attempt:
                        self.get_logger().info(
                            f"Frontier target selected (attemptable): cell={frontier_cell} len={len(path_attempt)}"
                        )
                        return path_attempt, "frontier_attemptable", frontier_cell
            return [], "failed", None

        if frontier_first:
            frontier_path, frontier_mode, frontier_cell = _frontier_stage()
            if frontier_path:
                return frontier_path, frontier_mode, frontier_cell

        known_path = self._astar_cells(start_cell, goal_cell, allow_unknown=False)
        if known_path:
            return known_path, "known", None

        if self.allow_unknown:
            attempt_path = self._astar_cells(start_cell, goal_cell, allow_unknown=True)
            if attempt_path:
                return attempt_path, "attemptable", None

        if not frontier_first:
            frontier_path, frontier_mode, frontier_cell = _frontier_stage()
            if frontier_path:
                return frontier_path, frontier_mode, frontier_cell

        return [], "failed", None

    def _astar_cells(self, start_cell: Cell, goal_cell: Cell, allow_unknown: bool) -> list[Cell]:
        """Run A* search on occupancy grid between two cells."""
        if self.map_msg is None:
            return []
        if self._is_blocked(start_cell[0], start_cell[1], allow_unknown):
            return []
        if self._is_blocked(goal_cell[0], goal_cell[1], allow_unknown):
            return []

        open_heap: list[tuple[float, float, Cell]] = []
        heapq.heappush(open_heap, (self._heuristic(start_cell, goal_cell), 0.0, start_cell))
        came_from: dict[Cell, Cell] = {}
        g_score: dict[Cell, float] = {start_cell: 0.0}
        visited: set[Cell] = set()

        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                if not self._in_bounds(nx, ny):
                    continue
                if self._is_blocked(nx, ny, allow_unknown):
                    continue
                step_cost = math.hypot(dx, dy) + self._unknown_cost(nx, ny, allow_unknown)
                tentative_g = current_g + step_cost
                node = (nx, ny)
                if tentative_g >= g_score.get(node, float("inf")):
                    continue
                came_from[node] = current
                g_score[node] = tentative_g
                f_score = tentative_g + self._heuristic(node, goal_cell)
                heapq.heappush(open_heap, (f_score, tentative_g, node))
        return []

    def _extract_frontier_cells(self, start_cell: Cell, goal_cell: Cell, now: float) -> list[Cell]:
        """Extract and rank reachable frontier (known-free adjacent to unknown) cells."""
        if self.map_msg is None or self.map_grid is None:
            return []
        width = int(self.map_msg.info.width)
        height = int(self.map_msg.info.height)
        if width <= 2 or height <= 2:
            return []

        grid = self.map_grid
        unknown = self.unknown_mask if self.unknown_mask is not None else (grid < 0)
        inflated = (
            self.inflated_obstacle_mask
            if self.inflated_obstacle_mask is not None
            else (grid >= self.obstacle_threshold)
        )
        free_known = (grid >= 0) & (~inflated)

        reachable_known = self._reachable_known_mask(start_cell, free_known)
        if not np.any(reachable_known):
            return []

        unknown_neighbor = np.zeros_like(reachable_known, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                src_y0 = max(0, -dy)
                src_y1 = height - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = width - max(0, dx)
                dst_y0 = max(0, dy)
                dst_y1 = height - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = width - max(0, -dx)
                unknown_neighbor[dst_y0:dst_y1, dst_x0:dst_x1] |= unknown[src_y0:src_y1, src_x0:src_x1]

        frontier_mask = reachable_known & unknown_neighbor
        frontier_idx = np.argwhere(frontier_mask)  # [y, x]
        if frontier_idx.size == 0:
            return []

        if len(frontier_idx) > self.frontier_sample_limit:
            step = int(math.ceil(len(frontier_idx) / float(self.frontier_sample_limit)))
            frontier_idx = frontier_idx[::max(1, step)]

        resolution = float(self.map_msg.info.resolution)
        min_dist_cells = int(max(1.0, self.frontier_min_distance / max(resolution, 1e-3)))
        min_sep_cells = int(max(1.0, self.frontier_min_separation / max(resolution, 1e-3)))
        diag = math.hypot(width, height)
        diag = max(diag, 1.0)
        goal_vec = np.array(
            [float(goal_cell[0] - start_cell[0]), float(goal_cell[1] - start_cell[1])],
            dtype=np.float32,
        )
        goal_norm = float(np.linalg.norm(goal_vec))
        if goal_norm > 1e-6:
            goal_unit = goal_vec / goal_norm
        else:
            goal_unit = np.array([1.0, 0.0], dtype=np.float32)

        scored: list[tuple[float, Cell]] = []
        blocked_scored: list[tuple[float, Cell]] = []
        for y, x in frontier_idx:
            cell = (int(x), int(y))
            dist_start = self._heuristic(start_cell, cell)
            if dist_start < float(min_dist_cells):
                continue

            frontier_vec = np.array(
                [float(cell[0] - start_cell[0]), float(cell[1] - start_cell[1])],
                dtype=np.float32,
            )
            frontier_norm = float(np.linalg.norm(frontier_vec))
            if frontier_norm > 1e-6:
                frontier_unit = frontier_vec / frontier_norm
                heading_cos = float(np.dot(frontier_unit, goal_unit))
            else:
                heading_cos = 0.0

            # info gain: unknown cells in local 5x5 patch
            x0 = max(0, cell[0] - 2)
            x1 = min(width, cell[0] + 3)
            y0 = max(0, cell[1] - 2)
            y1 = min(height, cell[1] + 3)
            gain = float(np.count_nonzero(unknown[y0:y1, x0:x1]))

            dist_goal = self._heuristic(cell, goal_cell)
            fail_penalty = self._frontier_failure_penalty(cell, now)
            revisit_penalty = self._frontier_revisit_penalty(cell)
            heading_penalty = 0.0
            if heading_cos < self.frontier_goal_heading_min_cos:
                heading_penalty = (self.frontier_goal_heading_min_cos - heading_cos) * 1.5
            score = (
                self.frontier_goal_weight * (dist_goal / diag)
                + self.frontier_start_weight * (dist_start / diag)
                - self.frontier_gain_weight * min(gain / 25.0, 1.0)
                - self.frontier_goal_heading_weight * heading_cos
                + heading_penalty
                + fail_penalty
                + revisit_penalty
            )
            if self._frontier_is_hard_blocked(cell, now):
                blocked_scored.append((score + 4.0, cell))
                continue
            scored.append((score, cell))

        # If all candidates are hard-blocked, allow the best blocked options as fallback.
        if (not scored) and blocked_scored:
            scored = blocked_scored

        scored.sort(key=lambda t: t[0])
        return self._select_diverse_frontiers(scored, min_sep_cells)

    def _reachable_known_mask(self, start_cell: Cell, free_known: np.ndarray) -> np.ndarray:
        """BFS reachable mask on known-free cells from start."""
        height, width = free_known.shape
        mask = np.zeros_like(free_known, dtype=bool)
        sx, sy = start_cell
        if not (0 <= sx < width and 0 <= sy < height):
            return mask
        if not free_known[sy, sx]:
            return mask

        q: Deque[Cell] = deque()
        q.append((sx, sy))
        mask[sy, sx] = True

        while q:
            x, y = q.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if mask[ny, nx] or (not free_known[ny, nx]):
                    continue
                mask[ny, nx] = True
                q.append((nx, ny))
        return mask

    def _extract_best_waypoint(self, robot_pose_map: PoseStamped) -> Optional[PoseStamped]:
        """Select the most useful waypoint every cycle using map-aware scoring."""
        if not self.path_world:
            return None

        now_sec = self._now_sec()
        robot_x = float(robot_pose_map.pose.position.x)
        robot_y = float(robot_pose_map.pose.position.y)
        robot_yaw = self._yaw_from_quaternion(robot_pose_map.pose.orientation)
        robot = np.array([robot_x, robot_y], dtype=np.float32)
        points = np.asarray(self.path_world, dtype=np.float32)

        d2 = np.sum((points - robot[None, :]) ** 2, axis=1)
        nearest_idx = int(np.argmin(d2))
        self.path_progress_idx = max(self.path_progress_idx, nearest_idx)

        map_diag_m = self._map_diag_meters()
        min_arc = max(self.waypoint_min_distance, 0.25)
        max_arc = max(self.waypoint_max_distance, min_arc + 0.2, self.lookahead_distance)

        candidates: list[tuple[float, int]] = []
        accum = 0.0
        start_idx = min(self.path_progress_idx, len(self.path_world) - 1)
        for i in range(start_idx, len(self.path_world) - 1):
            x0, y0 = self.path_world[i]
            x1, y1 = self.path_world[i + 1]
            accum += math.hypot(x1 - x0, y1 - y0)
            if accum < min_arc:
                continue
            if accum > max_arc:
                break

            idx = i + 1
            wx, wy = self.path_world[idx]
            dist_goal = (
                math.hypot(
                    wx - float(self.goal_in_map.pose.position.x),
                    wy - float(self.goal_in_map.pose.position.y),
                )
                if self.goal_in_map is not None
                else 0.0
            )
            goal_term = -self.waypoint_goal_weight * (dist_goal / max(map_diag_m, 1e-3))

            heading = math.atan2(wy - robot_y, wx - robot_x)
            heading_raw = self._normalize_angle(heading - robot_yaw)
            # Avoid selecting far waypoints that require near U-turns.
            if abs(heading_raw) > 2.7 and accum > (min_arc + 0.4):
                continue
            heading_err = abs(heading_raw) / math.pi
            heading_term = -self.waypoint_heading_weight * heading_err

            cell = self._world_to_map(wx, wy)
            revisit = self._waypoint_revisit_penalty(cell) if cell is not None else 0.0
            revisit_term = -self.waypoint_revisit_weight * revisit

            clearance = self._waypoint_clearance_score(cell) if cell is not None else 0.0
            clearance_term = self.waypoint_clearance_weight * clearance

            progress_term = 0.9 * float(np.clip(accum / max_arc, 0.0, 1.0))
            score = goal_term + heading_term + revisit_term + clearance_term + progress_term
            candidates.append((score, idx))

        if not candidates:
            chosen_idx = min(max(self.path_progress_idx + 1, 0), len(self.path_world) - 1)
        else:
            candidates.sort(key=lambda t: t[0], reverse=True)
            chosen_idx = candidates[0][1]

        # If a waypoint stays too long and still not reached, advance one step.
        if chosen_idx == self.current_waypoint_idx:
            if self.current_waypoint_set_sec < -1e8:
                self.current_waypoint_set_sec = now_sec
            elif (now_sec - self.current_waypoint_set_sec) > self.waypoint_hold_timeout_sec:
                if self.current_waypoint_idx < (len(self.path_world) - 1):
                    cw_x, cw_y = self.path_world[self.current_waypoint_idx]
                    if math.hypot(cw_x - robot_x, cw_y - robot_y) > (1.15 * self.waypoint_reached_distance):
                        chosen_idx = self.current_waypoint_idx + 1
        else:
            self.current_waypoint_set_sec = now_sec

        self.current_waypoint_idx = int(np.clip(chosen_idx, 0, len(self.path_world) - 1))
        wx, wy = self.path_world[self.current_waypoint_idx]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = wx
        msg.pose.position.y = wy
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def _waypoint_revisit_penalty(self, cell: Cell) -> float:
        """Return normalized revisit penalty [0,1] based on local visitation heat."""
        if self.visit_heatmap is None:
            return 0.0
        x, y = cell
        h, w = self.visit_heatmap.shape
        x0 = max(0, x - 2)
        x1 = min(w, x + 3)
        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        heat = float(np.sum(self.visit_heatmap[y0:y1, x0:x1]))
        return float(np.clip(heat / 20.0, 0.0, 1.0))

    def _waypoint_clearance_score(self, cell: Cell) -> float:
        """Return clearance score [0,1], higher when farther from inflated obstacles."""
        if (self.inflated_obstacle_mask is None) or (self.map_msg is None):
            return 0.0
        x, y = cell
        mask = self.inflated_obstacle_mask
        h, w = mask.shape
        radius_cells = max(1, int(round(self._meter_to_cells(self.waypoint_clearance_radius_m))))
        x0 = max(0, x - radius_cells)
        x1 = min(w, x + radius_cells + 1)
        y0 = max(0, y - radius_cells)
        y1 = min(h, y + radius_cells + 1)
        local = mask[y0:y1, x0:x1]
        if local.size == 0:
            return 0.0
        obs = np.argwhere(local)
        if obs.size == 0:
            return 1.0
        cx = x - x0
        cy = y - y0
        dx = obs[:, 1].astype(np.float32) - float(cx)
        dy = obs[:, 0].astype(np.float32) - float(cy)
        min_dist = float(np.min(np.sqrt(dx * dx + dy * dy)))
        return float(np.clip(min_dist / max(float(radius_cells), 1.0), 0.0, 1.0))

    def _map_diag_meters(self) -> float:
        """Return map diagonal length in meters."""
        if self.map_msg is None:
            return 1.0
        w = float(self.map_msg.info.width)
        h = float(self.map_msg.info.height)
        res = float(self.map_msg.info.resolution)
        return max(1e-3, math.hypot(w, h) * res)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_debug_path(self) -> None:
        """Publish current global path for visualization/debug."""
        if not self.path_world:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for x, y in self.path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_waypoint_marker(self, waypoint: PoseStamped) -> None:
        """Publish the active local waypoint marker for RViz."""
        if (not self.publish_waypoint_marker) or (self.waypoint_marker_pub is None):
            return
        marker = Marker()
        marker.header = waypoint.header
        marker.ns = "hrl_local_waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = waypoint.pose
        marker.scale.x = self.waypoint_marker_scale
        marker.scale.y = self.waypoint_marker_scale
        marker.scale.z = self.waypoint_marker_scale * 0.4
        marker.color.a = 0.95
        marker.color.r = 0.10
        marker.color.g = 0.95
        marker.color.b = 0.10
        marker.lifetime = Duration(seconds=1.0).to_msg()
        self.waypoint_marker_pub.publish(marker)

    def _publish_goal_direction_marker(self, robot_x: float, robot_y: float) -> None:
        """Publish a blue line indicating coarse direction toward final goal."""
        if (
            (not self.publish_goal_direction_marker)
            or (self.goal_direction_marker_pub is None)
            or (self.goal_in_map is None)
        ):
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = "hrl_goal_direction"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.goal_direction_width
        marker.color.a = 0.95
        marker.color.r = 0.10
        marker.color.g = 0.35
        marker.color.b = 1.00
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=1.0).to_msg()

        p0 = Point()
        p0.x = float(robot_x)
        p0.y = float(robot_y)
        p0.z = 0.05
        p1 = Point()
        p1.x = float(self.goal_in_map.pose.position.x)
        p1.y = float(self.goal_in_map.pose.position.y)
        p1.z = 0.05
        marker.points = [p0, p1]
        self.goal_direction_marker_pub.publish(marker)

    def _publish_deadzone_markers(self) -> None:
        """Publish yellow markers at repeatedly failed (dead-zone) frontier areas."""
        if (not self.publish_deadzone_marker) or (self.deadzone_marker_pub is None):
            return
        now = self._now_sec()

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        for cell, (count, t_fail) in self.frontier_failures.items():
            age = now - t_fail
            if age > (self.frontier_fail_cooldown_sec * 3.0):
                continue
            wx, wy = self._map_to_world(cell[0], cell[1])
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.map_frame
            marker.ns = "hrl_deadzone"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.04
            marker.pose.orientation.w = 1.0

            growth = min(1.0 + 0.35 * float(count), 2.0)
            marker.scale.x = self.deadzone_marker_scale * growth
            marker.scale.y = self.deadzone_marker_scale * growth
            marker.scale.z = self.deadzone_marker_scale * 0.35
            marker.color.a = 0.65
            marker.color.r = 1.00
            marker.color.g = 1.00
            marker.color.b = 0.10
            marker.lifetime = Duration(seconds=2.0).to_msg()
            marker_array.markers.append(marker)

        self.deadzone_marker_pub.publish(marker_array)

    def _robot_pose_in_map(self) -> Optional[PoseStamped]:
        """Return robot pose transformed into map frame."""
        if self.last_odom is None:
            return None
        pose = PoseStamped()
        pose.header = self.last_odom.header
        pose.pose = self.last_odom.pose.pose
        if not pose.header.frame_id:
            pose.header.frame_id = "odom"
        return self._transform_pose(pose, self.map_frame)

    def _transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform pose into target frame using TF."""
        if pose.header.frame_id == target_frame:
            return pose
        # Query latest TF and apply transform explicitly to avoid stale timestamp mismatch.
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

    def _update_pose_history(self, now_sec: float, x: float, y: float) -> None:
        """Store recent positions for stuck detection."""
        self.pose_history.append((now_sec, x, y))
        min_t = now_sec - self.stuck_window_sec
        while self.pose_history and self.pose_history[0][0] < min_t:
            self.pose_history.popleft()

    def _is_stuck(self, now_sec: float) -> bool:
        """Return True if robot appears static/looping in the window."""
        if len(self.pose_history) < 3:
            return False
        dt = self.pose_history[-1][0] - self.pose_history[0][0]
        if dt < self.stuck_window_sec * 0.9:
            return False

        start = self.pose_history[0]
        end = self.pose_history[-1]
        displacement = math.hypot(end[1] - start[1], end[2] - start[2])

        travel = 0.0
        prev = self.pose_history[0]
        for cur in list(self.pose_history)[1:]:
            travel += math.hypot(cur[1] - prev[1], cur[2] - prev[2])
            prev = cur

        static_like = displacement < self.stuck_distance_threshold and travel < 0.18
        loop_like = travel > 1.0 and displacement < 0.12
        return static_like or loop_like

    def _frontier_should_abort(self, now_sec: float, robot_cell: Cell) -> bool:
        """Abort current frontier target when progress stagnates."""
        if self.active_frontier_cell is None:
            return False
        if self.frontier_track_start_sec <= 0.0:
            return False

        cur_dist = self._heuristic(robot_cell, self.active_frontier_cell)
        if cur_dist < self.frontier_best_dist_cells:
            self.frontier_best_dist_cells = cur_dist

        if (now_sec - self.frontier_track_start_sec) < self.frontier_stagnation_sec:
            return False

        progress_cells = self.frontier_start_dist_cells - self.frontier_best_dist_cells
        eps_cells = self._meter_to_cells(self.frontier_progress_epsilon)
        return progress_cells < max(1.0, eps_cells)

    def _mark_frontier_failed(self, cell: Cell, now_sec: float, reason: str) -> None:
        """Record failed frontier to reduce repeated looping on same area."""
        count, _ = self.frontier_failures.get(cell, (0, now_sec))
        self.frontier_failures[cell] = (count + 1, now_sec)
        self.get_logger().warn(
            f"Frontier marked failed ({reason}): cell={cell} fail_count={count + 1}"
        )

    def _cleanup_frontier_failures(self, now_sec: float) -> None:
        """Drop stale frontier failure memories."""
        if not self.frontier_failures:
            return
        stale_keys = [
            c
            for c, (_, t) in self.frontier_failures.items()
            if (now_sec - t) > (self.frontier_fail_cooldown_sec * 3.0)
        ]
        for c in stale_keys:
            self.frontier_failures.pop(c, None)

    def _frontier_failure_penalty(self, cell: Cell, now_sec: float) -> float:
        """Penalty term for candidate frontiers near recently failed frontiers."""
        if not self.frontier_failures:
            return 0.0
        penalty = 0.0
        radius = max(1, self.frontier_fail_radius_cells)
        for failed_cell, (count, t_fail) in self.frontier_failures.items():
            age = now_sec - t_fail
            if age > self.frontier_fail_cooldown_sec:
                continue
            dist = self._heuristic(cell, failed_cell)
            if dist > float(radius):
                continue
            locality = (float(radius) - dist + 1.0) / (float(radius) + 1.0)
            freshness = 1.0 - (age / max(self.frontier_fail_cooldown_sec, 1e-3))
            penalty += float(count) * 0.8 * locality * freshness
        return penalty

    def _frontier_is_hard_blocked(self, cell: Cell, now_sec: float) -> bool:
        """Return True when frontier should be temporarily excluded as dead-zone."""
        if not self.frontier_failures:
            return False
        hard_radius = max(1, self.frontier_fail_hard_radius_cells)
        for failed_cell, (count, t_fail) in self.frontier_failures.items():
            if count < self.frontier_fail_hard_threshold:
                continue
            age = now_sec - t_fail
            if age > self.frontier_fail_cooldown_sec:
                continue
            if self._heuristic(cell, failed_cell) <= float(hard_radius):
                return True
        return False

    def _frontier_revisit_penalty(self, cell: Cell) -> float:
        """Penalty term for frequently revisited areas near the frontier."""
        if self.visit_heatmap is None:
            return 0.0
        x, y = cell
        h, w = self.visit_heatmap.shape
        x0 = max(0, x - 3)
        x1 = min(w, x + 4)
        y0 = max(0, y - 3)
        y1 = min(h, y + 4)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        revisit_sum = float(np.sum(self.visit_heatmap[y0:y1, x0:x1]))
        revisit_norm = min(revisit_sum / 20.0, 2.5)
        return self.frontier_revisit_weight * revisit_norm

    def _select_diverse_frontiers(
        self,
        scored_cells: list[tuple[float, Cell]],
        min_separation_cells: int,
    ) -> list[Cell]:
        """Select well-separated frontier candidates from sorted scored list."""
        if not scored_cells:
            return []
        selected: list[Cell] = []
        sep_sq = float(max(1, min_separation_cells) ** 2)
        for _, cell in scored_cells:
            keep = True
            for sx, sy in selected:
                dx = float(cell[0] - sx)
                dy = float(cell[1] - sy)
                if (dx * dx + dy * dy) <= sep_sq:
                    keep = False
                    break
            if keep:
                selected.append(cell)
            if len(selected) >= self.frontier_top_k * 3:
                break
        return selected

    def _record_visit(self, robot_cell: Cell) -> None:
        """Update robot visitation heatmap to discourage repeated frontier choices."""
        if self.visit_heatmap is None:
            return
        x, y = robot_cell
        h, w = self.visit_heatmap.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return
        self.visit_heatmap *= 0.995
        self.visit_heatmap[y, x] = min(self.visit_heatmap[y, x] + 1.0, 15.0)

    def _meter_to_cells(self, meter: float) -> float:
        """Convert meter distance to map cells."""
        if self.map_msg is None:
            return meter * 10.0
        res = max(float(self.map_msg.info.resolution), 1e-3)
        return meter / res

    @staticmethod
    def _inflate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
        """Inflate occupied cells by disk radius in cell units."""
        if radius_cells <= 0:
            return mask.copy()
        h, w = mask.shape
        inflated = mask.copy()
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) > (radius_cells * radius_cells):
                    continue
                src_y0 = max(0, -dy)
                src_y1 = h - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = w - max(0, dx)
                dst_y0 = max(0, dy)
                dst_y1 = h - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = w - max(0, -dx)
                inflated[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
        return inflated

    def _world_to_map(self, x: float, y: float) -> Optional[Cell]:
        """Convert world coordinates to occupancy-grid cell."""
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        mx = int(math.floor((x - info.origin.position.x) / info.resolution))
        my = int(math.floor((y - info.origin.position.y) / info.resolution))
        if not self._in_bounds(mx, my):
            return None
        return mx, my

    def _world_to_map_clamped(self, x: float, y: float) -> Optional[Cell]:
        """Convert world coordinates to map cell with clamping to map bounds."""
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        width = int(info.width)
        height = int(info.height)
        if width <= 0 or height <= 0:
            return None
        fx = (x - info.origin.position.x) / info.resolution
        fy = (y - info.origin.position.y) / info.resolution
        mx = int(np.clip(math.floor(fx), 0, width - 1))
        my = int(np.clip(math.floor(fy), 0, height - 1))
        return mx, my

    def _map_to_world(self, mx: int, my: int) -> tuple[float, float]:
        """Convert occupancy-grid cell to world coordinate (cell center)."""
        assert self.map_msg is not None
        info = self.map_msg.info
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return float(wx), float(wy)

    def _in_bounds(self, mx: int, my: int) -> bool:
        """Check whether cell index is inside map."""
        if self.map_msg is None:
            return False
        return 0 <= mx < int(self.map_msg.info.width) and 0 <= my < int(self.map_msg.info.height)

    def _cell_value(self, mx: int, my: int) -> int:
        """Return occupancy value at map cell."""
        if self.map_grid is not None:
            return int(self.map_grid[my, mx])
        assert self.map_msg is not None
        width = int(self.map_msg.info.width)
        return int(self.map_msg.data[my * width + mx])

    def _is_blocked(self, mx: int, my: int, allow_unknown: bool) -> bool:
        """Check whether a cell is blocked under a given unknown-space policy."""
        if not self._in_bounds(mx, my):
            return True
        if self.inflated_obstacle_mask is not None and bool(self.inflated_obstacle_mask[my, mx]):
            return True
        val = self._cell_value(mx, my)
        if val < 0:
            return not allow_unknown
        if self.inflated_obstacle_mask is None and val >= self.obstacle_threshold:
            return True
        return False

    def _unknown_cost(self, mx: int, my: int, allow_unknown: bool) -> float:
        """Penalty for traversing unknown cells in attemptable mode."""
        if not allow_unknown:
            return 0.0
        if self.unknown_mask is not None:
            if bool(self.unknown_mask[my, mx]):
                return self.unknown_penalty
            return 0.0
        val = self._cell_value(mx, my)
        if val < 0:
            return self.unknown_penalty
        return 0.0

    def _nearest_navigable_cell(self, cell: Cell, allow_unknown: bool, max_radius: int = 20) -> Optional[Cell]:
        """Find nearest non-blocked cell around given cell."""
        mx0, my0 = cell
        if self._in_bounds(mx0, my0) and (not self._is_blocked(mx0, my0, allow_unknown)):
            return cell
        for r in range(1, max_radius + 1):
            x0 = mx0 - r
            x1 = mx0 + r
            y0 = my0 - r
            y1 = my0 + r
            for mx in range(x0, x1 + 1):
                for my in (y0, y1):
                    if self._in_bounds(mx, my) and (not self._is_blocked(mx, my, allow_unknown)):
                        return (mx, my)
            for my in range(y0 + 1, y1):
                for mx in (x0, x1):
                    if self._in_bounds(mx, my) and (not self._is_blocked(mx, my, allow_unknown)):
                        return (mx, my)
        return None

    def _smooth_path_cells(self, path_cells: list[Cell], allow_unknown: bool) -> list[Cell]:
        """Greedy line-of-sight path smoothing."""
        if len(path_cells) <= 2:
            return path_cells
        smoothed = [path_cells[0]]
        i = 0
        max_skip = max(2, self.path_max_skip_cells)
        while i < len(path_cells) - 1:
            j = min(len(path_cells) - 1, i + max_skip)
            while j > i + 1:
                if self._line_traversable(path_cells[i], path_cells[j], allow_unknown):
                    break
                j -= 1
            if path_cells[j] != smoothed[-1]:
                smoothed.append(path_cells[j])
            i = j
        return smoothed

    def _line_traversable(self, a: Cell, b: Cell, allow_unknown: bool) -> bool:
        """Check traversability along line segment in map cells."""
        for mx, my in self._bresenham_cells(a, b):
            if not self._in_bounds(mx, my):
                return False
            if self._is_blocked(mx, my, allow_unknown):
                return False
        return True

    @staticmethod
    def _bresenham_cells(a: Cell, b: Cell) -> list[Cell]:
        """Return rasterized cells between two points using Bresenham."""
        x0, y0 = a
        x1, y1 = b
        cells: list[Cell] = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    @staticmethod
    def _heuristic(a: Cell, b: Cell) -> float:
        """Euclidean heuristic for A*."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: dict[Cell, Cell], goal_node: Cell) -> list[Cell]:
        """Backtrack A* parent links from goal to start."""
        path = [goal_node]
        current = goal_node
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _now_sec(self) -> float:
        """Return current ROS time in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args: Optional[list[str]] = None) -> None:
    """Entrypoint for ROS 2 execution."""
    rclpy.init(args=args)
    node = HRLGlobalPlanner()
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
