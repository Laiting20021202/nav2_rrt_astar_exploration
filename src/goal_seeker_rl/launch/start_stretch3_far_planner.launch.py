"""Launch FAR-style global planner + RL local driver for a real Stretch3 ROS graph."""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start only navigation nodes; robot drivers, map, odom, scan, and TF are provided externally."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_topic = LaunchConfiguration("map_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    goal_topic = LaunchConfiguration("goal_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    base_frame = LaunchConfiguration("base_frame")
    map_frame = LaunchConfiguration("map_frame")
    model_dir = LaunchConfiguration("model_dir")
    model_name = LaunchConfiguration("model_name")
    model_path = LaunchConfiguration("model_path")

    default_workspace = os.environ.get("RL_BASE_WS", "/workspace")
    default_model_dir = os.environ.get("RL_BASE_MODEL_DIR", os.path.join(default_workspace, "navigation_model"))
    default_model_path = PathJoinSubstitution([model_dir, model_name])

    hrl_global_planner = Node(
        package="goal_seeker_rl",
        executable="hrl_global_planner",
        name="hrl_global_planner",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "lookahead_distance": LaunchConfiguration("lookahead_distance"),
                "goal_reached_distance": LaunchConfiguration("planner_goal_reached_distance"),
                "timer_period_sec": 0.25,
                "stuck_window_sec": 6.0,
                "stuck_distance_threshold": 0.08,
                "replan_cooldown_sec": 1.0,
                "periodic_replan_sec": LaunchConfiguration("planner_periodic_replan_sec"),
                "occupancy_obstacle_threshold": 65,
                "obstacle_inflation_radius_m": LaunchConfiguration("planner_obstacle_inflation_radius_m"),
                "allow_unknown": True,
                "unknown_penalty": 2.5,
                "frontier_search_enabled": True,
                "frontier_sample_limit": 1200,
                "frontier_top_k": 40,
                "frontier_goal_weight": 2.4,
                "frontier_start_weight": 0.55,
                "frontier_min_distance": 2.0,
                "frontier_min_separation": 1.5,
                "frontier_goal_heading_weight": LaunchConfiguration("planner_frontier_goal_heading_weight"),
                "frontier_goal_heading_min_cos": 0.12,
                "frontier_goal_progress_min_m": 0.9,
                "path_smoothing_enabled": True,
                "path_max_skip_cells": 24,
                "waypoint_reached_distance": LaunchConfiguration("planner_waypoint_reached_distance"),
                "waypoint_hold_timeout_sec": LaunchConfiguration("planner_waypoint_hold_timeout_sec"),
                "waypoint_min_distance": LaunchConfiguration("planner_waypoint_min_distance"),
                "waypoint_max_distance": LaunchConfiguration("planner_waypoint_max_distance"),
                "waypoint_goal_weight": LaunchConfiguration("planner_waypoint_goal_weight"),
                "waypoint_heading_weight": LaunchConfiguration("planner_waypoint_heading_weight"),
                "waypoint_revisit_weight": LaunchConfiguration("planner_waypoint_revisit_weight"),
                "waypoint_clearance_weight": LaunchConfiguration("planner_waypoint_clearance_weight"),
                "waypoint_clearance_radius_m": LaunchConfiguration("planner_waypoint_clearance_radius_m"),
                "frontier_revisit_weight": 3.2,
                "frontier_fail_radius_cells": LaunchConfiguration("planner_frontier_fail_radius_cells"),
                "frontier_fail_hard_threshold": LaunchConfiguration("planner_frontier_fail_hard_threshold"),
                "frontier_fail_hard_radius_cells": LaunchConfiguration("planner_frontier_fail_hard_radius_cells"),
                "frontier_stagnation_sec": 7.0,
                "map_topic": map_topic,
                "odom_topic": odom_topic,
                "goal_topic": goal_topic,
                "goal_active_topic": "/hrl_goal_active",
                "local_waypoint_topic": "/hrl_local_waypoint",
                "global_path_topic": "/hrl_global_path",
                "map_frame": map_frame,
                "waypoint_publish_frame": map_frame,
                "publish_waypoint_marker": True,
                "waypoint_marker_topic": "/hrl_waypoint_marker",
                "publish_goal_direction_marker": True,
                "goal_direction_marker_topic": "/hrl_goal_direction_marker",
                "goal_direction_width": 0.07,
                "publish_deadzone_marker": True,
                "deadzone_marker_topic": "/hrl_deadzone_markers",
                "deadzone_marker_scale": 0.30,
            }
        ],
    )

    rl_local_driver = Node(
        package="goal_seeker_rl",
        executable="rl_local_driver",
        name="rl_local_driver",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "scan_topic": scan_topic,
                "waypoint_topic": "/hrl_local_waypoint",
                "goal_active_topic": "/hrl_goal_active",
                "cmd_vel_topic": cmd_vel_topic,
                "base_frame": base_frame,
                "scan_samples": LaunchConfiguration("local_scan_samples"),
                "lidar_max_range": LaunchConfiguration("lidar_max_range"),
                "control_rate_hz": LaunchConfiguration("local_control_rate_hz"),
                "linear_speed_max": LaunchConfiguration("linear_speed_max"),
                "angular_speed_max": LaunchConfiguration("angular_speed_max"),
                "goal_active_timeout_sec": LaunchConfiguration("local_goal_active_timeout"),
                "waypoint_timeout_sec": 2.0,
                "obstacle_stop_distance": 0.18,
                "obstacle_slow_distance": LaunchConfiguration("local_obstacle_slow_distance"),
                "obstacle_hard_stop_distance": LaunchConfiguration("local_obstacle_hard_stop_distance"),
                "emergency_stop_distance": LaunchConfiguration("local_emergency_stop_distance"),
                "proactive_avoid_distance": LaunchConfiguration("local_proactive_avoid_distance"),
                "side_guard_distance": LaunchConfiguration("local_side_guard_distance"),
                "avoid_turn_boost": LaunchConfiguration("local_avoid_turn_boost"),
                "waypoint_close_distance": LaunchConfiguration("local_waypoint_close_distance"),
                "waypoint_stop_distance": LaunchConfiguration("local_waypoint_stop_distance"),
                "turn_in_place_angle": LaunchConfiguration("local_turn_in_place_angle"),
                "orbit_break_angle": LaunchConfiguration("local_orbit_break_angle"),
                "policy_blend_far": LaunchConfiguration("policy_blend_far"),
                "policy_blend_near_obstacle": LaunchConfiguration("policy_blend_near_obstacle"),
                "policy_blend_obstacle_distance": LaunchConfiguration("policy_blend_obstacle_distance"),
                "enable_local_escape": LaunchConfiguration("enable_local_escape"),
                "policy_source": LaunchConfiguration("local_policy_source"),
                "model_path": model_path,
                "network_variant": LaunchConfiguration("local_network_variant"),
                "hidden_dim": LaunchConfiguration("local_hidden_dim"),
                "append_prev_action_to_state": LaunchConfiguration("local_append_prev_action"),
                "model_strict": False,
                "policy_max_goal_distance": LaunchConfiguration("local_policy_max_goal_distance"),
                "publish_rl_path": True,
                "rl_path_topic": "/rl_model_path",
                "rl_path_frame": base_frame,
                "rl_path_horizon_steps": 28,
                "rl_path_dt_sec": 0.20,
                "lookaround_enabled": True,
                "lookaround_front_distance": 0.75,
                "lookaround_clear_distance": 1.10,
                "lookaround_turn_speed": 0.55,
                "lookaround_duration_sec": 1.30,
                "lookaround_min_duration_sec": 0.45,
                "lookaround_cooldown_sec": 0.80,
                "waiting_scan_creep_enabled": False,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("map_topic", default_value="/map"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("goal_topic", default_value="/goal_pose"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("model_dir", default_value=default_model_dir),
            DeclareLaunchArgument("model_name", default_value="td3_latest.pth"),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("lookahead_distance", default_value="1.5"),
            DeclareLaunchArgument("local_scan_samples", default_value="40"),
            DeclareLaunchArgument("lidar_max_range", default_value="3.5"),
            DeclareLaunchArgument("local_control_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("linear_speed_max", default_value="0.12"),
            DeclareLaunchArgument("angular_speed_max", default_value="1.2"),
            DeclareLaunchArgument("local_policy_source", default_value="td3"),
            DeclareLaunchArgument("local_network_variant", default_value="reference"),
            DeclareLaunchArgument("local_hidden_dim", default_value="512"),
            DeclareLaunchArgument("local_append_prev_action", default_value="true"),
            DeclareLaunchArgument("local_policy_max_goal_distance", default_value="5.94"),
            DeclareLaunchArgument("planner_waypoint_reached_distance", default_value="0.50"),
            DeclareLaunchArgument("planner_goal_reached_distance", default_value="0.35"),
            DeclareLaunchArgument("planner_periodic_replan_sec", default_value="2.0"),
            DeclareLaunchArgument("planner_waypoint_hold_timeout_sec", default_value="1.6"),
            DeclareLaunchArgument("planner_waypoint_min_distance", default_value="0.8"),
            DeclareLaunchArgument("planner_waypoint_max_distance", default_value="2.4"),
            DeclareLaunchArgument("planner_waypoint_goal_weight", default_value="2.8"),
            DeclareLaunchArgument("planner_waypoint_heading_weight", default_value="0.85"),
            DeclareLaunchArgument("planner_waypoint_revisit_weight", default_value="1.6"),
            DeclareLaunchArgument("planner_waypoint_clearance_weight", default_value="1.2"),
            DeclareLaunchArgument("planner_waypoint_clearance_radius_m", default_value="0.75"),
            DeclareLaunchArgument("local_waypoint_close_distance", default_value="0.50"),
            DeclareLaunchArgument("local_waypoint_stop_distance", default_value="0.18"),
            DeclareLaunchArgument("planner_obstacle_inflation_radius_m", default_value="0.24"),
            DeclareLaunchArgument("planner_frontier_fail_radius_cells", default_value="30"),
            DeclareLaunchArgument("planner_frontier_fail_hard_threshold", default_value="2"),
            DeclareLaunchArgument("planner_frontier_fail_hard_radius_cells", default_value="30"),
            DeclareLaunchArgument("planner_frontier_goal_heading_weight", default_value="2.4"),
            DeclareLaunchArgument("policy_blend_far", default_value="0.04"),
            DeclareLaunchArgument("policy_blend_near_obstacle", default_value="0.12"),
            DeclareLaunchArgument("policy_blend_obstacle_distance", default_value="1.00"),
            DeclareLaunchArgument("enable_local_escape", default_value="true"),
            DeclareLaunchArgument("local_obstacle_slow_distance", default_value="0.45"),
            DeclareLaunchArgument("local_obstacle_hard_stop_distance", default_value="0.25"),
            DeclareLaunchArgument("local_side_guard_distance", default_value="0.26"),
            DeclareLaunchArgument("local_avoid_turn_boost", default_value="0.45"),
            DeclareLaunchArgument("local_turn_in_place_angle", default_value="1.25"),
            DeclareLaunchArgument("local_orbit_break_angle", default_value="1.65"),
            DeclareLaunchArgument("local_goal_active_timeout", default_value="1.2"),
            DeclareLaunchArgument("local_emergency_stop_distance", default_value="0.24"),
            DeclareLaunchArgument("local_proactive_avoid_distance", default_value="0.85"),
            hrl_global_planner,
            rl_local_driver,
        ]
    )
