"""Launch full hierarchical RL navigation stack in Gazebo."""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Start Gazebo + SLAM + HRL global planner + RL local driver + RViz."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")
    use_joint_state_publisher = LaunchConfiguration("use_joint_state_publisher")
    world = LaunchConfiguration("world")
    slam_params_file = LaunchConfiguration("slam_params_file")
    lookahead_distance = LaunchConfiguration("lookahead_distance")
    local_scan_samples = LaunchConfiguration("local_scan_samples")
    linear_speed_max = LaunchConfiguration("linear_speed_max")
    angular_speed_max = LaunchConfiguration("angular_speed_max")
    local_policy_source = LaunchConfiguration("local_policy_source")
    local_model_dir = LaunchConfiguration("local_model_dir")
    local_model_name = LaunchConfiguration("local_model_name")
    local_model_path = LaunchConfiguration("local_model_path")
    local_network_variant = LaunchConfiguration("local_network_variant")
    local_hidden_dim = LaunchConfiguration("local_hidden_dim")
    local_append_prev_action = LaunchConfiguration("local_append_prev_action")
    local_policy_max_goal_distance = LaunchConfiguration("local_policy_max_goal_distance")
    planner_waypoint_reached_distance = LaunchConfiguration("planner_waypoint_reached_distance")
    planner_goal_reached_distance = LaunchConfiguration("planner_goal_reached_distance")
    local_waypoint_close_distance = LaunchConfiguration("local_waypoint_close_distance")
    local_waypoint_stop_distance = LaunchConfiguration("local_waypoint_stop_distance")
    planner_obstacle_inflation_radius_m = LaunchConfiguration("planner_obstacle_inflation_radius_m")
    planner_frontier_fail_radius_cells = LaunchConfiguration("planner_frontier_fail_radius_cells")
    planner_frontier_fail_hard_threshold = LaunchConfiguration("planner_frontier_fail_hard_threshold")
    planner_frontier_fail_hard_radius_cells = LaunchConfiguration("planner_frontier_fail_hard_radius_cells")
    planner_frontier_goal_heading_weight = LaunchConfiguration("planner_frontier_goal_heading_weight")
    planner_periodic_replan_sec = LaunchConfiguration("planner_periodic_replan_sec")
    planner_waypoint_hold_timeout_sec = LaunchConfiguration("planner_waypoint_hold_timeout_sec")
    planner_waypoint_min_distance = LaunchConfiguration("planner_waypoint_min_distance")
    planner_waypoint_max_distance = LaunchConfiguration("planner_waypoint_max_distance")
    planner_waypoint_goal_weight = LaunchConfiguration("planner_waypoint_goal_weight")
    planner_waypoint_heading_weight = LaunchConfiguration("planner_waypoint_heading_weight")
    planner_waypoint_revisit_weight = LaunchConfiguration("planner_waypoint_revisit_weight")
    planner_waypoint_clearance_weight = LaunchConfiguration("planner_waypoint_clearance_weight")
    planner_waypoint_clearance_radius_m = LaunchConfiguration("planner_waypoint_clearance_radius_m")
    policy_blend_far = LaunchConfiguration("policy_blend_far")
    policy_blend_near_obstacle = LaunchConfiguration("policy_blend_near_obstacle")
    policy_blend_obstacle_distance = LaunchConfiguration("policy_blend_obstacle_distance")
    enable_local_escape = LaunchConfiguration("enable_local_escape")
    local_obstacle_slow_distance = LaunchConfiguration("local_obstacle_slow_distance")
    local_obstacle_hard_stop_distance = LaunchConfiguration("local_obstacle_hard_stop_distance")
    local_side_guard_distance = LaunchConfiguration("local_side_guard_distance")
    local_avoid_turn_boost = LaunchConfiguration("local_avoid_turn_boost")
    local_control_rate_hz = LaunchConfiguration("local_control_rate_hz")
    local_turn_in_place_angle = LaunchConfiguration("local_turn_in_place_angle")
    local_orbit_break_angle = LaunchConfiguration("local_orbit_break_angle")
    local_goal_active_timeout = LaunchConfiguration("local_goal_active_timeout")
    local_emergency_stop_distance = LaunchConfiguration("local_emergency_stop_distance")
    local_proactive_avoid_distance = LaunchConfiguration("local_proactive_avoid_distance")

    default_workspace = os.environ.get("RL_BASE_WS", "/home/david/Desktop/laiting/rl_base_navigation")
    default_model_dir = os.environ.get("RL_BASE_MODEL_DIR", os.path.join(default_workspace, "navigation_model"))
    default_local_model_path = PathJoinSubstitution([local_model_dir, local_model_name])
    default_world = PathJoinSubstitution(
        [FindPackageShare("goal_seeker_rl"), "worlds", "goal_seeker_large_dynamic.world"]
    )
    default_slam_params = PathJoinSubstitution(
        [FindPackageShare("goal_seeker_rl"), "config", "slam_realsense_mapper.yaml"]
    )
    robot_urdf = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "urdf", "turtlebot3_waffle_minimal.urdf"])
    local_gazebo_models = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "models"])
    gazebo_models = PathJoinSubstitution([FindPackageShare("turtlebot3_gazebo"), "models"])
    turtlebot_common_models = os.environ.get("TURTLEBOT3_COMMON_MODEL_PATH", "")
    gazebo_obstacle_plugins = PathJoinSubstitution(
        [
            FindPackageShare("turtlebot3_gazebo"),
            "models",
            "turtlebot3_drl_world",
            "obstacle_plugin",
            "lib",
        ]
    )
    set_gazebo_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=[
            local_gazebo_models,
            ":",
            gazebo_models,
            ":",
            turtlebot_common_models,
            ":",
            EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
        ],
    )
    set_gazebo_plugin_path = SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=[
            gazebo_obstacle_plugins,
            ":",
            EnvironmentVariable("GAZEBO_PLUGIN_PATH", default_value=""),
        ],
    )
    set_gazebo_model_database_uri = SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value="")

    gzserver_launch = ExecuteProcess(
        cmd=["gzserver", "--verbose", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", world],
        output="screen",
    )

    gzclient_launch = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
        condition=UnlessCondition(headless),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[robot_urdf, "--ros-args", "--log-level", "robot_state_publisher:=fatal"],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_joint_state_publisher),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"])
        ),
        launch_arguments={"use_sim_time": use_sim_time, "slam_params_file": slam_params_file}.items(),
    )

    depth_to_scan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[
            {
                "scan_time": 0.033,
                "range_min": 0.12,
                "range_max": 3.5,
                "scan_height": 16,
                "output_frame": "base_link",
            }
        ],
        remappings=[
            ("depth", "/camera/depth/image_raw"),
            ("depth_camera_info", "/camera/depth/camera_info"),
            ("scan", "/scan"),
        ],
    )

    hrl_global_planner = Node(
        package="goal_seeker_rl",
        executable="hrl_global_planner",
        name="hrl_global_planner",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "lookahead_distance": lookahead_distance,
                "goal_reached_distance": planner_goal_reached_distance,
                "timer_period_sec": 0.25,
                "stuck_window_sec": 6.0,
                "stuck_distance_threshold": 0.08,
                "replan_cooldown_sec": 1.0,
                "periodic_replan_sec": planner_periodic_replan_sec,
                "occupancy_obstacle_threshold": 65,
                "obstacle_inflation_radius_m": planner_obstacle_inflation_radius_m,
                "allow_unknown": True,
                "unknown_penalty": 2.5,
                "frontier_search_enabled": True,
                "frontier_sample_limit": 1200,
                "frontier_top_k": 40,
                "frontier_goal_weight": 2.4,
                "frontier_start_weight": 0.55,
                "frontier_min_distance": 2.0,
                "frontier_min_separation": 1.5,
                "frontier_goal_heading_weight": planner_frontier_goal_heading_weight,
                "frontier_goal_heading_min_cos": 0.12,
                "frontier_goal_progress_min_m": 0.9,
                "path_smoothing_enabled": True,
                "path_max_skip_cells": 24,
                "waypoint_reached_distance": planner_waypoint_reached_distance,
                "waypoint_hold_timeout_sec": planner_waypoint_hold_timeout_sec,
                "waypoint_min_distance": planner_waypoint_min_distance,
                "waypoint_max_distance": planner_waypoint_max_distance,
                "waypoint_goal_weight": planner_waypoint_goal_weight,
                "waypoint_heading_weight": planner_waypoint_heading_weight,
                "waypoint_revisit_weight": planner_waypoint_revisit_weight,
                "waypoint_clearance_weight": planner_waypoint_clearance_weight,
                "waypoint_clearance_radius_m": planner_waypoint_clearance_radius_m,
                "frontier_revisit_weight": 3.2,
                "frontier_fail_radius_cells": planner_frontier_fail_radius_cells,
                "frontier_fail_hard_threshold": planner_frontier_fail_hard_threshold,
                "frontier_fail_hard_radius_cells": planner_frontier_fail_hard_radius_cells,
                "frontier_stagnation_sec": 7.0,
                "map_topic": "/map",
                "odom_topic": "/odom",
                "goal_topic": "/goal_pose",
                "goal_active_topic": "/hrl_goal_active",
                "local_waypoint_topic": "/hrl_local_waypoint",
                "global_path_topic": "/hrl_global_path",
                "map_frame": "map",
                "waypoint_publish_frame": "map",
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
                "scan_topic": "/scan",
                "waypoint_topic": "/hrl_local_waypoint",
                "goal_active_topic": "/hrl_goal_active",
                "cmd_vel_topic": "/cmd_vel",
                "base_frame": "base_link",
                "scan_samples": local_scan_samples,
                "lidar_max_range": 3.5,
                "control_rate_hz": local_control_rate_hz,
                "linear_speed_max": linear_speed_max,
                "angular_speed_max": angular_speed_max,
                "goal_active_timeout_sec": local_goal_active_timeout,
                "waypoint_timeout_sec": 2.0,
                "obstacle_stop_distance": 0.18,
                "obstacle_slow_distance": local_obstacle_slow_distance,
                "obstacle_hard_stop_distance": local_obstacle_hard_stop_distance,
                "emergency_stop_distance": local_emergency_stop_distance,
                "proactive_avoid_distance": local_proactive_avoid_distance,
                "side_guard_distance": local_side_guard_distance,
                "avoid_turn_boost": local_avoid_turn_boost,
                "waypoint_close_distance": local_waypoint_close_distance,
                "waypoint_stop_distance": local_waypoint_stop_distance,
                "turn_in_place_angle": local_turn_in_place_angle,
                "orbit_break_angle": local_orbit_break_angle,
                "policy_blend_far": policy_blend_far,
                "policy_blend_near_obstacle": policy_blend_near_obstacle,
                "policy_blend_obstacle_distance": policy_blend_obstacle_distance,
                "enable_local_escape": enable_local_escape,
                "policy_source": local_policy_source,
                "model_path": local_model_path,
                "network_variant": local_network_variant,
                "hidden_dim": local_hidden_dim,
                "append_prev_action_to_state": local_append_prev_action,
                "model_strict": False,
                "policy_max_goal_distance": local_policy_max_goal_distance,
                "publish_rl_path": True,
                "rl_path_topic": "/rl_model_path",
                "rl_path_frame": "base_link",
                "rl_path_horizon_steps": 28,
                "rl_path_dt_sec": 0.20,
                "lookaround_enabled": True,
                "lookaround_front_distance": 0.75,
                "lookaround_clear_distance": 1.10,
                "lookaround_turn_speed": 0.75,
                "lookaround_duration_sec": 1.30,
                "lookaround_min_duration_sec": 0.45,
                "lookaround_cooldown_sec": 0.80,
                "waiting_scan_creep_enabled": True,
                "waiting_scan_creep_speed": 0.06,
                "waiting_scan_turn_speed": 0.35,
                "waiting_scan_front_clearance": 1.0,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "nav_config.rviz"])],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_joint_state_publisher", default_value="false"),
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("slam_params_file", default_value=default_slam_params),
            DeclareLaunchArgument("lookahead_distance", default_value="1.5"),
            DeclareLaunchArgument("local_scan_samples", default_value="40"),
            DeclareLaunchArgument("local_control_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("linear_speed_max", default_value="0.20"),
            DeclareLaunchArgument("angular_speed_max", default_value="2.2"),
            DeclareLaunchArgument("local_policy_source", default_value="td3"),
            DeclareLaunchArgument("local_model_dir", default_value=default_model_dir),
            DeclareLaunchArgument("local_model_name", default_value="td3_latest.pth"),
            DeclareLaunchArgument(
                "local_model_path",
                default_value=default_local_model_path,
            ),
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
            set_gazebo_model_path,
            set_gazebo_plugin_path,
            set_gazebo_model_database_uri,
            gzserver_launch,
            gzclient_launch,
            robot_state_publisher,
            joint_state_publisher,
            depth_to_scan,
            slam,
            hrl_global_planner,
            rl_local_driver,
            rviz,
        ]
    )
