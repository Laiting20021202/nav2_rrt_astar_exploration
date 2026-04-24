"""Launch full hierarchical RL navigation stack in Gazebo."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
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
    lookahead_distance = LaunchConfiguration("lookahead_distance")
    local_scan_samples = LaunchConfiguration("local_scan_samples")
    linear_speed_max = LaunchConfiguration("linear_speed_max")
    angular_speed_max = LaunchConfiguration("angular_speed_max")
    local_policy_source = LaunchConfiguration("local_policy_source")
    local_model_path = LaunchConfiguration("local_model_path")
    local_network_variant = LaunchConfiguration("local_network_variant")
    local_hidden_dim = LaunchConfiguration("local_hidden_dim")
    local_append_prev_action = LaunchConfiguration("local_append_prev_action")
    local_policy_max_goal_distance = LaunchConfiguration("local_policy_max_goal_distance")
    planner_waypoint_reached_distance = LaunchConfiguration("planner_waypoint_reached_distance")
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

    default_world = PathJoinSubstitution(
        [FindPackageShare("goal_seeker_rl"), "worlds", "goal_seeker_large_dynamic.world"]
    )
    robot_urdf = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "urdf", "turtlebot3_waffle_minimal.urdf"])
    gazebo_models = PathJoinSubstitution([FindPackageShare("turtlebot3_gazebo"), "models"])
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
        value=[gazebo_models, ":", EnvironmentVariable("GAZEBO_MODEL_PATH", default_value="")],
    )
    set_gazebo_plugin_path = SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=[
            gazebo_obstacle_plugins,
            ":",
            EnvironmentVariable("GAZEBO_PLUGIN_PATH", default_value=""),
        ],
    )

    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gzserver.launch.py"])
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gzclient.launch.py"])
        ),
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
        launch_arguments={"use_sim_time": use_sim_time}.items(),
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
                "frontier_min_distance": 1.5,
                "frontier_min_separation": 1.5,
                "frontier_goal_heading_weight": planner_frontier_goal_heading_weight,
                "frontier_goal_heading_min_cos": 0.12,
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
                "local_waypoint_topic": "/hrl_local_waypoint",
                "global_path_topic": "/hrl_global_path",
                "map_frame": "map",
                "waypoint_publish_frame": "odom",
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
                "cmd_vel_topic": "/cmd_vel",
                "base_frame": "base_link",
                "scan_samples": local_scan_samples,
                "lidar_max_range": 3.5,
                "control_rate_hz": local_control_rate_hz,
                "linear_speed_max": linear_speed_max,
                "angular_speed_max": angular_speed_max,
                "waypoint_timeout_sec": 2.0,
                "obstacle_stop_distance": 0.18,
                "obstacle_slow_distance": local_obstacle_slow_distance,
                "obstacle_hard_stop_distance": local_obstacle_hard_stop_distance,
                "side_guard_distance": local_side_guard_distance,
                "avoid_turn_boost": local_avoid_turn_boost,
                "waypoint_close_distance": local_waypoint_close_distance,
                "waypoint_stop_distance": local_waypoint_stop_distance,
                "turn_in_place_angle": local_turn_in_place_angle,
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
            DeclareLaunchArgument("lookahead_distance", default_value="1.5"),
            DeclareLaunchArgument("local_scan_samples", default_value="40"),
            DeclareLaunchArgument("local_control_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("linear_speed_max", default_value="0.20"),
            DeclareLaunchArgument("angular_speed_max", default_value="2.2"),
            DeclareLaunchArgument("local_policy_source", default_value="reference_actor"),
            DeclareLaunchArgument(
                "local_model_path",
                default_value="/home/david/Desktop/laiting/rl_base_navigation/reference/turtlebot3_drlnav/src/turtlebot3_drl/model/examples/ddpg_0_stage9/actor_stage9_episode8000.pt",
            ),
            DeclareLaunchArgument("local_network_variant", default_value="reference"),
            DeclareLaunchArgument("local_hidden_dim", default_value="512"),
            DeclareLaunchArgument("local_append_prev_action", default_value="true"),
            DeclareLaunchArgument("local_policy_max_goal_distance", default_value="5.94"),
            DeclareLaunchArgument("planner_waypoint_reached_distance", default_value="0.50"),
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
            set_gazebo_model_path,
            set_gazebo_plugin_path,
            gzserver_launch,
            gzclient_launch,
            robot_state_publisher,
            joint_state_publisher,
            slam,
            hrl_global_planner,
            rl_local_driver,
            rviz,
        ]
    )
