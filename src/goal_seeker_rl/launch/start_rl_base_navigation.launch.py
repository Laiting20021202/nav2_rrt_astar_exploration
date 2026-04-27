"""Launch Gazebo + RViz with only the rl_base TD3 controller active."""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Start Realsense depth scan conversion and the RL base policy only."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")
    world = LaunchConfiguration("world")
    model_dir = LaunchConfiguration("model_dir")
    model_name = LaunchConfiguration("model_name")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")

    default_workspace = os.environ.get("RL_BASE_WS", "/home/david/Desktop/laiting/rl_base_navigation")
    default_model_dir = os.environ.get("RL_BASE_MODEL_DIR", os.path.join(default_workspace, "navigation_model"))
    default_model_path = PathJoinSubstitution([model_dir, model_name])
    default_world = PathJoinSubstitution(
        [FindPackageShare("goal_seeker_rl"), "worlds", "goal_seeker_large_dynamic.world"]
    )
    default_rviz = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "rl_base_config.rviz"])
    robot_urdf = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "urdf", "turtlebot3_waffle_minimal.urdf"])
    local_gazebo_models = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "models"])
    turtlebot_gazebo_models = PathJoinSubstitution([FindPackageShare("turtlebot3_gazebo"), "models"])
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
            turtlebot_gazebo_models,
            ":",
            turtlebot_common_models,
            ":",
            EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
        ],
    )
    set_gazebo_plugin_path = SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=[gazebo_obstacle_plugins, ":", EnvironmentVariable("GAZEBO_PLUGIN_PATH", default_value="")],
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
        arguments=[robot_urdf],
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

    rl_base = Node(
        package="goal_seeker_rl",
        executable="goal_seeker_main",
        name="goal_seeker_rl",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "inference_mode": True,
                "policy_source": "td3",
                "model_path": model_path,
                "state_scan_samples": 40,
                "append_prev_action_to_state": True,
                "network_variant": "reference",
                "hidden_dim": 512,
                "scan_topic": "/scan",
                "odom_topic": "/odom",
                "cmd_vel_topic": "/cmd_vel",
                "goal_topic": "/goal_pose",
                "lidar_max_range": 3.5,
                "max_goal_distance": 5.94,
                "goal_tolerance": 0.38,
                "collision_distance": 0.30,
                "linear_speed_max": 0.22,
                "angular_speed_max": 1.0,
                "inference_reset_on_stuck": False,
                "hybrid_exploration_enabled": False,
                "escape_override_enabled": True,
                "safety_override_enabled": True,
                "safety_front_stop_distance": 0.42,
                "safety_front_slow_distance": 0.95,
                "safety_front_angle_deg": 28.0,
                "safety_side_stop_distance": 0.34,
                "safety_turn_min": 0.62,
                "safety_side_linear_cap": 0.04,
                "goal_stop_distance": 0.38,
                "goal_slow_distance": 0.85,
                "lookaround_enabled": True,
                "lookaround_front_distance": 0.75,
                "lookaround_clear_distance": 1.10,
                "lookaround_front_angle_deg": 34.0,
                "lookaround_turn_speed": 0.55,
                "lookaround_duration_sec": 1.35,
                "lookaround_min_duration_sec": 0.45,
                "lookaround_cooldown_sec": 0.80,
                "publish_rl_path": True,
                "rl_path_topic": "/rl_model_path",
                "rl_path_frame": "odom",
                "rl_path_horizon_steps": 35,
                "rl_path_dt_sec": 0.20,
                "rl_path_obstacle_margin": 0.42,
                "rl_path_clearance_angle_window_deg": 8.0,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("model_dir", default_value=default_model_dir),
            DeclareLaunchArgument("model_name", default_value="td3_latest.pth"),
            DeclareLaunchArgument(
                "model_path",
                default_value=default_model_path,
            ),
            set_gazebo_model_path,
            set_gazebo_plugin_path,
            set_gazebo_model_database_uri,
            gzserver_launch,
            gzclient_launch,
            robot_state_publisher,
            depth_to_scan,
            rl_base,
            rviz,
        ]
    )
