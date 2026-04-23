"""Launch Stretch 3 simulation + RViz + goal_seeker_rl node."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate a Stretch 3 (MuJoCo) navigation launch with goal_seeker_rl."""
    use_rviz = LaunchConfiguration("use_rviz")
    use_mujoco_viewer = LaunchConfiguration("use_mujoco_viewer")
    use_robocasa = LaunchConfiguration("use_robocasa")
    robocasa_layout = LaunchConfiguration("robocasa_layout")
    robocasa_style = LaunchConfiguration("robocasa_style")
    inference_mode = LaunchConfiguration("inference_mode")
    policy_source = LaunchConfiguration("policy_source")
    model_path = LaunchConfiguration("model_path")
    reference_actor_path = LaunchConfiguration("reference_actor_path")
    reference_state_scan_samples = LaunchConfiguration("reference_state_scan_samples")
    resume_model_path = LaunchConfiguration("resume_model_path")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    auto_goal_training = LaunchConfiguration("auto_goal_training")
    rviz_config = LaunchConfiguration("rviz_config")

    default_rviz = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "stretch_nav_config.rviz"])

    stretch_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("stretch_simulation"), "launch", "stretch_mujoco_driver.launch.py"])
        ),
        launch_arguments={
            "mode": "navigation",
            "use_rviz": "false",
            "use_mujoco_viewer": use_mujoco_viewer,
            "use_robocasa": use_robocasa,
            "robocasa_layout": robocasa_layout,
            "robocasa_style": robocasa_style,
        }.items(),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(use_rviz),
    )

    goal_seeker_main = Node(
        package="goal_seeker_rl",
        executable="goal_seeker_main",
        name="goal_seeker_rl",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "inference_mode": inference_mode,
                "policy_source": policy_source,
                "model_path": model_path,
                "reference_actor_path": reference_actor_path,
                "reference_state_scan_samples": reference_state_scan_samples,
                "resume_model_path": resume_model_path,
                "checkpoint_dir": checkpoint_dir,
                "scan_topic": "/scan_filtered",
                "odom_topic": "/odom",
                "cmd_vel_topic": "/stretch/cmd_vel",
                "goal_topic": "/goal_pose",
                "reset_service_name": "",
                "goal_marker_frame": "odom",
                "lidar_max_range": 12.0,
                "max_goal_distance": 12.0,
                "goal_tolerance": 0.35,
                "collision_distance": 0.18,
                "linear_speed_max": 0.26,
                "angular_speed_max": 1.0,
                "reset_on_episode_end": False,
                "auto_goal_training": auto_goal_training,
                "auto_goal_min_radius": 1.0,
                "auto_goal_max_radius": 4.5,
                "auto_goal_max_abs_x": 10.0,
                "auto_goal_max_abs_y": 10.0,
                "publish_goal_marker": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_mujoco_viewer", default_value="true"),
            DeclareLaunchArgument("use_robocasa", default_value="true"),
            DeclareLaunchArgument("robocasa_layout", default_value="G-shaped (large)"),
            DeclareLaunchArgument("robocasa_style", default_value="Modern_1"),
            DeclareLaunchArgument("inference_mode", default_value="false"),
            DeclareLaunchArgument("policy_source", default_value="td3"),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument(
                "reference_actor_path",
                default_value="/home/david/Desktop/laiting/rl_base_navigation/reference/turtlebot3_drlnav/src/turtlebot3_drl/model/examples/ddpg_0_stage9/actor_stage9_episode8000.pt",
            ),
            DeclareLaunchArgument("reference_state_scan_samples", default_value="40"),
            DeclareLaunchArgument("resume_model_path", default_value=""),
            DeclareLaunchArgument("auto_goal_training", default_value="false"),
            DeclareLaunchArgument(
                "checkpoint_dir",
                default_value="/home/david/Desktop/laiting/rl_base_navigation/src/goal_seeker_rl/model",
            ),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            stretch_sim_launch,
            rviz,
            goal_seeker_main,
        ]
    )
