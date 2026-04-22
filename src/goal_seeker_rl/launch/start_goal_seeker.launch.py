"""Launch Gazebo + RViz + goal_seeker_rl node for TurtleBot3 Waffle."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate full simulation and visualization launch."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")
    inference_mode = LaunchConfiguration("inference_mode")
    model_path = LaunchConfiguration("model_path")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    world = LaunchConfiguration("world")
    rviz_config = LaunchConfiguration("rviz_config")

    default_world = PathJoinSubstitution(
        [
            FindPackageShare("turtlebot3_gazebo"),
            "worlds",
            "turtlebot3_drl_stage1",
            "waffle.model",
        ]
    )
    default_rviz = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "nav_config.rviz"])
    robot_urdf = PathJoinSubstitution(
        [FindPackageShare("turtlebot3_description"), "urdf", "turtlebot3_waffle.urdf"]
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
        arguments=[robot_urdf],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
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

    goal_seeker_main = Node(
        package="goal_seeker_rl",
        executable="goal_seeker_main",
        name="goal_seeker_rl",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "inference_mode": inference_mode,
                "model_path": model_path,
                "checkpoint_dir": checkpoint_dir,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            # For faster training set: headless:=true use_rviz:=false
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("inference_mode", default_value="false"),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument(
                "checkpoint_dir",
                default_value="/home/david/Desktop/laiting/rl_base_navigation/src/goal_seeker_rl/model",
            ),
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            gzserver_launch,
            gzclient_launch,
            robot_state_publisher,
            joint_state_publisher,
            rviz,
            goal_seeker_main,
        ]
    )
