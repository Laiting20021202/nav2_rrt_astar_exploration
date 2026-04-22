#!/usr/bin/env python3
"""
Compatibility launch file.

`mapless_tb3_sim.launch.py` now forwards to the frontier exploration stack so
existing commands continue to work.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    local_tb3_gazebo = "/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo"
    pkg_mapless = get_package_share_directory("mapless_nav2")
    frontier_launch = os.path.join(pkg_mapless, "launch", "frontier_explore_tb3.launch.py")
    default_world = os.path.join(local_tb3_gazebo, "worlds", "turtlebot3_house.world")
    default_robot_sdf = os.path.join(local_tb3_gazebo, "models", "turtlebot3_burger", "model.sdf")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("use_composition", default_value="False"),
        DeclareLaunchArgument("use_respawn", default_value="False"),
        DeclareLaunchArgument("use_simulator", default_value="True"),
        DeclareLaunchArgument("use_rviz", default_value="True"),
        DeclareLaunchArgument("gui", default_value="True"),
        DeclareLaunchArgument("use_scan_stabilizer", default_value="True"),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(pkg_mapless, "config", "nav2_frontier_exploration_params.yaml"),
        ),
        DeclareLaunchArgument(
            "exploration_params_file",
            default_value=os.path.join(pkg_mapless, "config", "exploration_manager.yaml"),
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=os.path.join(pkg_mapless, "config", "slam_toolbox_online_async.yaml"),
        ),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=os.path.join(pkg_mapless, "rviz", "frontier_explore.rviz"),
        ),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("robot_sdf", default_value=default_robot_sdf),
        DeclareLaunchArgument("robot_name", default_value="turtlebot3_burger"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("z_pose", default_value="0.01"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
    ]

    include_frontier = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(frontier_launch),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": LaunchConfiguration("autostart"),
            "use_composition": LaunchConfiguration("use_composition"),
            "use_respawn": LaunchConfiguration("use_respawn"),
            "use_simulator": LaunchConfiguration("use_simulator"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "gui": LaunchConfiguration("gui"),
            "use_scan_stabilizer": LaunchConfiguration("use_scan_stabilizer"),
            "params_file": LaunchConfiguration("params_file"),
            "exploration_params_file": LaunchConfiguration("exploration_params_file"),
            "slam_params_file": LaunchConfiguration("slam_params_file"),
            "rviz_config_file": LaunchConfiguration("rviz_config_file"),
            "world": LaunchConfiguration("world"),
            "robot_sdf": LaunchConfiguration("robot_sdf"),
            "robot_name": LaunchConfiguration("robot_name"),
            "x_pose": LaunchConfiguration("x_pose"),
            "y_pose": LaunchConfiguration("y_pose"),
            "z_pose": LaunchConfiguration("z_pose"),
            "yaw": LaunchConfiguration("yaw"),
        }.items(),
    )

    ld = LaunchDescription()
    for arg in args:
        ld.add_action(arg)
    ld.add_action(LogInfo(msg="[mapless_nav2] mapless_tb3_sim launch now forwards to frontier exploration stack."))
    ld.add_action(include_frontier)
    return ld
