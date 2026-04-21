#!/usr/bin/env python3
"""
Compatibility launch file.

This project is rolled back to the stable legacy mapless pipeline:
  mapless_tb3_sim.launch.py

`frontier_explore_tb3.launch.py` is kept as an alias so existing commands
continue to work, but it now forwards to `mapless_tb3_sim.launch.py`.
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
    legacy_launch = os.path.join(pkg_mapless, "launch", "mapless_tb3_sim.launch.py")
    default_world = os.path.join(local_tb3_gazebo, "worlds", "turtlebot3_house.world")
    default_robot_sdf = os.path.join(
        pkg_mapless,
        "models",
        "turtlebot3_waffle_45deg",
        "model.sdf",
    )

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_simulator", default_value="True"),
        DeclareLaunchArgument("use_rviz", default_value="True"),
        DeclareLaunchArgument("gui", default_value="True"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(pkg_mapless, "config", "nav2_mapless_params.yaml")),
        DeclareLaunchArgument("rviz_config_file", default_value=os.path.join(pkg_mapless, "rviz", "mapless_nav2.rviz")),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("robot_sdf", default_value=default_robot_sdf),
        DeclareLaunchArgument("robot_name", default_value="turtlebot3_waffle"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("z_pose", default_value="0.01"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("use_safety_controller", default_value="False"),
        DeclareLaunchArgument("use_scan_stabilizer", default_value="True"),
        DeclareLaunchArgument("planner_profile", default_value="baseline"),
    ]

    include_legacy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(legacy_launch),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "use_simulator": LaunchConfiguration("use_simulator"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "gui": LaunchConfiguration("gui"),
            "params_file": LaunchConfiguration("params_file"),
            "rviz_config_file": LaunchConfiguration("rviz_config_file"),
            "world": LaunchConfiguration("world"),
            "robot_sdf": LaunchConfiguration("robot_sdf"),
            "robot_name": LaunchConfiguration("robot_name"),
            "x_pose": LaunchConfiguration("x_pose"),
            "y_pose": LaunchConfiguration("y_pose"),
            "z_pose": LaunchConfiguration("z_pose"),
            "yaw": LaunchConfiguration("yaw"),
            "use_safety_controller": LaunchConfiguration("use_safety_controller"),
            "use_scan_stabilizer": LaunchConfiguration("use_scan_stabilizer"),
            "planner_profile": LaunchConfiguration("planner_profile"),
        }.items(),
    )

    ld = LaunchDescription()
    for arg in args:
        ld.add_action(arg)
    ld.add_action(
        LogInfo(
            msg="[mapless_nav2] frontier_explore launch is aliased to legacy mapless_tb3_sim pipeline."
        )
    )
    ld.add_action(include_legacy)
    return ld
