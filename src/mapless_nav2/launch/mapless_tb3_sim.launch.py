#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    local_tb3_gazebo = "/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo"

    pkg_mapless = get_package_share_directory("mapless_nav2")
    pkg_nav2 = get_package_share_directory("nav2_bringup")
    try:
        get_package_share_directory("gazebo_plugins")
        has_gazebo_plugins = True
    except Exception:
        has_gazebo_plugins = False
    nav2_launch_dir = os.path.join(pkg_nav2, "launch")

    params_default = os.path.join(pkg_mapless, "config", "nav2_mapless_params.yaml")
    slam_params_default = os.path.join(pkg_mapless, "config", "slam_toolbox_online_async.yaml")
    rviz_default = os.path.join(pkg_mapless, "rviz", "mapless_nav2.rviz")
    world_default = os.path.join(local_tb3_gazebo, "worlds", "turtlebot3_house.world")
    robot_sdf_default = os.path.join(local_tb3_gazebo, "models", "turtlebot3_burger", "model.sdf")
    urdf_default = os.path.join(local_tb3_gazebo, "urdf", "turtlebot3_burger.urdf")

    with open(urdf_default, "r", encoding="utf-8") as f:
        robot_description = f.read()

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")

    use_simulator = LaunchConfiguration("use_simulator")
    use_rviz = LaunchConfiguration("use_rviz")
    use_safety_controller = LaunchConfiguration("use_safety_controller")
    use_scan_stabilizer = LaunchConfiguration("use_scan_stabilizer")
    use_slam = LaunchConfiguration("use_slam")
    slam_params_file = LaunchConfiguration("slam_params_file")
    planner_profile = LaunchConfiguration("planner_profile")
    gui = LaunchConfiguration("gui")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    world = LaunchConfiguration("world")
    tb3_model = LaunchConfiguration("tb3_model")
    tb3_models_path = LaunchConfiguration("tb3_models_path")

    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    z_pose = LaunchConfiguration("z_pose")
    yaw = LaunchConfiguration("yaw")
    robot_name = LaunchConfiguration("robot_name")
    robot_sdf = LaunchConfiguration("robot_sdf")

    declare_args = [
        DeclareLaunchArgument("namespace", default_value="", description="Top-level namespace"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation clock"),
        DeclareLaunchArgument("params_file", default_value=params_default, description="Nav2 params file"),
        DeclareLaunchArgument("autostart", default_value="true", description="Autostart nav2 lifecycle"),
        DeclareLaunchArgument("use_composition", default_value="False", description="Use composed nav2 nodes"),
        DeclareLaunchArgument("use_respawn", default_value="False", description="Respawn crashed nav2 nodes"),
        DeclareLaunchArgument("use_simulator", default_value="True", description="Start Gazebo"),
        DeclareLaunchArgument("use_rviz", default_value="True", description="Start RViz"),
        DeclareLaunchArgument(
            "use_safety_controller",
            default_value="False",
            description="Start mapless safety override controller",
        ),
        DeclareLaunchArgument(
            "use_scan_stabilizer",
            default_value="True",
            description="Start scan stabilizer to suppress tilt-induced lidar artifacts",
        ),
        DeclareLaunchArgument("use_slam", default_value="True", description="Start slam_toolbox for mission map building"),
        DeclareLaunchArgument("slam_params_file", default_value=slam_params_default, description="SLAM Toolbox params file"),
        DeclareLaunchArgument(
            "planner_profile",
            default_value="baseline",
            description="mapless planner profile: baseline or advanced",
        ),
        DeclareLaunchArgument("gui", default_value="True", description="Start Gazebo GUI"),
        DeclareLaunchArgument("rviz_config_file", default_value=rviz_default, description="RViz config"),
        DeclareLaunchArgument("tb3_model", default_value="burger", description="TurtleBot3 model"),
        DeclareLaunchArgument(
            "tb3_models_path",
            default_value=os.path.join(local_tb3_gazebo, "models"),
            description="Path to turtlebot3_gazebo/models",
        ),
        DeclareLaunchArgument("world", default_value=world_default, description="Gazebo world model"),
        DeclareLaunchArgument("x_pose", default_value="-2.0", description="Robot x pose"),
        DeclareLaunchArgument("y_pose", default_value="-0.5", description="Robot y pose"),
        DeclareLaunchArgument("z_pose", default_value="0.01", description="Robot z pose"),
        DeclareLaunchArgument("yaw", default_value="0.0", description="Robot yaw"),
        DeclareLaunchArgument("robot_name", default_value="turtlebot3_burger", description="Robot name"),
        DeclareLaunchArgument("robot_sdf", default_value=robot_sdf_default, description="Robot SDF"),
    ]

    set_tb3_model_env = SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value=tb3_model)
    set_gazebo_model_path_env = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=[
            tb3_models_path,
            ":",
            EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
        ],
    )

    start_gazebo_server_cmd = ExecuteProcess(
        condition=IfCondition(use_simulator),
        cmd=[
            "gzserver",
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
            world,
        ],
        output="screen",
        on_exit=[
            LogInfo(msg="[mapless_nav2] gzserver exited. Shutting down all ROS2 nodes (including RViz)."),
            EmitEvent(event=Shutdown(reason="gzserver exited")),
        ],
    )

    start_gazebo_client_cmd = ExecuteProcess(
        condition=IfCondition(PythonExpression([use_simulator, " and ", gui])),
        cmd=["gzclient"],
        output="screen",
    )

    robot_state_publisher_cmd = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
    )

    gazebo_spawner_cmd = Node(
        condition=IfCondition(use_simulator),
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            robot_name,
            "-file",
            robot_sdf,
            "-x",
            x_pose,
            "-y",
            y_pose,
            "-z",
            z_pose,
            "-Y",
            yaw,
        ],
    )

    nav2_navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, "navigation_launch.py")),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "params_file": params_file,
            "use_composition": use_composition,
            "use_respawn": use_respawn,
            "container_name": "nav2_container",
            "log_level": "info",
        }.items(),
    )

    mapless_goal_manager_cmd = Node(
        package="mapless_nav2",
        executable="mapless_goal_manager",
        name="mapless_goal_manager",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame": "odom",
                "robot_frame": "base_footprint",
                "scan_topic": "/scan_stable",
                "local_costmap_topic": "/global_costmap/costmap",
                "planner_profile": planner_profile,
                "grid_planner_enabled": True,
                "subgoal_lookahead": 1.5,
                "collision_clearance": 0.16,
                "mission_obstacle_hit_increment": 2.4,
                "mission_obstacle_clear_decrement": 0.05,
                "mission_obstacle_block_threshold": 0.65,
                "mission_obstacle_block_radius": 0.18,
                "experience_fail_penalty_weight": 1.6,
                "experience_revisit_penalty_weight": 0.45,
            }
        ],
    )

    mapless_scan_stabilizer_cmd = Node(
        condition=IfCondition(use_scan_stabilizer),
        package="mapless_nav2",
        executable="mapless_scan_stabilizer",
        name="mapless_scan_stabilizer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "input_scan_topic": "/scan",
                "output_scan_topic": "/scan_stable",
                "tilt_status_topic": "/scan_tilt_exceeded",
                "global_frame": "odom",
                "base_frame": "base_scan",
                "odom_topic": "/odom",
                "max_roll_deg": 4.0,
                "max_pitch_deg": 4.0,
                "hard_stop_deg": 6.5,
                "hysteresis_deg": 1.0,
                "max_yaw_rate_deg_s": 12.0,
                "max_yaw_delta_per_scan_deg": 1.2,
                "drop_scan_on_fast_turn": True,
            }
        ],
    )

    slam_toolbox_cmd = Node(
        condition=IfCondition(use_slam),
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[slam_params_file, {"use_sim_time": use_sim_time}],
    )

    mapless_safety_controller_cmd = Node(
        condition=IfCondition(use_safety_controller),
        package="mapless_nav2",
        executable="mapless_safety_controller",
        name="mapless_safety_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "input_cmd_topic": "/cmd_vel",
                "output_cmd_topic": "/cmd_vel_safe",
            }
        ],
    )

    rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()
    for action in declare_args:
        ld.add_action(action)

    if not has_gazebo_plugins:
        ld.add_action(
            LogInfo(
                msg=(
                    "[mapless_nav2] gazebo_plugins package not found. "
                    "Robot may spawn but /odom and /scan will be missing. "
                    "Install: sudo apt install ros-humble-gazebo-plugins"
                )
            )
        )

    ld.add_action(set_tb3_model_env)
    ld.add_action(set_gazebo_model_path_env)
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(gazebo_spawner_cmd)
    ld.add_action(mapless_scan_stabilizer_cmd)
    ld.add_action(slam_toolbox_cmd)
    ld.add_action(nav2_navigation_cmd)
    ld.add_action(mapless_goal_manager_cmd)
    ld.add_action(mapless_safety_controller_cmd)
    ld.add_action(rviz_cmd)

    return ld
