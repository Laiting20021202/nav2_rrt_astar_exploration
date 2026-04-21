#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription


def generate_launch_description() -> LaunchDescription:
    local_tb3_gazebo = "/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo"

    pkg_mapless = get_package_share_directory("mapless_nav2")
    pkg_nav2 = get_package_share_directory("nav2_bringup")
    nav2_launch_dir = os.path.join(pkg_nav2, "launch")
    try:
        get_package_share_directory("gazebo_plugins")
        has_gazebo_plugins = True
    except Exception:
        has_gazebo_plugins = False

    params_default = os.path.join(pkg_mapless, "config", "nav2_frontier_exploration_params.yaml")
    exploration_params = os.path.join(pkg_mapless, "config", "exploration_manager.yaml")
    slam_params = os.path.join(pkg_mapless, "config", "slam_toolbox_online_async.yaml")
    rviz_default = os.path.join(pkg_mapless, "rviz", "frontier_explore.rviz")
    world_default = os.path.join(local_tb3_gazebo, "worlds", "turtlebot3_house.world")
    robot_sdf_default = os.path.join(pkg_mapless, "models", "turtlebot3_waffle_45deg", "model.sdf")
    urdf_default = os.path.join(local_tb3_gazebo, "urdf", "turtlebot3_waffle.urdf")

    with open(urdf_default, "r", encoding="utf-8") as f:
        robot_description = f.read()

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")
    use_simulator = LaunchConfiguration("use_simulator")
    use_rviz = LaunchConfiguration("use_rviz")
    use_scan_stabilizer = LaunchConfiguration("use_scan_stabilizer")
    gui = LaunchConfiguration("gui")
    params_file = LaunchConfiguration("params_file")
    exploration_params_file = LaunchConfiguration("exploration_params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    world = LaunchConfiguration("world")
    tb3_models_path = LaunchConfiguration("tb3_models_path")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    z_pose = LaunchConfiguration("z_pose")
    yaw = LaunchConfiguration("yaw")
    robot_name = LaunchConfiguration("robot_name")
    robot_sdf = LaunchConfiguration("robot_sdf")

    declare_args = [
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("use_composition", default_value="False"),
        DeclareLaunchArgument("use_respawn", default_value="False"),
        DeclareLaunchArgument("use_simulator", default_value="True"),
        DeclareLaunchArgument("use_rviz", default_value="True"),
        DeclareLaunchArgument("use_scan_stabilizer", default_value="True"),
        DeclareLaunchArgument("gui", default_value="True"),
        DeclareLaunchArgument("params_file", default_value=params_default),
        DeclareLaunchArgument("exploration_params_file", default_value=exploration_params),
        DeclareLaunchArgument("slam_params_file", default_value=slam_params),
        DeclareLaunchArgument("rviz_config_file", default_value=rviz_default),
        DeclareLaunchArgument("tb3_models_path", default_value=os.path.join(local_tb3_gazebo, "models")),
        DeclareLaunchArgument("world", default_value=world_default),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("z_pose", default_value="0.01"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("robot_name", default_value="turtlebot3_waffle"),
        DeclareLaunchArgument("robot_sdf", default_value=robot_sdf_default),
    ]

    set_tb3_model_env = SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value="waffle")
    set_gazebo_model_path_env = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=[tb3_models_path, ":", EnvironmentVariable("GAZEBO_MODEL_PATH", default_value="")],
    )

    start_gazebo_server_cmd = ExecuteProcess(
        condition=IfCondition(use_simulator),
        cmd=["gzserver", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", world],
        output="screen",
        on_exit=[
            LogInfo(msg="[mapless_nav2] gzserver exited. Shutting down frontier stack."),
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

    slam_toolbox_cmd = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[slam_params_file, {"use_sim_time": use_sim_time}],
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

    nav2_bringup_log_cmd = LogInfo(
        msg=[
            "[mapless_nav2] Starting Nav2 bringup with autostart=",
            autostart,
            " params_file=",
            params_file,
            " use_composition=",
            use_composition,
        ]
    )

    nav2_lifecycle_hint_log_cmd = LogInfo(
        msg=(
            "[mapless_nav2] Watch bt_navigator / planner_server / controller_server / "
            "behavior_server / lifecycle_manager_navigation logs during startup. "
            "If Navigation stays inactive, one of these nodes failed configure/activate."
        )
    )

    scan_stabilizer_cmd = Node(
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
                "odom_topic": "/odom",
                "global_frame": "odom",
                "base_frame": "base_footprint",
            }
        ],
    )

    exploration_coordinator_cmd = Node(
        package="mapless_nav2",
        executable="exploration_coordinator",
        name="exploration_coordinator",
        output="screen",
        parameters=[exploration_params_file, {"use_sim_time": use_sim_time}],
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
    ld.add_action(scan_stabilizer_cmd)
    ld.add_action(slam_toolbox_cmd)
    ld.add_action(nav2_bringup_log_cmd)
    ld.add_action(nav2_lifecycle_hint_log_cmd)
    ld.add_action(nav2_navigation_cmd)
    ld.add_action(exploration_coordinator_cmd)
    ld.add_action(rviz_cmd)
    return ld
