"""Launch Stretch 3 simulation + RViz + goal_seeker_rl node."""

from __future__ import annotations

import os

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
    state_scan_samples = LaunchConfiguration("state_scan_samples")
    append_prev_action_to_state = LaunchConfiguration("append_prev_action_to_state")
    network_variant = LaunchConfiguration("network_variant")
    bootstrap_actor_path = LaunchConfiguration("bootstrap_actor_path")
    bootstrap_actor_strict = LaunchConfiguration("bootstrap_actor_strict")
    resume_model_path = LaunchConfiguration("resume_model_path")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    checkpoint_interval_steps = LaunchConfiguration("checkpoint_interval_steps")
    auto_goal_training = LaunchConfiguration("auto_goal_training")
    auto_goal_min_radius = LaunchConfiguration("auto_goal_min_radius")
    auto_goal_max_radius = LaunchConfiguration("auto_goal_max_radius")
    auto_goal_curriculum_steps = LaunchConfiguration("auto_goal_curriculum_steps")
    auto_goal_start_scale = LaunchConfiguration("auto_goal_start_scale")
    stuck_cell_size = LaunchConfiguration("stuck_cell_size")
    stuck_overlap_threshold = LaunchConfiguration("stuck_overlap_threshold")
    stuck_min_displacement = LaunchConfiguration("stuck_min_displacement")
    goal_tolerance = LaunchConfiguration("goal_tolerance")
    collision_distance = LaunchConfiguration("collision_distance")
    max_episode_steps = LaunchConfiguration("max_episode_steps")
    warmup_steps = LaunchConfiguration("warmup_steps")
    hidden_dim = LaunchConfiguration("hidden_dim")
    actor_lr = LaunchConfiguration("actor_lr")
    critic_lr = LaunchConfiguration("critic_lr")
    tau = LaunchConfiguration("tau")
    replay_size = LaunchConfiguration("replay_size")
    episodic_memory_enabled = LaunchConfiguration("episodic_memory_enabled")
    memory_cell_size = LaunchConfiguration("memory_cell_size")
    memory_novelty_reward = LaunchConfiguration("memory_novelty_reward")
    memory_revisit_penalty = LaunchConfiguration("memory_revisit_penalty")
    dead_end_front_distance = LaunchConfiguration("dead_end_front_distance")
    dead_end_side_distance = LaunchConfiguration("dead_end_side_distance")
    escape_override_enabled = LaunchConfiguration("escape_override_enabled")
    escape_blend_gain = LaunchConfiguration("escape_blend_gain")
    escape_linear_cap = LaunchConfiguration("escape_linear_cap")
    escape_min_turn = LaunchConfiguration("escape_min_turn")
    escape_overlap_gate = LaunchConfiguration("escape_overlap_gate")
    rviz_config = LaunchConfiguration("rviz_config")

    default_rviz = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "stretch_nav_config.rviz"])
    default_workspace = os.environ.get("RL_BASE_WS", "/home/david/Desktop/laiting/rl_base_navigation")
    default_model_dir = os.environ.get("RL_BASE_MODEL_DIR", os.path.join(default_workspace, "navigation_model"))
    default_reference_actor = os.path.join(
        default_workspace,
        "reference",
        "turtlebot3_drlnav",
        "src",
        "turtlebot3_drl",
        "model",
        "examples",
        "ddpg_0_stage9",
        "actor_stage9_episode8000.pt",
    )

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
                "state_scan_samples": state_scan_samples,
                "append_prev_action_to_state": append_prev_action_to_state,
                "network_variant": network_variant,
                "bootstrap_actor_path": bootstrap_actor_path,
                "bootstrap_actor_strict": bootstrap_actor_strict,
                "resume_model_path": resume_model_path,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_interval_steps": checkpoint_interval_steps,
                "max_episode_steps": max_episode_steps,
                "warmup_steps": warmup_steps,
                "hidden_dim": hidden_dim,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "tau": tau,
                "replay_size": replay_size,
                "episodic_memory_enabled": episodic_memory_enabled,
                "memory_cell_size": memory_cell_size,
                "memory_novelty_reward": memory_novelty_reward,
                "memory_revisit_penalty": memory_revisit_penalty,
                "dead_end_front_distance": dead_end_front_distance,
                "dead_end_side_distance": dead_end_side_distance,
                "escape_override_enabled": escape_override_enabled,
                "escape_blend_gain": escape_blend_gain,
                "escape_linear_cap": escape_linear_cap,
                "escape_min_turn": escape_min_turn,
                "escape_overlap_gate": escape_overlap_gate,
                "scan_topic": "/scan_filtered",
                "odom_topic": "/odom",
                "cmd_vel_topic": "/stretch/cmd_vel",
                "goal_topic": "/goal_pose",
                "reset_service_name": "",
                "goal_marker_frame": "odom",
                "lidar_max_range": 12.0,
                "max_goal_distance": 12.0,
                "goal_tolerance": goal_tolerance,
                "collision_distance": collision_distance,
                "linear_speed_max": 0.26,
                "angular_speed_max": 1.0,
                "reset_on_episode_end": False,
                "auto_goal_training": auto_goal_training,
                "auto_goal_min_radius": auto_goal_min_radius,
                "auto_goal_max_radius": auto_goal_max_radius,
                "auto_goal_curriculum_steps": auto_goal_curriculum_steps,
                "auto_goal_start_scale": auto_goal_start_scale,
                "auto_goal_max_abs_x": 10.0,
                "auto_goal_max_abs_y": 10.0,
                "stuck_cell_size": stuck_cell_size,
                "stuck_overlap_threshold": stuck_overlap_threshold,
                "stuck_min_displacement": stuck_min_displacement,
                "publish_goal_marker": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_mujoco_viewer", default_value="true"),
            DeclareLaunchArgument("use_robocasa", default_value="true"),
            DeclareLaunchArgument("robocasa_layout", default_value="Wraparound"),
            DeclareLaunchArgument("robocasa_style", default_value="Modern_1"),
            DeclareLaunchArgument("inference_mode", default_value="false"),
            DeclareLaunchArgument("policy_source", default_value="td3"),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument(
                "reference_actor_path",
                default_value=default_reference_actor,
            ),
            DeclareLaunchArgument("reference_state_scan_samples", default_value="40"),
            DeclareLaunchArgument("state_scan_samples", default_value="40"),
            DeclareLaunchArgument("append_prev_action_to_state", default_value="true"),
            DeclareLaunchArgument("network_variant", default_value="reference"),
            DeclareLaunchArgument(
                "bootstrap_actor_path",
                default_value=default_reference_actor,
            ),
            DeclareLaunchArgument("bootstrap_actor_strict", default_value="true"),
            DeclareLaunchArgument("resume_model_path", default_value=""),
            DeclareLaunchArgument("auto_goal_training", default_value="false"),
            DeclareLaunchArgument("auto_goal_min_radius", default_value="0.8"),
            DeclareLaunchArgument("auto_goal_max_radius", default_value="5.5"),
            DeclareLaunchArgument("auto_goal_curriculum_steps", default_value="60000"),
            DeclareLaunchArgument("auto_goal_start_scale", default_value="0.25"),
            DeclareLaunchArgument("stuck_cell_size", default_value="0.20"),
            DeclareLaunchArgument("stuck_overlap_threshold", default_value="0.85"),
            DeclareLaunchArgument("stuck_min_displacement", default_value="0.75"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.45"),
            DeclareLaunchArgument("collision_distance", default_value="0.18"),
            DeclareLaunchArgument("max_episode_steps", default_value="1800"),
            DeclareLaunchArgument("warmup_steps", default_value="2000"),
            DeclareLaunchArgument("hidden_dim", default_value="512"),
            DeclareLaunchArgument("actor_lr", default_value="0.003"),
            DeclareLaunchArgument("critic_lr", default_value="0.003"),
            DeclareLaunchArgument("tau", default_value="0.003"),
            DeclareLaunchArgument("replay_size", default_value="1000000"),
            DeclareLaunchArgument("episodic_memory_enabled", default_value="true"),
            DeclareLaunchArgument("memory_cell_size", default_value="0.20"),
            DeclareLaunchArgument("memory_novelty_reward", default_value="0.40"),
            DeclareLaunchArgument("memory_revisit_penalty", default_value="0.80"),
            DeclareLaunchArgument("dead_end_front_distance", default_value="0.70"),
            DeclareLaunchArgument("dead_end_side_distance", default_value="0.80"),
            DeclareLaunchArgument("escape_override_enabled", default_value="true"),
            DeclareLaunchArgument("escape_blend_gain", default_value="0.90"),
            DeclareLaunchArgument("escape_linear_cap", default_value="0.05"),
            DeclareLaunchArgument("escape_min_turn", default_value="0.60"),
            DeclareLaunchArgument("escape_overlap_gate", default_value="0.55"),
            DeclareLaunchArgument("checkpoint_interval_steps", default_value="2000"),
            DeclareLaunchArgument(
                "checkpoint_dir",
                default_value=default_model_dir,
            ),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            stretch_sim_launch,
            rviz,
            goal_seeker_main,
        ]
    )
