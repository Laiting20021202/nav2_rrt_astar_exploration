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
    hidden_dim = LaunchConfiguration("hidden_dim")
    batch_size = LaunchConfiguration("batch_size")
    replay_size = LaunchConfiguration("replay_size")
    actor_lr = LaunchConfiguration("actor_lr")
    critic_lr = LaunchConfiguration("critic_lr")
    gamma = LaunchConfiguration("gamma")
    tau = LaunchConfiguration("tau")
    policy_noise = LaunchConfiguration("policy_noise")
    noise_clip = LaunchConfiguration("noise_clip")
    policy_delay = LaunchConfiguration("policy_delay")
    exploration_std = LaunchConfiguration("exploration_std")
    checkpoint_interval_steps = LaunchConfiguration("checkpoint_interval_steps")
    max_episode_steps = LaunchConfiguration("max_episode_steps")
    warmup_steps = LaunchConfiguration("warmup_steps")
    auto_goal_training = LaunchConfiguration("auto_goal_training")
    auto_goal_min_radius = LaunchConfiguration("auto_goal_min_radius")
    auto_goal_max_radius = LaunchConfiguration("auto_goal_max_radius")
    auto_goal_curriculum_steps = LaunchConfiguration("auto_goal_curriculum_steps")
    auto_goal_start_scale = LaunchConfiguration("auto_goal_start_scale")
    goal_tolerance = LaunchConfiguration("goal_tolerance")
    collision_distance = LaunchConfiguration("collision_distance")
    linear_speed_max = LaunchConfiguration("linear_speed_max")
    angular_speed_max = LaunchConfiguration("angular_speed_max")
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
    hybrid_exploration_enabled = LaunchConfiguration("hybrid_exploration_enabled")
    hybrid_revisit_trigger = LaunchConfiguration("hybrid_revisit_trigger")
    hybrid_progress_window_sec = LaunchConfiguration("hybrid_progress_window_sec")
    hybrid_min_progress_delta = LaunchConfiguration("hybrid_min_progress_delta")
    hybrid_replan_cooldown_sec = LaunchConfiguration("hybrid_replan_cooldown_sec")
    hybrid_subgoal_timeout_sec = LaunchConfiguration("hybrid_subgoal_timeout_sec")
    hybrid_subgoal_min_distance = LaunchConfiguration("hybrid_subgoal_min_distance")
    hybrid_subgoal_max_distance = LaunchConfiguration("hybrid_subgoal_max_distance")
    hybrid_subgoal_reach_tolerance = LaunchConfiguration("hybrid_subgoal_reach_tolerance")
    hybrid_subgoal_safety_margin = LaunchConfiguration("hybrid_subgoal_safety_margin")
    hybrid_subgoal_goal_align_weight = LaunchConfiguration("hybrid_subgoal_goal_align_weight")
    hybrid_subgoal_gain_weight = LaunchConfiguration("hybrid_subgoal_gain_weight")
    hybrid_subgoal_revisit_weight = LaunchConfiguration("hybrid_subgoal_revisit_weight")
    hybrid_subgoal_repeat_penalty = LaunchConfiguration("hybrid_subgoal_repeat_penalty")
    hybrid_subgoal_random_topk = LaunchConfiguration("hybrid_subgoal_random_topk")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    world = LaunchConfiguration("world")
    rviz_config = LaunchConfiguration("rviz_config")

    default_world = PathJoinSubstitution(
        [
            FindPackageShare("turtlebot3_gazebo"),
            "worlds",
            "turtlebot3_houses",
            "waffle.model",
        ]
    )
    default_rviz = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "rviz", "nav_config.rviz"])
    robot_urdf = PathJoinSubstitution([FindPackageShare("goal_seeker_rl"), "urdf", "turtlebot3_waffle_minimal.urdf"])

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
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "replay_size": replay_size,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "tau": tau,
                "policy_noise": policy_noise,
                "noise_clip": noise_clip,
                "policy_delay": policy_delay,
                "exploration_std": exploration_std,
                "checkpoint_interval_steps": checkpoint_interval_steps,
                "max_episode_steps": max_episode_steps,
                "warmup_steps": warmup_steps,
                "auto_goal_training": auto_goal_training,
                "auto_goal_min_radius": auto_goal_min_radius,
                "auto_goal_max_radius": auto_goal_max_radius,
                "auto_goal_curriculum_steps": auto_goal_curriculum_steps,
                "auto_goal_start_scale": auto_goal_start_scale,
                "goal_tolerance": goal_tolerance,
                "collision_distance": collision_distance,
                "linear_speed_max": linear_speed_max,
                "angular_speed_max": angular_speed_max,
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
                "hybrid_exploration_enabled": hybrid_exploration_enabled,
                "hybrid_revisit_trigger": hybrid_revisit_trigger,
                "hybrid_progress_window_sec": hybrid_progress_window_sec,
                "hybrid_min_progress_delta": hybrid_min_progress_delta,
                "hybrid_replan_cooldown_sec": hybrid_replan_cooldown_sec,
                "hybrid_subgoal_timeout_sec": hybrid_subgoal_timeout_sec,
                "hybrid_subgoal_min_distance": hybrid_subgoal_min_distance,
                "hybrid_subgoal_max_distance": hybrid_subgoal_max_distance,
                "hybrid_subgoal_reach_tolerance": hybrid_subgoal_reach_tolerance,
                "hybrid_subgoal_safety_margin": hybrid_subgoal_safety_margin,
                "hybrid_subgoal_goal_align_weight": hybrid_subgoal_goal_align_weight,
                "hybrid_subgoal_gain_weight": hybrid_subgoal_gain_weight,
                "hybrid_subgoal_revisit_weight": hybrid_subgoal_revisit_weight,
                "hybrid_subgoal_repeat_penalty": hybrid_subgoal_repeat_penalty,
                "hybrid_subgoal_random_topk": hybrid_subgoal_random_topk,
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
            DeclareLaunchArgument("policy_source", default_value="td3"),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument(
                "reference_actor_path",
                default_value="/home/david/Desktop/laiting/rl_base_navigation/reference/turtlebot3_drlnav/src/turtlebot3_drl/model/examples/ddpg_0_stage9/actor_stage9_episode8000.pt",
            ),
            DeclareLaunchArgument("reference_state_scan_samples", default_value="40"),
            DeclareLaunchArgument("state_scan_samples", default_value="24"),
            DeclareLaunchArgument("append_prev_action_to_state", default_value="false"),
            DeclareLaunchArgument("network_variant", default_value="default"),
            DeclareLaunchArgument("bootstrap_actor_path", default_value=""),
            DeclareLaunchArgument("bootstrap_actor_strict", default_value="false"),
            DeclareLaunchArgument("resume_model_path", default_value=""),
            DeclareLaunchArgument("hidden_dim", default_value="256"),
            DeclareLaunchArgument("batch_size", default_value="128"),
            DeclareLaunchArgument("replay_size", default_value="200000"),
            DeclareLaunchArgument("actor_lr", default_value="0.0003"),
            DeclareLaunchArgument("critic_lr", default_value="0.0003"),
            DeclareLaunchArgument("gamma", default_value="0.99"),
            DeclareLaunchArgument("tau", default_value="0.005"),
            DeclareLaunchArgument("policy_noise", default_value="0.2"),
            DeclareLaunchArgument("noise_clip", default_value="0.5"),
            DeclareLaunchArgument("policy_delay", default_value="2"),
            DeclareLaunchArgument("exploration_std", default_value="0.1"),
            DeclareLaunchArgument("checkpoint_interval_steps", default_value="2000"),
            DeclareLaunchArgument("max_episode_steps", default_value="1200"),
            DeclareLaunchArgument("warmup_steps", default_value="2000"),
            DeclareLaunchArgument("auto_goal_training", default_value="false"),
            DeclareLaunchArgument("auto_goal_min_radius", default_value="0.8"),
            DeclareLaunchArgument("auto_goal_max_radius", default_value="3.5"),
            DeclareLaunchArgument("auto_goal_curriculum_steps", default_value="30000"),
            DeclareLaunchArgument("auto_goal_start_scale", default_value="0.35"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.20"),
            DeclareLaunchArgument("collision_distance", default_value="0.13"),
            DeclareLaunchArgument("linear_speed_max", default_value="0.22"),
            DeclareLaunchArgument("angular_speed_max", default_value="2.0"),
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
            DeclareLaunchArgument("hybrid_exploration_enabled", default_value="true"),
            DeclareLaunchArgument("hybrid_revisit_trigger", default_value="0.35"),
            DeclareLaunchArgument("hybrid_progress_window_sec", default_value="12.0"),
            DeclareLaunchArgument("hybrid_min_progress_delta", default_value="0.35"),
            DeclareLaunchArgument("hybrid_replan_cooldown_sec", default_value="6.0"),
            DeclareLaunchArgument("hybrid_subgoal_timeout_sec", default_value="30.0"),
            DeclareLaunchArgument("hybrid_subgoal_min_distance", default_value="0.8"),
            DeclareLaunchArgument("hybrid_subgoal_max_distance", default_value="2.5"),
            DeclareLaunchArgument("hybrid_subgoal_reach_tolerance", default_value="0.45"),
            DeclareLaunchArgument("hybrid_subgoal_safety_margin", default_value="0.25"),
            DeclareLaunchArgument("hybrid_subgoal_goal_align_weight", default_value="0.55"),
            DeclareLaunchArgument("hybrid_subgoal_gain_weight", default_value="1.40"),
            DeclareLaunchArgument("hybrid_subgoal_revisit_weight", default_value="0.80"),
            DeclareLaunchArgument("hybrid_subgoal_repeat_penalty", default_value="0.90"),
            DeclareLaunchArgument("hybrid_subgoal_random_topk", default_value="4"),
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
