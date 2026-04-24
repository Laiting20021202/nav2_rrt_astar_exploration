"""Main ROS 2 node for TD3-based goal-seeking navigation."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import torch
import torch.nn as nn
from visualization_msgs.msg import Marker

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency guard
    SummaryWriter = None

from .environment_node import GoalSeekerEnvironment, StepResult
from .td3_agent import TD3Agent, TD3Config


class ReferenceActorPolicy(nn.Module):
    """Reference actor architecture used by turtlebot3_drlnav pretrained models."""

    def __init__(self, state_dim: int, hidden_dim: int = 512, action_dim: int = 2) -> None:
        super().__init__()
        self.fa1 = nn.Linear(state_dim, hidden_dim)
        self.fa2 = nn.Linear(hidden_dim, hidden_dim)
        self.fa3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized action in [-1, 1]."""
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        return torch.tanh(self.fa3(x))


class GoalSeekerMainNode(Node):
    """Coordinates ROS I/O, environment stepping, and TD3 training/inference."""

    def __init__(self) -> None:
        super().__init__("goal_seeker_rl")
        self._declare_parameters()

        self.inference_mode = bool(self.get_parameter("inference_mode").value)
        self.policy_source = str(self.get_parameter("policy_source").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.reference_actor_path = str(self.get_parameter("reference_actor_path").value)
        self.reference_state_scan_samples = int(self.get_parameter("reference_state_scan_samples").value)
        self.reference_hidden_dim = int(self.get_parameter("reference_hidden_dim").value)
        self.state_scan_samples = int(self.get_parameter("state_scan_samples").value)
        self.append_prev_action_to_state = bool(self.get_parameter("append_prev_action_to_state").value)
        self.network_variant = str(self.get_parameter("network_variant").value)
        self.bootstrap_actor_path = str(self.get_parameter("bootstrap_actor_path").value)
        self.bootstrap_actor_strict = bool(self.get_parameter("bootstrap_actor_strict").value)
        self.linear_speed_max = float(self.get_parameter("linear_speed_max").value)
        self.reference_linear_speed_max = float(self.get_parameter("reference_linear_speed_max").value)
        self.angular_speed_max = float(self.get_parameter("angular_speed_max").value)
        self.reference_angular_speed_max = float(self.get_parameter("reference_angular_speed_max").value)
        self.inference_reset_on_stuck = bool(self.get_parameter("inference_reset_on_stuck").value)
        self.resume_model_path = str(self.get_parameter("resume_model_path").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_episode_steps = int(self.get_parameter("max_episode_steps").value)
        self.reset_pause_sec = float(self.get_parameter("reset_pause_sec").value)
        self.checkpoint_dir = Path(str(self.get_parameter("checkpoint_dir").value))
        self.checkpoint_interval_steps = int(self.get_parameter("checkpoint_interval_steps").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)
        self.reset_on_episode_end = bool(self.get_parameter("reset_on_episode_end").value)
        self.auto_goal_training = bool(self.get_parameter("auto_goal_training").value)
        self.auto_goal_min_radius = float(self.get_parameter("auto_goal_min_radius").value)
        self.auto_goal_max_radius = float(self.get_parameter("auto_goal_max_radius").value)
        self.auto_goal_max_abs_x = float(self.get_parameter("auto_goal_max_abs_x").value)
        self.auto_goal_max_abs_y = float(self.get_parameter("auto_goal_max_abs_y").value)
        self.auto_goal_curriculum_steps = int(self.get_parameter("auto_goal_curriculum_steps").value)
        self.auto_goal_start_scale = float(self.get_parameter("auto_goal_start_scale").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.goal_marker_frame = str(self.get_parameter("goal_marker_frame").value)
        self.publish_goal_marker = bool(self.get_parameter("publish_goal_marker").value)
        self.tensorboard_enabled = bool(self.get_parameter("tensorboard_enabled").value)
        self.tensorboard_log_dir = str(self.get_parameter("tensorboard_log_dir").value)
        self.tensorboard_flush_secs = int(self.get_parameter("tensorboard_flush_secs").value)
        self.success_window_size = int(self.get_parameter("success_window_size").value)
        self.escape_override_enabled = bool(self.get_parameter("escape_override_enabled").value)
        self.escape_blend_gain = float(self.get_parameter("escape_blend_gain").value)
        self.escape_linear_cap = float(self.get_parameter("escape_linear_cap").value)
        self.escape_min_turn = float(self.get_parameter("escape_min_turn").value)
        self.escape_overlap_gate = float(self.get_parameter("escape_overlap_gate").value)
        self.hybrid_exploration_enabled = bool(self.get_parameter("hybrid_exploration_enabled").value)
        self.hybrid_revisit_trigger = float(self.get_parameter("hybrid_revisit_trigger").value)
        self.hybrid_progress_window_sec = float(self.get_parameter("hybrid_progress_window_sec").value)
        self.hybrid_min_progress_delta = float(self.get_parameter("hybrid_min_progress_delta").value)
        self.hybrid_replan_cooldown_sec = float(self.get_parameter("hybrid_replan_cooldown_sec").value)
        self.hybrid_subgoal_timeout_sec = float(self.get_parameter("hybrid_subgoal_timeout_sec").value)
        self.hybrid_subgoal_min_distance = float(self.get_parameter("hybrid_subgoal_min_distance").value)
        self.hybrid_subgoal_max_distance = float(self.get_parameter("hybrid_subgoal_max_distance").value)
        self.hybrid_subgoal_reach_tolerance = float(self.get_parameter("hybrid_subgoal_reach_tolerance").value)
        self.hybrid_subgoal_safety_margin = float(self.get_parameter("hybrid_subgoal_safety_margin").value)
        self.hybrid_subgoal_goal_align_weight = float(self.get_parameter("hybrid_subgoal_goal_align_weight").value)
        self.hybrid_subgoal_gain_weight = float(self.get_parameter("hybrid_subgoal_gain_weight").value)
        self.hybrid_subgoal_revisit_weight = float(self.get_parameter("hybrid_subgoal_revisit_weight").value)
        self.hybrid_subgoal_repeat_penalty = float(self.get_parameter("hybrid_subgoal_repeat_penalty").value)
        self.hybrid_subgoal_random_topk = int(self.get_parameter("hybrid_subgoal_random_topk").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reset_service_name = str(self.get_parameter("reset_service_name").value)
        self._rng = np.random.default_rng(self.random_seed)

        self.use_reference_policy = self.inference_mode and self.policy_source == "reference_actor"
        lidar_samples = self.reference_state_scan_samples if self.use_reference_policy else self.state_scan_samples

        self.environment = GoalSeekerEnvironment(
            lidar_samples=lidar_samples,
            lidar_max_range=float(self.get_parameter("lidar_max_range").value),
            max_goal_distance=float(self.get_parameter("max_goal_distance").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
            collision_distance=float(self.get_parameter("collision_distance").value),
            stuck_window_sec=10.0,
            stuck_cell_size=float(self.get_parameter("stuck_cell_size").value),
            stuck_overlap_threshold=float(self.get_parameter("stuck_overlap_threshold").value),
            stuck_min_displacement=float(self.get_parameter("stuck_min_displacement").value),
            spin_filter_angular_threshold=0.5,
            spin_filter_min_range=0.15,
            episodic_memory_enabled=bool(self.get_parameter("episodic_memory_enabled").value),
            memory_cell_size=float(self.get_parameter("memory_cell_size").value),
            memory_novelty_reward=float(self.get_parameter("memory_novelty_reward").value),
            memory_revisit_penalty=float(self.get_parameter("memory_revisit_penalty").value),
            memory_revisit_saturation=int(self.get_parameter("memory_revisit_saturation").value),
            dead_end_front_distance=float(self.get_parameter("dead_end_front_distance").value),
            dead_end_side_distance=float(self.get_parameter("dead_end_side_distance").value),
            dead_end_clearance_margin=float(self.get_parameter("dead_end_clearance_margin").value),
            dead_end_front_angle_deg=float(self.get_parameter("dead_end_front_angle_deg").value),
            dead_end_side_min_angle_deg=float(self.get_parameter("dead_end_side_min_angle_deg").value),
            dead_end_side_max_angle_deg=float(self.get_parameter("dead_end_side_max_angle_deg").value),
            dead_end_revisit_gate=float(self.get_parameter("dead_end_revisit_gate").value),
        )
        self.append_prev_action_to_state = self.append_prev_action_to_state and (not self.use_reference_policy)
        self.policy_state_dim = self.environment.state_dim + (2 if self.append_prev_action_to_state else 0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")
        self.agent: Optional[TD3Agent] = None
        self.reference_policy: Optional[ReferenceActorPolicy] = None
        self.prev_action_norm = np.zeros(2, dtype=np.float32)

        if self.use_reference_policy:
            reference_state_dim = self.environment.state_dim + 2  # append previous action (lin, ang)
            self.reference_policy = ReferenceActorPolicy(
                state_dim=reference_state_dim,
                hidden_dim=self.reference_hidden_dim,
            ).to(device)
            if not self.reference_actor_path:
                raise RuntimeError("policy_source=reference_actor requires reference_actor_path in inference_mode.")
            state_dict = torch.load(self.reference_actor_path, map_location=device)
            self.reference_policy.load_state_dict(state_dict, strict=True)
            self.reference_policy.eval()
            self.get_logger().info(
                "Inference mode enabled with reference actor: "
                f"{self.reference_actor_path} (state_dim={reference_state_dim})"
            )
        else:
            td3_cfg = TD3Config(
                batch_size=int(self.get_parameter("batch_size").value),
                replay_size=int(self.get_parameter("replay_size").value),
                policy_noise=float(self.get_parameter("policy_noise").value),
                noise_clip=float(self.get_parameter("noise_clip").value),
                policy_delay=int(self.get_parameter("policy_delay").value),
                exploration_std=float(self.get_parameter("exploration_std").value),
                warmup_steps=self.warmup_steps,
                hidden_dim=int(self.get_parameter("hidden_dim").value),
                gamma=float(self.get_parameter("gamma").value),
                tau=float(self.get_parameter("tau").value),
                actor_lr=float(self.get_parameter("actor_lr").value),
                critic_lr=float(self.get_parameter("critic_lr").value),
                network_variant=self.network_variant,
            )
            self.agent = TD3Agent(
                state_dim=self.policy_state_dim,
                action_dim=2,
                device=device,
                config=td3_cfg,
            )

            if self.inference_mode:
                if not self.model_path:
                    raise RuntimeError("inference_mode=true requires a valid model_path.")
                self.agent.load(self.model_path)
                self.get_logger().info(f"Inference mode enabled. Loaded weights from: {self.model_path}")
            elif self.resume_model_path:
                self.agent.load(self.resume_model_path, strict=False)
                self.get_logger().info(f"Training resumed from: {self.resume_model_path}")
            elif self.bootstrap_actor_path:
                self.agent.load_actor(self.bootstrap_actor_path, strict=self.bootstrap_actor_strict)
                self.get_logger().info(
                    "Actor bootstrap loaded from reference weights: "
                    f"{self.bootstrap_actor_path} (strict={self.bootstrap_actor_strict})"
                )
            self.get_logger().info(
                "TD3 state config: "
                f"scan_samples={lidar_samples} policy_state_dim={self.policy_state_dim} "
                f"append_prev_action={self.append_prev_action_to_state} network_variant={self.network_variant}"
            )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.goal_marker_pub = (
            self.create_publisher(Marker, self.goal_marker_topic, 1) if self.publish_goal_marker else None
        )

        self.reset_client = None
        if self.reset_service_name:
            self.reset_client = self.create_client(Empty, self.reset_service_name)
        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self._control_loop)

        self.goal_active = False
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[np.ndarray] = None
        self.pause_until_sec = 0.0
        self.primary_goal_xy: Optional[tuple[float, float]] = None
        self.active_subgoal_xy: Optional[tuple[float, float]] = None
        self.current_target_is_subgoal = False
        self.subgoal_start_sec = 0.0
        self.last_replan_sec = -1e9
        self.progress_history: deque[tuple[float, float]] = deque()
        self.recent_subgoals: deque[tuple[float, float]] = deque(maxlen=12)

        self.total_steps = 0
        self.episode_index = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_checkpoint_step = 0
        self.last_train_info: Optional[dict] = None
        self.episode_outcomes: deque[int] = deque(maxlen=max(1, self.success_window_size))
        self.reason_counts: dict[str, int] = {"goal": 0, "collision": 0, "stuck": 0, "timeout": 0}
        self.tb_writer = None
        if self.tensorboard_enabled and SummaryWriter is not None:
            log_dir = Path(self.tensorboard_log_dir).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(log_dir), flush_secs=self.tensorboard_flush_secs)
            self.get_logger().info(f"TensorBoard logging enabled at: {log_dir}")
        elif self.tensorboard_enabled:
            self.get_logger().warn("TensorBoard disabled: torch.utils.tensorboard is unavailable in this environment.")

        self.get_logger().info("goal_seeker_rl node initialized. Waiting for /goal_pose ...")

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by the main node."""
        self.declare_parameter("inference_mode", False)
        self.declare_parameter("policy_source", "td3")
        self.declare_parameter("model_path", "")
        self.declare_parameter(
            "reference_actor_path",
            "/home/david/Desktop/laiting/rl_base_navigation/reference/turtlebot3_drlnav/src/turtlebot3_drl/model/examples/ddpg_0_stage9/actor_stage9_episode8000.pt",
        )
        self.declare_parameter("reference_state_scan_samples", 40)
        self.declare_parameter("reference_hidden_dim", 512)
        self.declare_parameter("state_scan_samples", 24)
        self.declare_parameter("append_prev_action_to_state", False)
        self.declare_parameter("network_variant", "default")
        self.declare_parameter("bootstrap_actor_path", "")
        self.declare_parameter("bootstrap_actor_strict", False)
        self.declare_parameter("resume_model_path", "")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("max_episode_steps", 1500)
        self.declare_parameter("reset_pause_sec", 1.5)

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("reset_service_name", "/reset_simulation")

        self.declare_parameter("lidar_max_range", 3.5)
        self.declare_parameter("max_goal_distance", 5.94)
        self.declare_parameter("goal_tolerance", 0.20)
        self.declare_parameter("collision_distance", 0.13)
        self.declare_parameter("stuck_cell_size", 0.10)
        self.declare_parameter("stuck_overlap_threshold", 0.70)
        self.declare_parameter("stuck_min_displacement", 0.50)
        self.declare_parameter("episodic_memory_enabled", True)
        self.declare_parameter("memory_cell_size", 0.20)
        self.declare_parameter("memory_novelty_reward", 0.40)
        self.declare_parameter("memory_revisit_penalty", 0.80)
        self.declare_parameter("memory_revisit_saturation", 4)
        self.declare_parameter("dead_end_front_distance", 0.70)
        self.declare_parameter("dead_end_side_distance", 0.80)
        self.declare_parameter("dead_end_clearance_margin", 0.20)
        self.declare_parameter("dead_end_front_angle_deg", 30.0)
        self.declare_parameter("dead_end_side_min_angle_deg", 45.0)
        self.declare_parameter("dead_end_side_max_angle_deg", 140.0)
        self.declare_parameter("dead_end_revisit_gate", 0.50)
        self.declare_parameter("linear_speed_max", 0.26)
        self.declare_parameter("reference_linear_speed_max", 0.22)
        self.declare_parameter("angular_speed_max", 1.0)
        self.declare_parameter("reference_angular_speed_max", 2.0)
        self.declare_parameter("inference_reset_on_stuck", False)
        self.declare_parameter("escape_override_enabled", True)
        self.declare_parameter("escape_blend_gain", 0.90)
        self.declare_parameter("escape_linear_cap", 0.05)
        self.declare_parameter("escape_min_turn", 0.60)
        self.declare_parameter("escape_overlap_gate", 0.55)
        self.declare_parameter("hybrid_exploration_enabled", True)
        self.declare_parameter("hybrid_revisit_trigger", 0.35)
        self.declare_parameter("hybrid_progress_window_sec", 12.0)
        self.declare_parameter("hybrid_min_progress_delta", 0.35)
        self.declare_parameter("hybrid_replan_cooldown_sec", 6.0)
        self.declare_parameter("hybrid_subgoal_timeout_sec", 30.0)
        self.declare_parameter("hybrid_subgoal_min_distance", 0.8)
        self.declare_parameter("hybrid_subgoal_max_distance", 2.5)
        self.declare_parameter("hybrid_subgoal_reach_tolerance", 0.45)
        self.declare_parameter("hybrid_subgoal_safety_margin", 0.25)
        self.declare_parameter("hybrid_subgoal_goal_align_weight", 0.55)
        self.declare_parameter("hybrid_subgoal_gain_weight", 1.40)
        self.declare_parameter("hybrid_subgoal_revisit_weight", 0.80)
        self.declare_parameter("hybrid_subgoal_repeat_penalty", 0.90)
        self.declare_parameter("hybrid_subgoal_random_topk", 4)

        self.declare_parameter("hidden_dim", 256)
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("replay_size", 200000)
        self.declare_parameter("actor_lr", 3e-4)
        self.declare_parameter("critic_lr", 3e-4)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("tau", 0.005)
        self.declare_parameter("policy_noise", 0.2)
        self.declare_parameter("noise_clip", 0.5)
        self.declare_parameter("policy_delay", 2)
        self.declare_parameter("exploration_std", 0.1)
        self.declare_parameter("warmup_steps", 2000)

        self.declare_parameter("checkpoint_dir", "checkpoints")
        self.declare_parameter("checkpoint_interval_steps", 5000)
        self.declare_parameter("reset_on_episode_end", True)
        self.declare_parameter("auto_goal_training", False)
        self.declare_parameter("auto_goal_min_radius", 0.8)
        self.declare_parameter("auto_goal_max_radius", 3.5)
        self.declare_parameter("auto_goal_max_abs_x", 8.0)
        self.declare_parameter("auto_goal_max_abs_y", 8.0)
        self.declare_parameter("auto_goal_curriculum_steps", 30000)
        self.declare_parameter("auto_goal_start_scale", 0.35)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("goal_marker_topic", "/goal_marker")
        self.declare_parameter("goal_marker_frame", "odom")
        self.declare_parameter("publish_goal_marker", True)
        self.declare_parameter("tensorboard_enabled", True)
        self.declare_parameter(
            "tensorboard_log_dir",
            "/home/david/Desktop/laiting/rl_base_navigation/src/goal_seeker_rl/model/tb",
        )
        self.declare_parameter("tensorboard_flush_secs", 5)
        self.declare_parameter("success_window_size", 20)

    def _scan_callback(self, msg: LaserScan) -> None:
        """Forward scan messages to environment module."""
        self.environment.update_scan(msg)

    def _odom_callback(self, msg: Odometry) -> None:
        """Forward odometry messages to environment module."""
        self.environment.update_odom(msg, self._now_sec())

    def _goal_callback(self, msg: PoseStamped) -> None:
        """Start navigation immediately once a goal pose is received."""
        now_sec = self._now_sec()
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)
        self.primary_goal_xy = (gx, gy)
        self.progress_history.clear()
        self.recent_subgoals.clear()
        self._activate_goal(gx, gy, now_sec, source="RViz", reset_episode=True, is_subgoal=False)

    def _control_loop(self) -> None:
        """Main control/training/inference loop."""
        now_sec = self._now_sec()

        if now_sec < self.pause_until_sec:
            self._publish_stop()
            return

        if not self.goal_active:
            if (
                self.auto_goal_training
                and (not self.inference_mode)
                and self.environment.has_odom
                and self.environment.has_scan
            ):
                self._set_auto_goal(now_sec)
            else:
                return

        if not self.environment.ready_for_control():
            return

        result = self.environment.evaluate_step(now_sec)
        policy_state = self._compose_policy_state(result.state)
        if self.inference_mode and not self.inference_reset_on_stuck and result.reason == "stuck":
            # Keep trying in inference instead of stopping too early.
            result.done = False
            result.reason = "running"
        if self._handle_hybrid_navigation(now_sec, result):
            return

        # The result at tick t is treated as outcome of action from tick t-1.
        if self.last_state is not None and self.last_action is not None:
            self._consume_transition(result, policy_state)

        timed_out = self.episode_steps >= self.max_episode_steps
        if result.done or timed_out:
            reason = result.reason if result.done else "timeout"
            self._finish_episode(reason, result)
            return

        action_norm = self._select_action(policy_state)
        self._publish_action(action_norm)
        self.last_state = policy_state
        self.last_action = action_norm

    def _consume_transition(self, result: StepResult, next_policy_state: np.ndarray) -> None:
        """Store transition and optionally train the TD3 model."""
        if not self.inference_mode:
            if self.agent is None:
                return
            self.agent.store_transition(
                state=self.last_state,
                action=self.last_action,
                reward=result.reward,
                next_state=next_policy_state,
                done=result.done,
            )

            if self.total_steps >= self.warmup_steps:
                self.last_train_info = self.agent.train_step()

        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += result.reward

    def _handle_hybrid_navigation(self, now_sec: float, result: StepResult) -> bool:
        """Switch to temporary exploration sub-goals when final-goal progress stalls."""
        if (not self.inference_mode) or (not self.hybrid_exploration_enabled):
            return False
        if self.primary_goal_xy is None:
            return False

        final_distance = self._distance_to_point(*self.primary_goal_xy)
        self._update_progress_history(now_sec, final_distance)

        if self.current_target_is_subgoal:
            reached_subgoal = result.done and result.reason == "goal"
            if self.active_subgoal_xy is not None:
                reached_subgoal = reached_subgoal or (
                    self._distance_to_point(*self.active_subgoal_xy) <= self.hybrid_subgoal_reach_tolerance
                )
            if reached_subgoal:
                self._restore_primary_goal(now_sec, reason="subgoal reached")
                return True
            if (now_sec - self.subgoal_start_sec) >= self.hybrid_subgoal_timeout_sec:
                self._restore_primary_goal(now_sec, reason="subgoal timeout")
                self.last_replan_sec = now_sec
                return True
            return False

        if result.done and result.reason == "goal":
            return False
        if (now_sec - self.last_replan_sec) < self.hybrid_replan_cooldown_sec:
            return False

        stagnating, progress_delta = self._is_progress_stagnating()
        revisit_high = self.environment.last_revisit_ratio >= self.hybrid_revisit_trigger
        if not (stagnating or revisit_high):
            return False

        candidate = self._sample_exploration_subgoal()
        if candidate is None:
            return False

        gx, gy, score = candidate
        self.last_replan_sec = now_sec
        self._activate_goal(gx, gy, now_sec, source="hybrid subgoal", reset_episode=False, is_subgoal=True)
        self.get_logger().info(
            "Hybrid exploration subgoal | "
            f"x={gx:.2f} y={gy:.2f} score={score:.3f} "
            f"final_dist={final_distance:.3f} progress_delta={progress_delta:.3f} "
            f"revisit={self.environment.last_revisit_ratio:.2f}"
        )
        return True

    def _update_progress_history(self, now_sec: float, distance_to_primary_goal: float) -> None:
        """Track recent distance-to-goal trend for stagnation detection."""
        self.progress_history.append((now_sec, distance_to_primary_goal))
        min_t = now_sec - self.hybrid_progress_window_sec
        while self.progress_history and self.progress_history[0][0] < min_t:
            self.progress_history.popleft()

    def _is_progress_stagnating(self) -> tuple[bool, float]:
        """Return whether final-goal progress is below threshold over the configured window."""
        if len(self.progress_history) < 2:
            return False, 0.0
        window_dt = self.progress_history[-1][0] - self.progress_history[0][0]
        progress_delta = self.progress_history[0][1] - self.progress_history[-1][1]
        if window_dt < self.hybrid_progress_window_sec * 0.7:
            return False, progress_delta
        far_enough = self.progress_history[-1][1] > (self.environment.goal_tolerance * 1.5)
        stagnating = far_enough and (progress_delta < self.hybrid_min_progress_delta)
        return stagnating, progress_delta

    def _sample_exploration_subgoal(self) -> Optional[tuple[float, float, float]]:
        """Pick a LiDAR-visible waypoint that is open, less revisited, and somewhat goal-aligned."""
        if self.primary_goal_xy is None:
            return None

        scan = self.environment.scan_sampled
        rel_angles = self.environment.scan_angles
        if len(scan) == 0 or len(scan) != len(rel_angles):
            return None

        min_d = max(0.20, self.hybrid_subgoal_min_distance)
        max_d = max(min_d, self.hybrid_subgoal_max_distance)
        revisit_scale = max(1.0, float(self.environment.memory_revisit_saturation))
        current_final_dist = self._distance_to_point(*self.primary_goal_xy)
        goal_bearing = math.atan2(
            self.primary_goal_xy[1] - self.environment.robot_y,
            self.primary_goal_xy[0] - self.environment.robot_x,
        )
        candidates: list[tuple[float, float, float, float, float]] = []

        for clear, rel_angle in zip(scan, rel_angles):
            clear_f = float(clear)
            if clear_f < (min_d + self.hybrid_subgoal_safety_margin):
                continue

            travel = min(max_d, clear_f - self.hybrid_subgoal_safety_margin)
            if travel < min_d:
                continue

            world_angle = self.environment.robot_yaw + float(rel_angle)
            gx = float(self.environment.robot_x + travel * math.cos(world_angle))
            gy = float(self.environment.robot_y + travel * math.sin(world_angle))
            if abs(gx) > self.auto_goal_max_abs_x or abs(gy) > self.auto_goal_max_abs_y:
                continue

            openness = float(
                np.clip(
                    (clear_f - min_d) / max(1e-6, self.environment.lidar_max_range - min_d),
                    0.0,
                    1.0,
                )
            )
            alignment = 0.5 * (1.0 + math.cos(self._wrap_angle(goal_bearing - world_angle)))
            candidate_final_dist = math.hypot(self.primary_goal_xy[0] - gx, self.primary_goal_xy[1] - gy)
            distance_gain = float(np.clip((current_final_dist - candidate_final_dist) / max(0.5, max_d), -1.0, 1.0))
            if alignment < 0.15 and distance_gain < 0.20:
                # Avoid aggressively exploring behind the goal unless it clearly improves distance.
                continue
            visits = float(self.environment.get_visit_count(gx, gy))
            revisit_penalty = float(np.clip(visits / revisit_scale, 0.0, 1.0))
            repeat_penalty = 0.0
            for sx, sy in self.recent_subgoals:
                d = math.hypot(gx - sx, gy - sy)
                if d < 1.2:
                    repeat_penalty = max(repeat_penalty, (1.2 - d) / 1.2)
            score = openness + (self.hybrid_subgoal_goal_align_weight * alignment)
            score += self.hybrid_subgoal_gain_weight * distance_gain
            score -= self.hybrid_subgoal_revisit_weight * revisit_penalty
            score -= self.hybrid_subgoal_repeat_penalty * repeat_penalty
            candidates.append((score, distance_gain, alignment, gx, gy))

        if not candidates:
            return None

        preferred = [c for c in candidates if c[1] >= 0.0 and c[2] >= 0.30]
        if not preferred:
            preferred = [c for c in candidates if c[1] >= -0.05 and c[2] >= 0.20]
        pool_source = preferred if preferred else candidates
        pool_source.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, min(self.hybrid_subgoal_random_topk, len(pool_source)))
        picked = pool_source[int(self._rng.integers(0, top_k))]
        return picked[3], picked[4], picked[0]

    def _restore_primary_goal(self, now_sec: float, reason: str) -> None:
        """Switch control target back to the original user goal after exploration."""
        if self.primary_goal_xy is None:
            return
        gx, gy = self.primary_goal_xy
        self._activate_goal(gx, gy, now_sec, source=f"resume primary ({reason})", reset_episode=False, is_subgoal=False)

    def _distance_to_point(self, x: float, y: float) -> float:
        """Return Euclidean distance from robot to a world point."""
        return math.hypot(x - self.environment.robot_x, y - self.environment.robot_y)

    def _finish_episode(self, reason: str, result: StepResult) -> None:
        """Stop robot, log episode summary, and handle reset behavior."""
        self._publish_stop()

        critic_loss = None
        actor_loss = None
        if self.last_train_info is not None:
            critic_loss = self.last_train_info.get("critic_loss")
            actor_loss = self.last_train_info.get("actor_loss")

        is_goal = int(reason == "goal")
        self.episode_outcomes.append(is_goal)
        success_rate = float(sum(self.episode_outcomes) / max(1, len(self.episode_outcomes)))
        if reason not in self.reason_counts:
            self.reason_counts[reason] = 0
        self.reason_counts[reason] += 1

        self.get_logger().info(
            "Episode summary | "
            f"steps={self.episode_steps} reward={self.episode_reward:.2f} reason={reason} "
            f"goal_dist={result.goal_distance:.3f} heading={result.heading_angle:.3f} "
            f"min_scan={result.min_obstacle_distance:.3f} overlap={self.environment.last_overlap_ratio:.2f} "
            f"revisit={self.environment.last_revisit_ratio:.2f} dead_end={int(self.environment.dead_end_detected)} "
            f"success_rate@{len(self.episode_outcomes)}={success_rate:.2f} "
            f"critic_loss={critic_loss} actor_loss={actor_loss}"
        )
        self._log_episode_tensorboard(reason, success_rate, result, critic_loss, actor_loss)

        self.last_state = None
        self.last_action = None
        self.prev_action_norm = np.zeros(2, dtype=np.float32)
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.goal_active = False
        self.environment.clear_goal()
        self.primary_goal_xy = None
        self.active_subgoal_xy = None
        self.current_target_is_subgoal = False
        self.progress_history.clear()
        self.recent_subgoals.clear()
        self._clear_subgoal_marker()

        if self.inference_mode:
            return

        self.environment.register_reset(self._now_sec())
        if self.reset_on_episode_end and reason in {"stuck", "collision", "timeout", "goal"}:
            self._reset_simulation()
        self.pause_until_sec = self._now_sec() + self.reset_pause_sec

        self._maybe_save_checkpoint()

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Choose action according to mode and exploration phase."""
        if self.use_reference_policy:
            if self.reference_policy is None:
                raise RuntimeError("Reference policy was not initialized.")
            policy_input = np.concatenate([state.astype(np.float32), self.prev_action_norm], axis=0)
            with torch.no_grad():
                state_t = torch.as_tensor(policy_input, dtype=torch.float32, device=next(self.reference_policy.parameters()).device)
                action = self.reference_policy(state_t.unsqueeze(0)).cpu().numpy()[0]
            return np.clip(action, -1.0, 1.0).astype(np.float32)

        if self.inference_mode:
            if self.agent is None:
                raise RuntimeError("TD3 agent was not initialized.")
            return self.agent.select_action(state, explore=False)

        if self.total_steps < self.warmup_steps:
            if self.agent is None:
                raise RuntimeError("TD3 agent was not initialized.")
            return self.agent.sample_random_action()

        if self.agent is None:
            raise RuntimeError("TD3 agent was not initialized.")
        return self.agent.select_action(state, explore=True)

    def _compose_policy_state(self, state: np.ndarray) -> np.ndarray:
        """Optionally append previous action to match reference training state design."""
        if not self.append_prev_action_to_state:
            return state
        return np.concatenate([state.astype(np.float32), self.prev_action_norm], axis=0)

    def _publish_action(self, action_norm: np.ndarray) -> None:
        """Map normalized action to robot velocity command and publish."""
        action = np.clip(action_norm, -1.0, 1.0)
        linear_cap = self.reference_linear_speed_max if self.use_reference_policy else self.linear_speed_max
        angular_cap = self.reference_angular_speed_max if self.use_reference_policy else self.angular_speed_max
        linear_x = float((action[0] + 1.0) * 0.5 * linear_cap)
        angular_z = float(np.clip(action[1] * angular_cap, -angular_cap, angular_cap))
        linear_x, angular_z = self._apply_escape_override(linear_x, angular_z, angular_cap)

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)
        linear_norm = float(np.clip((linear_x / max(1e-6, linear_cap)) * 2.0 - 1.0, -1.0, 1.0))
        angular_norm = float(np.clip(angular_z / max(1e-6, angular_cap), -1.0, 1.0))
        self.prev_action_norm = np.array([linear_norm, angular_norm], dtype=np.float32)

    def _apply_escape_override(self, linear_x: float, angular_z: float, angular_cap: float) -> tuple[float, float]:
        """Bias action away from repeated dead-end behavior using episodic memory cues."""
        if not self.escape_override_enabled:
            return linear_x, angular_z

        dead_end, turn_hint, dead_end_intensity = self.environment.get_escape_signal()
        overlap_ratio = self.environment.last_overlap_ratio
        revisit_ratio = self.environment.last_revisit_ratio
        should_escape = dead_end or (overlap_ratio >= self.escape_overlap_gate)
        if not should_escape:
            return linear_x, angular_z

        if abs(turn_hint) < 1e-3:
            turn_hint = 1.0 if self.prev_action_norm[1] <= 0.0 else -1.0

        escape_strength = float(np.clip(max(dead_end_intensity, overlap_ratio, revisit_ratio), 0.0, 1.0))
        blend = float(np.clip(self.escape_blend_gain * escape_strength, 0.0, 1.0))
        target_turn = turn_hint * max(self.escape_min_turn * angular_cap, abs(angular_z))
        angular_z = float(np.clip((1.0 - blend) * angular_z + blend * target_turn, -angular_cap, angular_cap))
        linear_x = float(min(linear_x, self.escape_linear_cap))
        return linear_x, angular_z

    def _publish_stop(self) -> None:
        """Publish zero velocity command."""
        self.cmd_pub.publish(Twist())

    def _reset_simulation(self) -> None:
        """Call Gazebo simulation reset service asynchronously."""
        if not self.reset_service_name:
            return
        if self.reset_client is None:
            return
        if not self.reset_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Reset service unavailable: {self.reset_service_name}")
            return
        self.reset_client.call_async(Empty.Request())

    def _activate_goal(
        self,
        gx: float,
        gy: float,
        now_sec: float,
        source: str,
        reset_episode: bool,
        is_subgoal: bool,
    ) -> None:
        """Set control target (final goal or temporary sub-goal)."""
        self.environment.set_goal(gx, gy, now_sec)
        self.goal_active = True
        self.last_state = None
        self.last_action = None
        self.prev_action_norm = np.zeros(2, dtype=np.float32)

        if reset_episode:
            self.episode_steps = 0
            self.episode_reward = 0.0
            self.episode_index += 1

        if is_subgoal:
            self.current_target_is_subgoal = True
            self.active_subgoal_xy = (gx, gy)
            self.subgoal_start_sec = now_sec
            self.recent_subgoals.append((gx, gy))
            self._publish_subgoal_marker(gx, gy)
            self.get_logger().info(
                f"Subgoal activated ({source}): x={gx:.2f}, y={gy:.2f}. "
                f"Primary goal remains x={self.primary_goal_xy[0]:.2f}, y={self.primary_goal_xy[1]:.2f}."
            )
            return

        self.current_target_is_subgoal = False
        self.active_subgoal_xy = None
        self._clear_subgoal_marker()
        self._publish_goal_marker(gx, gy)
        if reset_episode:
            self.get_logger().info(
                f"Goal #{self.episode_index} received from {source}: x={gx:.2f}, y={gy:.2f}. Navigation started."
            )
        else:
            self.get_logger().info(f"Primary goal restored ({source}): x={gx:.2f}, y={gy:.2f}.")

    def _set_auto_goal(self, now_sec: float) -> None:
        """Sample a training goal around the robot when auto-goal mode is enabled."""
        min_r = max(0.1, self.auto_goal_min_radius)
        full_max_r = max(min_r, self.auto_goal_max_radius)
        curriculum_steps = max(1, self.auto_goal_curriculum_steps)
        progress = float(np.clip(self.total_steps / float(curriculum_steps), 0.0, 1.0))
        start_scale = float(np.clip(self.auto_goal_start_scale, 0.1, 1.0))
        scale = start_scale + (1.0 - start_scale) * progress
        max_r = float(min_r + (full_max_r - min_r) * scale)

        # Prefer LiDAR-visible goals to avoid sampling across walls in maze-like worlds.
        scan = self.environment.scan_sampled
        angles = self.environment.scan_angles
        if len(scan) == len(angles) and len(scan) > 0:
            safety_margin = max(
                0.25,
                2.0 * float(self.get_parameter("collision_distance").value),
                1.5 * float(self.get_parameter("goal_tolerance").value),
            )
            visible_idx = np.where(scan > (min_r + safety_margin))[0]
            if len(visible_idx) > 0:
                for _ in range(80):
                    idx = int(self._rng.choice(visible_idx))
                    beam_clear = float(scan[idx] - safety_margin)
                    if beam_clear <= min_r:
                        continue
                    radius = float(self._rng.uniform(min_r, min(max_r, beam_clear)))
                    world_angle = float(self.environment.robot_yaw + angles[idx])
                    gx = float(self.environment.robot_x + radius * np.cos(world_angle))
                    gy = float(self.environment.robot_y + radius * np.sin(world_angle))
                    if abs(gx) <= self.auto_goal_max_abs_x and abs(gy) <= self.auto_goal_max_abs_y:
                        self.primary_goal_xy = (gx, gy)
                        self.progress_history.clear()
                        self.recent_subgoals.clear()
                        self._activate_goal(
                            gx, gy, now_sec, source="auto lidar curriculum", reset_episode=True, is_subgoal=False
                        )
                        return

        for _ in range(50):
            radius = float(self._rng.uniform(min_r, max_r))
            angle = float(self._rng.uniform(-np.pi, np.pi))
            gx = float(self.environment.robot_x + radius * np.cos(angle))
            gy = float(self.environment.robot_y + radius * np.sin(angle))
            if abs(gx) <= self.auto_goal_max_abs_x and abs(gy) <= self.auto_goal_max_abs_y:
                self.primary_goal_xy = (gx, gy)
                self.progress_history.clear()
                self.recent_subgoals.clear()
                self._activate_goal(gx, gy, now_sec, source="auto curriculum", reset_episode=True, is_subgoal=False)
                return

        # Fallback: clip into configured workspace bounds.
        gx = float(np.clip(self.environment.robot_x + max_r, -self.auto_goal_max_abs_x, self.auto_goal_max_abs_x))
        gy = float(np.clip(self.environment.robot_y, -self.auto_goal_max_abs_y, self.auto_goal_max_abs_y))
        self.primary_goal_xy = (gx, gy)
        self.progress_history.clear()
        self.recent_subgoals.clear()
        self._activate_goal(gx, gy, now_sec, source="auto fallback", reset_episode=True, is_subgoal=False)

    def _publish_goal_marker(self, gx: float, gy: float) -> None:
        """Publish a marker so the active goal is always visible in RViz."""
        if self.goal_marker_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = self.goal_marker_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_seeker"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.a = 0.95
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.2
        self.goal_marker_pub.publish(marker)

    def _publish_subgoal_marker(self, gx: float, gy: float) -> None:
        """Publish temporary exploration sub-goal marker."""
        if self.goal_marker_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = self.goal_marker_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_seeker"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20
        marker.color.a = 0.9
        marker.color.r = 0.15
        marker.color.g = 0.95
        marker.color.b = 0.95
        self.goal_marker_pub.publish(marker)

    def _clear_subgoal_marker(self) -> None:
        """Hide temporary exploration marker when not in use."""
        if self.goal_marker_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = self.goal_marker_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_seeker"
        marker.id = 1
        marker.action = Marker.DELETE
        self.goal_marker_pub.publish(marker)

    def _maybe_save_checkpoint(self) -> None:
        """Persist model periodically during training."""
        if self.inference_mode:
            return

        if (self.total_steps - self.last_checkpoint_step) < self.checkpoint_interval_steps:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = self.checkpoint_dir / "td3_latest.pth"
        step_path = self.checkpoint_dir / f"td3_step_{self.total_steps}.pth"
        self.agent.save(str(latest_path))
        self.agent.save(str(step_path))
        self.last_checkpoint_step = self.total_steps
        self.get_logger().info(f"Checkpoint saved: {step_path}")

    def _log_episode_tensorboard(
        self,
        reason: str,
        success_rate: float,
        result: StepResult,
        critic_loss: Optional[float],
        actor_loss: Optional[float],
    ) -> None:
        """Write key training and navigation metrics to TensorBoard."""
        if self.tb_writer is None:
            return
        step = max(0, self.total_steps)
        self.tb_writer.add_scalar("episode/reward", self.episode_reward, step)
        self.tb_writer.add_scalar("episode/steps", self.episode_steps, step)
        self.tb_writer.add_scalar("episode/success_rate_window", success_rate, step)
        self.tb_writer.add_scalar("episode/goal_distance_end", result.goal_distance, step)
        self.tb_writer.add_scalar("episode/min_scan_end", result.min_obstacle_distance, step)
        self.tb_writer.add_scalar("episode/overlap_ratio", self.environment.last_overlap_ratio, step)
        self.tb_writer.add_scalar("episode/revisit_ratio", self.environment.last_revisit_ratio, step)
        self.tb_writer.add_scalar("episode/dead_end_detected", float(int(self.environment.dead_end_detected)), step)
        reason_map = {"goal": 0.0, "collision": 1.0, "stuck": 2.0, "timeout": 3.0, "running": 4.0, "idle": 5.0}
        self.tb_writer.add_scalar("episode/reason_code", reason_map.get(reason, 9.0), step)
        self.tb_writer.add_scalar("stats/goal_count", float(self.reason_counts.get("goal", 0)), step)
        self.tb_writer.add_scalar("stats/collision_count", float(self.reason_counts.get("collision", 0)), step)
        self.tb_writer.add_scalar("stats/stuck_count", float(self.reason_counts.get("stuck", 0)), step)
        self.tb_writer.add_scalar("stats/timeout_count", float(self.reason_counts.get("timeout", 0)), step)
        if critic_loss is not None:
            self.tb_writer.add_scalar("loss/critic", critic_loss, step)
        if actor_loss is not None:
            self.tb_writer.add_scalar("loss/actor", actor_loss, step)
        self.tb_writer.flush()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _now_sec(self) -> float:
        """Return current ROS time in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args: Optional[list[str]] = None) -> None:
    """ROS 2 entrypoint."""
    rclpy.init(args=args)
    node = GoalSeekerMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.inference_mode:
            node._maybe_save_checkpoint()
        if node.tb_writer is not None:
            node.tb_writer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
