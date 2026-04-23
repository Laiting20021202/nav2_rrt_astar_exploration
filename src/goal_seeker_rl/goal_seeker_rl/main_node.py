"""Main ROS 2 node for TD3-based goal-seeking navigation."""

from __future__ import annotations

from collections import deque
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
        self._activate_goal(gx, gy, now_sec, source="RViz")

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

    def _activate_goal(self, gx: float, gy: float, now_sec: float, source: str) -> None:
        """Apply a new goal and reset episode bookkeeping."""
        self.environment.set_goal(gx, gy, now_sec)
        self.goal_active = True
        self.last_state = None
        self.last_action = None
        self.prev_action_norm = np.zeros(2, dtype=np.float32)
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_index += 1
        self._publish_goal_marker(gx, gy)
        self.get_logger().info(
            f"Goal #{self.episode_index} received from {source}: x={gx:.2f}, y={gy:.2f}. Navigation started."
        )

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
                        self._activate_goal(gx, gy, now_sec, source="auto lidar curriculum")
                        return

        for _ in range(50):
            radius = float(self._rng.uniform(min_r, max_r))
            angle = float(self._rng.uniform(-np.pi, np.pi))
            gx = float(self.environment.robot_x + radius * np.cos(angle))
            gy = float(self.environment.robot_y + radius * np.sin(angle))
            if abs(gx) <= self.auto_goal_max_abs_x and abs(gy) <= self.auto_goal_max_abs_y:
                self._activate_goal(gx, gy, now_sec, source="auto curriculum")
                return

        # Fallback: clip into configured workspace bounds.
        gx = float(np.clip(self.environment.robot_x + max_r, -self.auto_goal_max_abs_x, self.auto_goal_max_abs_x))
        gy = float(np.clip(self.environment.robot_y, -self.auto_goal_max_abs_y, self.auto_goal_max_abs_y))
        self._activate_goal(gx, gy, now_sec, source="auto fallback")

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
