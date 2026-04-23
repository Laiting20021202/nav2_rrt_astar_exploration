"""Main ROS 2 node for TD3-based goal-seeking navigation."""

from __future__ import annotations

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
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.goal_marker_frame = str(self.get_parameter("goal_marker_frame").value)
        self.publish_goal_marker = bool(self.get_parameter("publish_goal_marker").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reset_service_name = str(self.get_parameter("reset_service_name").value)
        self._rng = np.random.default_rng(self.random_seed)

        self.use_reference_policy = self.inference_mode and self.policy_source == "reference_actor"
        lidar_samples = self.reference_state_scan_samples if self.use_reference_policy else 24

        self.environment = GoalSeekerEnvironment(
            lidar_samples=lidar_samples,
            lidar_max_range=float(self.get_parameter("lidar_max_range").value),
            max_goal_distance=float(self.get_parameter("max_goal_distance").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
            collision_distance=float(self.get_parameter("collision_distance").value),
            stuck_window_sec=10.0,
            stuck_cell_size=float(self.get_parameter("stuck_cell_size").value),
            stuck_overlap_threshold=float(self.get_parameter("stuck_overlap_threshold").value),
            spin_filter_angular_threshold=0.5,
            spin_filter_min_range=0.15,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")
        self.agent: Optional[TD3Agent] = None
        self.reference_policy: Optional[ReferenceActorPolicy] = None
        self.reference_prev_action = np.zeros(2, dtype=np.float32)

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
            )
            self.agent = TD3Agent(
                state_dim=self.environment.state_dim,
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
        self.declare_parameter("linear_speed_max", 0.26)
        self.declare_parameter("reference_linear_speed_max", 0.22)
        self.declare_parameter("angular_speed_max", 1.0)
        self.declare_parameter("reference_angular_speed_max", 2.0)
        self.declare_parameter("inference_reset_on_stuck", False)

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
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("goal_marker_topic", "/goal_marker")
        self.declare_parameter("goal_marker_frame", "odom")
        self.declare_parameter("publish_goal_marker", True)

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
        if self.inference_mode and not self.inference_reset_on_stuck and result.reason == "stuck":
            # Keep trying in inference instead of stopping too early.
            result.done = False
            result.reason = "running"

        # The result at tick t is treated as outcome of action from tick t-1.
        if self.last_state is not None and self.last_action is not None:
            self._consume_transition(result)

        timed_out = self.episode_steps >= self.max_episode_steps
        if result.done or timed_out:
            reason = result.reason if result.done else "timeout"
            self._finish_episode(reason, result)
            return

        action_norm = self._select_action(result.state)
        self._publish_action(action_norm)
        self.last_state = result.state
        self.last_action = action_norm

    def _consume_transition(self, result: StepResult) -> None:
        """Store transition and optionally train the TD3 model."""
        if not self.inference_mode:
            if self.agent is None:
                return
            self.agent.store_transition(
                state=self.last_state,
                action=self.last_action,
                reward=result.reward,
                next_state=result.state,
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

        self.get_logger().info(
            "Episode summary | "
            f"steps={self.episode_steps} reward={self.episode_reward:.2f} reason={reason} "
            f"goal_dist={result.goal_distance:.3f} heading={result.heading_angle:.3f} "
            f"min_scan={result.min_obstacle_distance:.3f} overlap={self.environment.last_overlap_ratio:.2f} "
            f"critic_loss={critic_loss} actor_loss={actor_loss}"
        )

        self.last_state = None
        self.last_action = None
        self.reference_prev_action = np.zeros(2, dtype=np.float32)
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
            policy_input = np.concatenate([state.astype(np.float32), self.reference_prev_action], axis=0)
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

    def _publish_action(self, action_norm: np.ndarray) -> None:
        """Map normalized action to robot velocity command and publish."""
        action = np.clip(action_norm, -1.0, 1.0)
        linear_cap = self.reference_linear_speed_max if self.use_reference_policy else self.linear_speed_max
        angular_cap = self.reference_angular_speed_max if self.use_reference_policy else self.angular_speed_max
        linear_x = float((action[0] + 1.0) * 0.5 * linear_cap)
        angular_z = float(np.clip(action[1] * angular_cap, -angular_cap, angular_cap))

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)
        self.reference_prev_action = action.astype(np.float32)

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
        self.reference_prev_action = np.zeros(2, dtype=np.float32)
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
        max_r = max(min_r, self.auto_goal_max_radius)
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
