"""Main ROS 2 node for TD3-based goal-seeking navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import torch

from .environment_node import GoalSeekerEnvironment, StepResult
from .td3_agent import TD3Agent, TD3Config


class GoalSeekerMainNode(Node):
    """Coordinates ROS I/O, environment stepping, and TD3 training/inference."""

    def __init__(self) -> None:
        super().__init__("goal_seeker_rl")
        self._declare_parameters()

        self.inference_mode = bool(self.get_parameter("inference_mode").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.resume_model_path = str(self.get_parameter("resume_model_path").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_episode_steps = int(self.get_parameter("max_episode_steps").value)
        self.reset_pause_sec = float(self.get_parameter("reset_pause_sec").value)
        self.checkpoint_dir = Path(str(self.get_parameter("checkpoint_dir").value))
        self.checkpoint_interval_steps = int(self.get_parameter("checkpoint_interval_steps").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reset_service_name = str(self.get_parameter("reset_service_name").value)

        self.environment = GoalSeekerEnvironment(
            lidar_samples=24,
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
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)

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
        self.declare_parameter("model_path", "")
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
        self.declare_parameter("max_goal_distance", 10.0)
        self.declare_parameter("goal_tolerance", 0.20)
        self.declare_parameter("collision_distance", 0.13)
        self.declare_parameter("stuck_cell_size", 0.10)
        self.declare_parameter("stuck_overlap_threshold", 0.70)

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

        self.environment.set_goal(gx, gy, now_sec)
        self.goal_active = True
        self.last_state = None
        self.last_action = None
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_index += 1

        self.get_logger().info(
            f"Goal #{self.episode_index} received from RViz: x={gx:.2f}, y={gy:.2f}. Navigation started."
        )

    def _control_loop(self) -> None:
        """Main control/training/inference loop."""
        now_sec = self._now_sec()

        if now_sec < self.pause_until_sec:
            self._publish_stop()
            return

        if not self.goal_active:
            return

        if not self.environment.ready_for_control():
            return

        result = self.environment.evaluate_step(now_sec)

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
        self.episode_steps = 0
        self.episode_reward = 0.0

        if self.inference_mode:
            self.goal_active = False
            return

        if reason in {"stuck", "collision", "timeout", "goal"}:
            self._reset_simulation()
            self.environment.register_reset(self._now_sec())
            self.pause_until_sec = self._now_sec() + self.reset_pause_sec

        self._maybe_save_checkpoint()

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Choose action according to mode and exploration phase."""
        if self.inference_mode:
            return self.agent.select_action(state, explore=False)

        if self.total_steps < self.warmup_steps:
            return self.agent.sample_random_action()

        return self.agent.select_action(state, explore=True)

    def _publish_action(self, action_norm: np.ndarray) -> None:
        """Map normalized action to TurtleBot3 action space and publish."""
        action = np.clip(action_norm, -1.0, 1.0)
        linear_x = float((action[0] + 1.0) * 0.5 * 0.26)  # [0, 0.26]
        angular_z = float(action[1])  # [-1, 1]

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def _publish_stop(self) -> None:
        """Publish zero velocity command."""
        self.cmd_pub.publish(Twist())

    def _reset_simulation(self) -> None:
        """Call Gazebo simulation reset service asynchronously."""
        if not self.reset_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Reset service unavailable: {self.reset_service_name}")
            return
        self.reset_client.call_async(Empty.Request())

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
    finally:
        if not node.inference_mode:
            node._maybe_save_checkpoint()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

