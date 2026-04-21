#!/usr/bin/env python3
import math
from typing import Dict, List, Optional

from .exploration_memory import ExplorationMemory
from .exploration_types import FrontierCandidate, Pose2D
from .map_utils import angle_wrap


class FrontierScorer:
    def __init__(self, params: Dict[str, float]) -> None:
        self.w_info = float(params.get("w_info", 1.4))
        self.w_cost = float(params.get("w_cost", 1.1))
        self.w_visit = float(params.get("w_visit", 0.7))
        self.w_fail = float(params.get("w_fail", 1.0))
        self.w_turn = float(params.get("w_turn", 0.35))
        self.w_goal = float(params.get("w_goal", 0.45))
        self.w_commit = float(params.get("w_commit", 0.4))

        self.max_revisit_penalty = max(0.1, float(params.get("max_revisit_penalty", 6.0)))
        self.max_fail_penalty = max(0.1, float(params.get("max_fail_penalty", 8.0)))

        # Goal gravitation: near final goal -> stronger goal attraction, weaker exploration curiosity.
        self.goal_gravity_range_m = max(0.1, float(params.get("goal_gravity_range_m", 6.0)))
        self.goal_gravity_exp_gain = max(0.0, float(params.get("goal_gravity_exp_gain", 1.2)))
        self.goal_gravity_max_multiplier = max(
            1.0,
            float(params.get("goal_gravity_max_multiplier", 3.5)),
        )
        self.info_near_goal_min_scale = max(
            0.0,
            min(1.0, float(params.get("info_near_goal_min_scale", 0.35))),
        )

    def score_candidates(
        self,
        candidates: List[FrontierCandidate],
        robot_pose: Pose2D,
        final_goal_xy: Optional[tuple],
        memory: ExplorationMemory,
        now_sec: float,
    ) -> List[FrontierCandidate]:
        if not candidates:
            return []

        max_info = max(1e-3, max(c.information_gain for c in candidates))
        max_cost = max(1e-3, max(c.path_cost for c in candidates if math.isfinite(c.path_cost)))
        w_info_eff, w_goal_eff = self._effective_goal_weights(robot_pose, final_goal_xy)

        for c in candidates:
            info_norm = c.information_gain / max_info
            cost_norm = min(1.0, c.path_cost / max_cost)

            visit_norm = min(1.0, c.revisit_penalty / self.max_revisit_penalty)
            fail_norm = min(1.0, c.failed_penalty / self.max_fail_penalty)
            turn_norm = max(0.0, min(1.0, c.heading_penalty))

            goal_bonus_norm = max(0.0, min(1.0, c.goal_alignment_bonus))
            commit_norm = max(0.0, min(1.0, c.commitment_bonus))

            c.score = (
                +w_info_eff * info_norm
                -self.w_cost * cost_norm
                -self.w_visit * visit_norm
                -self.w_fail * fail_norm
                -self.w_turn * turn_norm
                +w_goal_eff * goal_bonus_norm
                +self.w_commit * commit_norm
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    @staticmethod
    def heading_change_penalty(robot_pose: Pose2D, target_xy: tuple) -> float:
        target_heading = math.atan2(target_xy[1] - robot_pose.y, target_xy[0] - robot_pose.x)
        dtheta = abs(angle_wrap(target_heading - robot_pose.yaw))
        return dtheta / math.pi

    @staticmethod
    def goal_alignment_bonus(robot_pose: Pose2D, target_xy: tuple, final_goal_xy: Optional[tuple]) -> float:
        if final_goal_xy is None:
            return 0.0

        vx1 = target_xy[0] - robot_pose.x
        vy1 = target_xy[1] - robot_pose.y
        vx2 = final_goal_xy[0] - robot_pose.x
        vy2 = final_goal_xy[1] - robot_pose.y
        n1 = math.hypot(vx1, vy1)
        n2 = math.hypot(vx2, vy2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cosang = (vx1 * vx2 + vy1 * vy2) / max(1e-6, n1 * n2)
        # map [-1,1] -> [0,1]
        return 0.5 * (cosang + 1.0)

    def _effective_goal_weights(self, robot_pose: Pose2D, final_goal_xy: Optional[tuple]) -> tuple:
        if final_goal_xy is None:
            return (self.w_info, self.w_goal)

        dist_to_final_goal = math.hypot(final_goal_xy[0] - robot_pose.x, final_goal_xy[1] - robot_pose.y)
        proximity = max(0.0, min(1.0, 1.0 - (dist_to_final_goal / self.goal_gravity_range_m)))

        # Linear suppression of information gain weight when close to final goal.
        info_scale = 1.0 - (1.0 - self.info_near_goal_min_scale) * proximity
        w_info_eff = self.w_info * info_scale

        # Exponential amplification of goal alignment weight near final goal.
        goal_multiplier = math.exp(self.goal_gravity_exp_gain * proximity)
        goal_multiplier = min(self.goal_gravity_max_multiplier, goal_multiplier)
        w_goal_eff = self.w_goal * goal_multiplier
        return (w_info_eff, w_goal_eff)
