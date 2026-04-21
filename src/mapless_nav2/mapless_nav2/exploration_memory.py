#!/usr/bin/env python3
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from .exploration_types import Pose2D

GridCell = Tuple[int, int]


@dataclass
class PoseSample:
    x: float
    y: float
    yaw: float
    stamp: float


class ExplorationMemory:
    def __init__(self, params: Dict[str, float]) -> None:
        self.visited_decay_tau = max(1.0, float(params.get("visited_decay_tau", 120.0)))
        self.visited_increment = max(0.1, float(params.get("visited_increment", 1.0)))
        self.visited_prune_threshold = max(0.001, float(params.get("visited_prune_threshold", 0.03)))

        self.frontier_cooldown_sec = max(0.0, float(params.get("frontier_cooldown_sec", 10.0)))
        self.frontier_blacklist_base_sec = max(1.0, float(params.get("frontier_blacklist_base_sec", 20.0)))
        self.frontier_blacklist_max_sec = max(self.frontier_blacklist_base_sec, float(params.get("frontier_blacklist_max_sec", 240.0)))
        self.frontier_fail_blacklist_threshold = max(1, int(params.get("frontier_fail_blacklist_threshold", 2)))

        self.commitment_horizon_sec = max(0.0, float(params.get("commitment_horizon_sec", 10.0)))

        self.stuck_window_sec = max(1.0, float(params.get("stuck_window_sec", 12.0)))
        self.stuck_radius_m = max(0.01, float(params.get("stuck_radius_m", 0.22)))

        self.oscillation_window_sec = max(1.0, float(params.get("oscillation_window_sec", 10.0)))
        self.oscillation_cell_size_m = max(0.05, float(params.get("oscillation_cell_size_m", 0.20)))
        self.oscillation_toggle_threshold = max(2, int(params.get("oscillation_toggle_threshold", 4)))

        self.stagnation_window_sec = max(2.0, float(params.get("stagnation_window_sec", 20.0)))
        self.stagnation_min_known_delta = max(1, int(params.get("stagnation_min_known_delta", 40)))
        self.stagnation_min_travel_m = max(0.05, float(params.get("stagnation_min_travel_m", 0.60)))

        self.visited_heat: Dict[GridCell, float] = {}
        self.last_decay_stamp: Optional[float] = None

        self.frontier_fail_count: Dict[str, int] = {}
        self.frontier_cooldown_until: Dict[str, float] = {}
        self.frontier_blacklist_until: Dict[str, float] = {}

        self.committed_frontier_id: Optional[str] = None
        self.commitment_until: float = 0.0

        self.pose_history: Deque[PoseSample] = deque()
        self.last_motion_pose: Optional[Tuple[float, float]] = None
        self.last_motion_stamp: float = 0.0
        self.best_known_cells = 0
        self.last_information_gain_stamp: float = 0.0

    def _decay_visited(self, now_sec: float) -> None:
        if self.last_decay_stamp is None:
            self.last_decay_stamp = now_sec
            return
        dt = max(0.0, now_sec - self.last_decay_stamp)
        if dt <= 1e-3:
            return

        decay = math.exp(-dt / self.visited_decay_tau)
        to_delete = []
        for key in self.visited_heat:
            self.visited_heat[key] *= decay
            if self.visited_heat[key] < self.visited_prune_threshold:
                to_delete.append(key)
        for key in to_delete:
            del self.visited_heat[key]

        self.last_decay_stamp = now_sec

    def update_pose(self, pose: Pose2D, now_sec: float, known_cell_count: int) -> None:
        self._decay_visited(now_sec)

        self.pose_history.append(PoseSample(pose.x, pose.y, pose.yaw, now_sec))
        cutoff = now_sec - max(self.stuck_window_sec, self.oscillation_window_sec) - 1.0
        while self.pose_history and self.pose_history[0].stamp < cutoff:
            self.pose_history.popleft()

        if known_cell_count >= self.best_known_cells + self.stagnation_min_known_delta:
            self.best_known_cells = known_cell_count
            self.last_information_gain_stamp = now_sec

        if self.last_motion_pose is None:
            self.last_motion_pose = (pose.x, pose.y)
            self.last_motion_stamp = now_sec
        else:
            moved = math.hypot(pose.x - self.last_motion_pose[0], pose.y - self.last_motion_pose[1])
            if moved >= self.stagnation_min_travel_m:
                self.last_motion_pose = (pose.x, pose.y)
                self.last_motion_stamp = now_sec

    def mark_cell_visited(self, cell: GridCell, now_sec: float, amount: float = 1.0) -> None:
        self._decay_visited(now_sec)
        self.visited_heat[cell] = self.visited_heat.get(cell, 0.0) + amount * self.visited_increment

    def mark_path_visited(self, path_cells, now_sec: float) -> None:
        if not path_cells:
            return
        stride = max(1, len(path_cells) // 40)
        for cell in path_cells[::stride]:
            self.mark_cell_visited(cell, now_sec, amount=0.5)

    def revisit_penalty(self, cell: GridCell, now_sec: float) -> float:
        self._decay_visited(now_sec)
        return self.visited_heat.get(cell, 0.0)

    def frontier_available(self, frontier_id: str, now_sec: float) -> bool:
        if self.frontier_blacklist_until.get(frontier_id, 0.0) > now_sec:
            return False
        if self.frontier_cooldown_until.get(frontier_id, 0.0) > now_sec:
            return False
        return True

    def frontier_failed_penalty(self, frontier_id: str, now_sec: float) -> float:
        fail_count = float(self.frontier_fail_count.get(frontier_id, 0))
        blacklisted = 1.0 if self.frontier_blacklist_until.get(frontier_id, 0.0) > now_sec else 0.0
        cooling = 1.0 if self.frontier_cooldown_until.get(frontier_id, 0.0) > now_sec else 0.0
        return fail_count + 2.0 * blacklisted + 0.5 * cooling

    def register_frontier_selected(self, frontier_id: str, now_sec: float) -> None:
        self.frontier_cooldown_until[frontier_id] = max(
            self.frontier_cooldown_until.get(frontier_id, 0.0),
            now_sec + self.frontier_cooldown_sec,
        )
        self.committed_frontier_id = frontier_id
        self.commitment_until = now_sec + self.commitment_horizon_sec

    def register_frontier_success(self, frontier_id: str) -> None:
        old = self.frontier_fail_count.get(frontier_id, 0)
        self.frontier_fail_count[frontier_id] = max(0, old - 1)

    def register_frontier_failure(self, frontier_id: str, now_sec: float) -> None:
        count = self.frontier_fail_count.get(frontier_id, 0) + 1
        self.frontier_fail_count[frontier_id] = count

        cooldown = self.frontier_cooldown_sec * min(4.0, 1.0 + 0.5 * count)
        self.frontier_cooldown_until[frontier_id] = now_sec + cooldown

        if count >= self.frontier_fail_blacklist_threshold:
            power = max(0, count - self.frontier_fail_blacklist_threshold)
            blacklist = min(self.frontier_blacklist_max_sec, self.frontier_blacklist_base_sec * (2.0 ** power))
            self.frontier_blacklist_until[frontier_id] = now_sec + blacklist

        if self.committed_frontier_id == frontier_id:
            self.committed_frontier_id = None
            self.commitment_until = 0.0

    def commitment_bonus(self, frontier_id: str, now_sec: float) -> float:
        if self.committed_frontier_id != frontier_id:
            return 0.0
        if now_sec > self.commitment_until:
            return 0.0
        return max(0.0, (self.commitment_until - now_sec) / max(self.commitment_horizon_sec, 1e-6))

    def commitment_active(self, frontier_id: str, now_sec: float) -> bool:
        return self.committed_frontier_id == frontier_id and now_sec <= self.commitment_until

    def clear_commitment(self) -> None:
        self.committed_frontier_id = None
        self.commitment_until = 0.0

    def is_stuck(self, now_sec: float) -> bool:
        if len(self.pose_history) < 6:
            return False
        cutoff = now_sec - self.stuck_window_sec
        samples = [p for p in self.pose_history if p.stamp >= cutoff]
        if len(samples) < 6:
            return False

        cx = sum(s.x for s in samples) / len(samples)
        cy = sum(s.y for s in samples) / len(samples)
        max_r = max(math.hypot(s.x - cx, s.y - cy) for s in samples)
        return max_r < self.stuck_radius_m

    def is_oscillating(self, now_sec: float) -> bool:
        if len(self.pose_history) < 8:
            return False
        cutoff = now_sec - self.oscillation_window_sec
        samples = [p for p in self.pose_history if p.stamp >= cutoff]
        if len(samples) < 8:
            return False

        seq = []
        inv = 1.0 / self.oscillation_cell_size_m
        for s in samples:
            seq.append((int(round(s.x * inv)), int(round(s.y * inv))))

        collapsed = [seq[0]]
        for cell in seq[1:]:
            if cell != collapsed[-1]:
                collapsed.append(cell)
        if len(collapsed) < 4:
            return False

        toggles = 0
        for i in range(2, len(collapsed)):
            if collapsed[i] == collapsed[i - 2] and collapsed[i] != collapsed[i - 1]:
                toggles += 1
        return toggles >= self.oscillation_toggle_threshold

    def is_stagnating(self, now_sec: float) -> bool:
        if self.last_information_gain_stamp == 0.0:
            self.last_information_gain_stamp = now_sec
        if self.last_motion_stamp == 0.0:
            self.last_motion_stamp = now_sec
        no_info = (now_sec - self.last_information_gain_stamp) > self.stagnation_window_sec
        no_motion = (now_sec - self.last_motion_stamp) > self.stagnation_window_sec
        return no_info and no_motion
