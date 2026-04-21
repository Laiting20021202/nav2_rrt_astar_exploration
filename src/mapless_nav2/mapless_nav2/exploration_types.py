#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class FrontierCluster:
    cluster_id: str
    cells: List[Tuple[int, int]]
    boundary_cells: List[Tuple[int, int]]
    centroid_cell: Tuple[int, int]
    centroid_world: Tuple[float, float]
    information_gain: float


@dataclass
class FrontierCandidate:
    candidate_id: str
    frontier_id: str
    world: Tuple[float, float]
    cell: Tuple[int, int]
    information_gain: float
    path_cost: float = float("inf")
    path_cells: List[Tuple[int, int]] = field(default_factory=list)
    revisit_penalty: float = 0.0
    failed_penalty: float = 0.0
    heading_penalty: float = 0.0
    goal_alignment_bonus: float = 0.0
    commitment_bonus: float = 0.0
    score: float = -float("inf")


@dataclass
class NavigationTarget:
    target_type: str
    target_id: str
    pose_world: Tuple[float, float, float]
    path_cells: List[Tuple[int, int]] = field(default_factory=list)
    frontier_id: Optional[str] = None
    candidate_id: Optional[str] = None
