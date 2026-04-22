#!/usr/bin/env python3
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from nav_msgs.msg import OccupancyGrid

from .exploration_types import FrontierCandidate, FrontierCluster
from .map_utils import (
    GridCell,
    connected_component,
    euclidean,
    grid_index,
    grid_to_world,
    in_bounds,
    is_free,
    is_unknown,
    line_collision_free,
    neighbors4,
    nearest_free_cell,
    unknown_count_in_radius,
    world_to_grid,
)


@dataclass
class RRTNode:
    world: Tuple[float, float]
    cell: GridCell
    parent: int


class FrontierExtractor:
    def __init__(self, params: Dict[str, float]) -> None:
        self.free_threshold = int(params.get("free_threshold", 15))
        self.occupied_threshold = int(params.get("occupied_threshold", 65))

        self.min_cluster_size = max(3, int(params.get("frontier_min_cluster_size", 10)))
        self.info_gain_radius_m = max(0.1, float(params.get("frontier_info_gain_radius_m", 1.0)))
        self.frontier_filter_min_distance = max(0.0, float(params.get("frontier_filter_min_distance", 0.5)))
        self.frontier_filter_max_distance = max(0.5, float(params.get("frontier_filter_max_distance", 15.0)))

        self.rrt_iterations = max(10, int(params.get("rrt_iterations", 400)))
        self.rrt_step_m = max(0.05, float(params.get("rrt_step_m", 0.25)))
        self.rrt_goal_bias = min(1.0, max(0.0, float(params.get("rrt_goal_bias", 0.35))))
        self.rrt_frontier_proximity_m = max(0.05, float(params.get("rrt_frontier_proximity_m", 0.50)))
        self.rrt_goal_snap_radius_cells = max(1, int(params.get("rrt_goal_snap_radius_cells", 8)))

        self.candidate_min_separation_m = max(0.05, float(params.get("candidate_min_separation_m", 0.45)))
        self.candidate_max_per_frontier = max(1, int(params.get("candidate_max_per_frontier", 3)))
        self.robot_candidate_min_distance_m = max(0.0, float(params.get("robot_candidate_min_distance_m", 0.6)))
        self.candidate_inward_offset_m = max(0.0, float(params.get("candidate_inward_offset_m", 0.18)))
        self.candidate_snap_radius_cells = max(1, int(params.get("candidate_snap_radius_cells", 6)))
        self.candidate_clearance_max_m = max(0.1, float(params.get("candidate_clearance_max_m", 1.2)))

    def extract_frontier_clusters(
        self,
        msg: OccupancyGrid,
        inflated_mask: Sequence[bool],
    ) -> List[FrontierCluster]:
        width = int(msg.info.width)
        height = int(msg.info.height)
        data = msg.data

        frontier_cells = set()
        for y in range(height):
            for x in range(width):
                cell = (x, y)
                idx = grid_index(width, cell)
                if inflated_mask[idx]:
                    continue
                value = int(data[idx])
                if not is_free(value, self.free_threshold):
                    continue

                is_frontier = False
                for nb in neighbors4(cell):
                    if not in_bounds(width, height, nb):
                        continue
                    nb_idx = grid_index(width, nb)
                    if is_unknown(int(data[nb_idx])):
                        is_frontier = True
                        break
                if is_frontier:
                    frontier_cells.add(cell)

        groups = connected_component(frontier_cells, width, height, connect8=True)
        clusters: List[FrontierCluster] = []
        info_radius_cells = int(max(1, round(self.info_gain_radius_m / max(msg.info.resolution, 1e-6))))

        for cells in groups:
            if len(cells) < self.min_cluster_size:
                continue

            cx = sum(c[0] for c in cells) / len(cells)
            cy = sum(c[1] for c in cells) / len(cells)
            centroid_cell = (int(round(cx)), int(round(cy)))
            if not in_bounds(width, height, centroid_cell):
                continue

            boundary = []
            cell_set = set(cells)
            for c in cells:
                for nb in neighbors4(c):
                    if nb not in cell_set:
                        boundary.append(c)
                        break

            centroid_world = grid_to_world(msg, centroid_cell)
            info_gain = float(unknown_count_in_radius(msg, centroid_cell, info_radius_cells))
            cluster_id = f"f_{centroid_cell[0]}_{centroid_cell[1]}"
            clusters.append(
                FrontierCluster(
                    cluster_id=cluster_id,
                    cells=cells,
                    boundary_cells=boundary if boundary else cells,
                    centroid_cell=centroid_cell,
                    centroid_world=centroid_world,
                    information_gain=info_gain,
                )
            )

        return clusters

    def generate_rrt_candidates(
        self,
        msg: OccupancyGrid,
        robot_cell: GridCell,
        robot_world: Tuple[float, float],
        clusters: List[FrontierCluster],
        inflated_mask: Sequence[bool],
        rng: random.Random,
    ) -> Tuple[List[FrontierCandidate], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        if not clusters:
            return ([], [])

        width = int(msg.info.width)
        height = int(msg.info.height)
        res = float(msg.info.resolution)
        data = msg.data

        start_cell = nearest_free_cell(msg, robot_cell, inflated_mask, self.free_threshold, max_radius=4)
        if start_cell is None:
            return ([], [])
        start_world = grid_to_world(msg, start_cell)

        cluster_centers_world = [c.centroid_world for c in clusters]
        cluster_by_id = {c.cluster_id: c for c in clusters}

        nodes: List[RRTNode] = [RRTNode(world=start_world, cell=start_cell, parent=-1)]
        tree_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        candidate_count: Dict[str, int] = {c.cluster_id: 0 for c in clusters}
        candidates: List[FrontierCandidate] = []

        step_cells = max(1, int(round(self.rrt_step_m / max(res, 1e-6))))
        proximity_cells = max(1, int(round(self.rrt_frontier_proximity_m / max(res, 1e-6))))

        def nearest_node_index(sample_world: Tuple[float, float]) -> int:
            best_idx = 0
            best_d = float("inf")
            for i, n in enumerate(nodes):
                d = euclidean(n.world, sample_world)
                if d < best_d:
                    best_d = d
                    best_idx = i
            return best_idx

        def steered_cell(src: GridCell, dst_world: Tuple[float, float]) -> GridCell:
            dst_cell = world_to_grid(msg, dst_world[0], dst_world[1])
            if dst_cell is None:
                # clip to map bounds by world clamp
                min_x = msg.info.origin.position.x
                min_y = msg.info.origin.position.y
                max_x = min_x + width * res
                max_y = min_y + height * res
                x = min(max(dst_world[0], min_x + 0.5 * res), max_x - 0.5 * res)
                y = min(max(dst_world[1], min_y + 0.5 * res), max_y - 0.5 * res)
                dst_cell = world_to_grid(msg, x, y)
                if dst_cell is None:
                    return src

            dx = dst_cell[0] - src[0]
            dy = dst_cell[1] - src[1]
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                return src
            scale = min(1.0, step_cells / norm)
            nx = int(round(src[0] + dx * scale))
            ny = int(round(src[1] + dy * scale))
            return (nx, ny)

        def traversable(cell: GridCell) -> bool:
            if not in_bounds(width, height, cell):
                return False
            idx = grid_index(width, cell)
            if inflated_mask[idx]:
                return False
            return is_free(int(data[idx]), self.free_threshold)

        for _ in range(self.rrt_iterations):
            if rng.random() < self.rrt_goal_bias:
                sample_world = rng.choice(cluster_centers_world)
            else:
                min_x = msg.info.origin.position.x
                min_y = msg.info.origin.position.y
                max_x = min_x + width * res
                max_y = min_y + height * res
                sample_world = (
                    rng.uniform(min_x, max_x),
                    rng.uniform(min_y, max_y),
                )

            near_idx = nearest_node_index(sample_world)
            near = nodes[near_idx]
            new_cell = steered_cell(near.cell, sample_world)
            if new_cell == near.cell:
                continue
            if not traversable(new_cell):
                continue
            if not line_collision_free(
                width,
                height,
                inflated_mask,
                data,
                near.cell,
                new_cell,
                self.free_threshold,
                allow_unknown=False,
            ):
                continue

            new_world = grid_to_world(msg, new_cell)
            nodes.append(RRTNode(world=new_world, cell=new_cell, parent=near_idx))
            tree_edges.append((near.world, new_world))

            # Match node to nearest frontier centroid.
            best_frontier: Optional[FrontierCluster] = None
            best_dist = float("inf")
            for cluster in clusters:
                dist_cells = math.hypot(new_cell[0] - cluster.centroid_cell[0], new_cell[1] - cluster.centroid_cell[1])
                if dist_cells < best_dist:
                    best_dist = dist_cells
                    best_frontier = cluster

            if best_frontier is None:
                continue
            if best_dist > proximity_cells:
                continue
            if candidate_count[best_frontier.cluster_id] >= self.candidate_max_per_frontier:
                continue

            if euclidean(new_world, robot_world) < self.robot_candidate_min_distance_m:
                continue

            staged = self.stage_candidate(msg, new_cell, robot_world, inflated_mask)
            if staged is None:
                continue
            staged_cell, staged_world = staged

            candidate = FrontierCandidate(
                candidate_id=f"c_{best_frontier.cluster_id}_{candidate_count[best_frontier.cluster_id]}",
                frontier_id=best_frontier.cluster_id,
                world=staged_world,
                cell=staged_cell,
                information_gain=best_frontier.information_gain,
                clearance_bonus=self.estimate_clearance(msg, staged_cell),
            )
            candidates.append(candidate)
            candidate_count[best_frontier.cluster_id] += 1

        # Fallback: use frontier centroids if RRT did not propose enough points.
        for cluster in clusters:
            if candidate_count[cluster.cluster_id] >= 1:
                continue
            centroid = nearest_free_cell(
                msg,
                cluster.centroid_cell,
                inflated_mask,
                self.free_threshold,
                max_radius=self.rrt_goal_snap_radius_cells,
            )
            if centroid is None:
                continue
            staged = self.stage_candidate(msg, centroid, robot_world, inflated_mask)
            if staged is None:
                continue
            centroid, centroid_world = staged
            if euclidean(centroid_world, robot_world) < self.robot_candidate_min_distance_m:
                continue
            candidates.append(
                FrontierCandidate(
                    candidate_id=f"c_{cluster.cluster_id}_fallback",
                    frontier_id=cluster.cluster_id,
                    world=centroid_world,
                    cell=centroid,
                    information_gain=cluster.information_gain,
                    clearance_bonus=self.estimate_clearance(msg, centroid),
                )
            )
            candidate_count[cluster.cluster_id] += 1

        candidates = self._deduplicate_candidates(candidates)

        # Keep only frontiers in distance window.
        filtered = []
        for c in candidates:
            d = euclidean(robot_world, c.world)
            if d < self.frontier_filter_min_distance or d > self.frontier_filter_max_distance:
                continue
            if c.frontier_id not in cluster_by_id:
                continue
            filtered.append(c)

        return (filtered, tree_edges)

    def _deduplicate_candidates(self, candidates: List[FrontierCandidate]) -> List[FrontierCandidate]:
        out: List[FrontierCandidate] = []
        for cand in sorted(candidates, key=lambda c: c.information_gain, reverse=True):
            keep = True
            for existing in out:
                if euclidean(cand.world, existing.world) < self.candidate_min_separation_m:
                    keep = False
                    break
            if keep:
                out.append(cand)
        return out

    def stage_candidate(
        self,
        msg: OccupancyGrid,
        cell: GridCell,
        reference_world: Tuple[float, float],
        inflated_mask: Sequence[bool],
    ) -> Optional[Tuple[GridCell, Tuple[float, float]]]:
        raw_world = grid_to_world(msg, cell)
        dx = reference_world[0] - raw_world[0]
        dy = reference_world[1] - raw_world[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-6 and self.candidate_inward_offset_m > 0.0:
            scale = min(self.candidate_inward_offset_m, norm) / norm
            stage_world = (raw_world[0] + dx * scale, raw_world[1] + dy * scale)
            stage_cell = world_to_grid(msg, stage_world[0], stage_world[1]) or cell
        else:
            stage_cell = cell

        snapped = nearest_free_cell(
            msg,
            stage_cell,
            inflated_mask,
            self.free_threshold,
            max_radius=self.candidate_snap_radius_cells,
        )
        if snapped is None:
            return None
        return (snapped, grid_to_world(msg, snapped))

    def estimate_clearance(self, msg: OccupancyGrid, cell: GridCell) -> float:
        width = int(msg.info.width)
        height = int(msg.info.height)
        max_radius_cells = max(1, int(round(self.candidate_clearance_max_m / max(msg.info.resolution, 1e-6))))
        if not in_bounds(width, height, cell):
            return 0.0

        for radius in range(1, max_radius_cells + 1):
            found = False
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nb = (cell[0] + dx, cell[1] + dy)
                    if not in_bounds(width, height, nb):
                        found = True
                        continue
                    idx = grid_index(width, nb)
                    if int(msg.data[idx]) >= self.occupied_threshold:
                        found = True
                        break
                if found:
                    break
            if found:
                return radius * float(msg.info.resolution)

        return self.candidate_clearance_max_m
