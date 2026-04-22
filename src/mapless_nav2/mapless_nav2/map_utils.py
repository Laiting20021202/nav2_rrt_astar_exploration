#!/usr/bin/env python3
import heapq
import math
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nav_msgs.msg import OccupancyGrid

GridCell = Tuple[int, int]


def grid_index(width: int, cell: GridCell) -> int:
    return cell[1] * width + cell[0]


def in_bounds(width: int, height: int, cell: GridCell) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def world_to_grid(msg: OccupancyGrid, x: float, y: float) -> Optional[GridCell]:
    info = msg.info
    if info.resolution <= 0.0:
        return None
    gx = int((x - info.origin.position.x) / info.resolution)
    gy = int((y - info.origin.position.y) / info.resolution)
    if not in_bounds(int(info.width), int(info.height), (gx, gy)):
        return None
    return (gx, gy)


def grid_to_world(msg: OccupancyGrid, cell: GridCell) -> Tuple[float, float]:
    info = msg.info
    x = info.origin.position.x + (cell[0] + 0.5) * info.resolution
    y = info.origin.position.y + (cell[1] + 0.5) * info.resolution
    return (float(x), float(y))


def neighbors4(cell: GridCell) -> Iterable[GridCell]:
    x, y = cell
    yield (x - 1, y)
    yield (x + 1, y)
    yield (x, y - 1)
    yield (x, y + 1)


def neighbors8(cell: GridCell) -> Iterable[GridCell]:
    x, y = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def is_unknown(value: int) -> bool:
    return value < 0


def is_occupied(value: int, occupied_threshold: int) -> bool:
    return value >= occupied_threshold


def is_free(value: int, free_threshold: int) -> bool:
    return value >= 0 and value <= free_threshold


def known_cell_count(msg: OccupancyGrid) -> int:
    return sum(1 for v in msg.data if v >= 0)


def unknown_cell_count(msg: OccupancyGrid) -> int:
    return sum(1 for v in msg.data if v < 0)


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def line_cells(a: GridCell, b: GridCell) -> List[GridCell]:
    x0, y0 = a
    x1, y1 = b
    cells: List[GridCell] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


def line_collision_free(
    width: int,
    height: int,
    inflated_mask: Sequence[bool],
    data: Sequence[int],
    start: GridCell,
    end: GridCell,
    free_threshold: int,
    allow_unknown: bool,
) -> bool:
    for cell in line_cells(start, end):
        if not in_bounds(width, height, cell):
            return False
        idx = grid_index(width, cell)
        if inflated_mask[idx]:
            return False
        value = int(data[idx])
        if is_unknown(value):
            if not allow_unknown:
                return False
            continue
        if not is_free(value, free_threshold):
            return False
    return True


def build_inflated_obstacle_mask(
    msg: OccupancyGrid,
    occupied_threshold: int,
    inflation_radius_m: float,
    ignore_occupied_cells: Optional[Set[GridCell]] = None,
) -> List[bool]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    data = msg.data
    mask = [False] * (width * height)
    ignored = ignore_occupied_cells or set()

    inflation_cells = int(math.ceil(inflation_radius_m / max(msg.info.resolution, 1e-6)))
    if inflation_cells <= 0:
        for y in range(height):
            for x in range(width):
                idx = grid_index(width, (x, y))
                if (x, y) in ignored:
                    mask[idx] = False
                    continue
                mask[idx] = is_occupied(int(data[idx]), occupied_threshold)
        return mask

    occupied_cells: List[GridCell] = []
    for y in range(height):
        for x in range(width):
            idx = grid_index(width, (x, y))
            if (x, y) in ignored:
                continue
            if is_occupied(int(data[idx]), occupied_threshold):
                occupied_cells.append((x, y))

    for ox, oy in occupied_cells:
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                nx = ox + dx
                ny = oy + dy
                if not in_bounds(width, height, (nx, ny)):
                    continue
                if math.hypot(dx, dy) > inflation_cells:
                    continue
                nidx = grid_index(width, (nx, ny))
                mask[nidx] = True

    return mask


def nearest_free_cell(
    msg: OccupancyGrid,
    start: GridCell,
    inflated_mask: Sequence[bool],
    free_threshold: int,
    max_radius: int,
) -> Optional[GridCell]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    if not in_bounds(width, height, start):
        return None

    def traversable(cell: GridCell) -> bool:
        idx = grid_index(width, cell)
        return (not inflated_mask[idx]) and is_free(int(msg.data[idx]), free_threshold)

    if traversable(start):
        return start

    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cell = (start[0] + dx, start[1] + dy)
                if not in_bounds(width, height, cell):
                    continue
                if traversable(cell):
                    return cell
    return None


def astar_path(
    msg: OccupancyGrid,
    inflated_mask: Sequence[bool],
    start: GridCell,
    goal: GridCell,
    free_threshold: int,
    allow_unknown: bool,
    revisit_heat: Optional[Dict[GridCell, float]] = None,
    revisit_cost_scale: float = 0.0,
    soft_obstacle_cells: Optional[Dict[GridCell, float]] = None,
    soft_obstacle_cost_scale: float = 0.0,
    unknown_cost_scale: float = 0.0,
) -> Tuple[List[GridCell], float]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    data = msg.data

    if not in_bounds(width, height, start) or not in_bounds(width, height, goal):
        return ([], float("inf"))

    def traversable(cell: GridCell) -> bool:
        soft_weight = None
        if soft_obstacle_cells is not None:
            soft_weight = soft_obstacle_cells.get(cell)

        idx = grid_index(width, cell)
        if inflated_mask[idx] and soft_weight is None:
            return False
        value = int(data[idx])
        if is_unknown(value):
            return allow_unknown
        if is_free(value, free_threshold):
            return True
        # Sensor transient cells can be traversed with extra cost.
        return soft_weight is not None

    if not traversable(start) or not traversable(goal):
        return ([], float("inf"))

    open_heap: List[Tuple[float, float, GridCell]] = []
    g: Dict[GridCell, float] = {start: 0.0}
    parent: Dict[GridCell, GridCell] = {}
    closed = set()

    def h(cell: GridCell) -> float:
        return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    heapq.heappush(open_heap, (h(start), 0.0, start))

    while open_heap:
        _, g_curr, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        closed.add(cell)

        if cell == goal:
            path = [cell]
            while path[-1] != start:
                prev = parent.get(path[-1])
                if prev is None:
                    break
                path.append(prev)
            path.reverse()
            return (path, g_curr)

        for nb in neighbors8(cell):
            if not in_bounds(width, height, nb):
                continue
            if nb in closed:
                continue
            if not traversable(nb):
                continue

            dx = nb[0] - cell[0]
            dy = nb[1] - cell[1]
            step = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0

            revisit_term = 0.0
            if revisit_heat is not None and revisit_cost_scale > 0.0:
                revisit_term = revisit_cost_scale * revisit_heat.get(nb, 0.0)

            value = int(data[grid_index(width, nb)])
            occupancy_term = 0.0
            if value >= 0:
                occupancy_term = 0.2 * min(1.0, value / 100.0)
            elif allow_unknown and unknown_cost_scale > 0.0:
                occupancy_term = unknown_cost_scale

            soft_term = 0.0
            if soft_obstacle_cells is not None and soft_obstacle_cost_scale > 0.0:
                soft_weight = soft_obstacle_cells.get(nb)
                if soft_weight is not None:
                    soft_term = soft_obstacle_cost_scale * max(0.0, min(1.0, soft_weight))

            tentative = g_curr + step + occupancy_term + revisit_term + soft_term
            if tentative >= g.get(nb, float("inf")):
                continue

            g[nb] = tentative
            parent[nb] = cell
            heapq.heappush(open_heap, (tentative + h(nb), tentative, nb))

    return ([], float("inf"))


def path_length_m(path_cells: Sequence[GridCell], resolution: float) -> float:
    if len(path_cells) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_cells)):
        dx = path_cells[i][0] - path_cells[i - 1][0]
        dy = path_cells[i][1] - path_cells[i - 1][1]
        total += math.hypot(dx, dy)
    return total * resolution


def unknown_count_in_radius(
    msg: OccupancyGrid,
    center: GridCell,
    radius_cells: int,
) -> int:
    width = int(msg.info.width)
    height = int(msg.info.height)
    if radius_cells <= 0:
        return 0
    count = 0
    cx, cy = center
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            cell = (cx + dx, cy + dy)
            if not in_bounds(width, height, cell):
                continue
            if is_unknown(int(msg.data[grid_index(width, cell)])):
                count += 1
    return count


def connected_component(
    seeds: Iterable[GridCell],
    width: int,
    height: int,
    connect8: bool = True,
) -> List[List[GridCell]]:
    seed_set = set(seeds)
    visited = set()
    groups: List[List[GridCell]] = []
    nbs = neighbors8 if connect8 else neighbors4

    for seed in seed_set:
        if seed in visited:
            continue
        q = deque([seed])
        visited.add(seed)
        group: List[GridCell] = []
        while q:
            cur = q.popleft()
            group.append(cur)
            for nb in nbs(cur):
                if not in_bounds(width, height, nb):
                    continue
                if nb in visited or nb not in seed_set:
                    continue
                visited.add(nb)
                q.append(nb)
        groups.append(group)

    return groups
