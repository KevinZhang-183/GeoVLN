"""Grid planning helpers shared by the executor and the waypoint policy."""

from __future__ import annotations

import heapq
import math

import numpy as np

from geonav._bootstrap import install

install()

from portal_map import _agent_axes, _dilate  # noqa: E402


def bearing_deg(pose, xz):
    """Signed degrees from the agent heading to a world ``(x, z)`` point. Right is positive."""
    agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
    delta = np.asarray(xz, dtype=np.float64) - agent
    return _bearing_vector(pose, delta)


def heading_error_deg(pose, heading):
    """Signed degrees from the agent heading to a world direction. Right is positive."""
    return _bearing_vector(pose, heading)


def _bearing_vector(pose, vector):
    forward, right = _agent_axes(pose)
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return 0.0
    vector = vector / norm
    return math.degrees(math.atan2(float(vector @ right), float(vector @ forward)))


def block_ahead(grid, pose, near_m=0.3, far_m=0.7, width_m=0.4):
    """Paint a patch of occupied cells in front of the agent. Used after a stuck step."""
    forward, right = _agent_axes(pose)
    origin = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
    samples = []
    distance = near_m
    while distance <= far_m + 1e-6:
        lateral = -width_m / 2.0
        while lateral <= width_m / 2.0 + 1e-6:
            samples.append(origin + forward * distance + right * lateral)
            lateral += grid.res
        distance += grid.res
    if not samples:
        return 0
    rows, cols, ok = grid.cells(np.stack(samples, axis=0))
    grid.logodds[rows[ok], cols[ok]] = grid.config.clamp_logodds
    return int(ok.sum())


def frontiers(grid, agent_xz, min_m=0.5, pull_m=0.5, limit=12):
    """Walkable cells touching unknown space, pulled back into known free space.

    Clusters whose longer side is shorter than ``min_m`` are dropped. Each kept
    cluster contributes one ``(x, z)`` point, nearest to the agent first.
    """
    if grid is None:
        return []
    occupied = grid.logodds > grid.config.occupied_logodds
    free = (grid.logodds < grid.config.free_logodds) & ~occupied
    unknown = ~free & ~occupied
    edge = _dilate(unknown, 1) & free & ~_dilate(occupied, 1)
    agent = np.asarray(agent_xz, dtype=np.float64)
    points = []
    for cells in _components(edge):
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        height = (max(rows) - min(rows) + 1) * grid.res
        width = (max(cols) - min(cols) + 1) * grid.res
        if max(height, width) < min_m:
            continue
        centroid = grid.origin + np.array([np.mean(cols) + 0.5, np.mean(rows) + 0.5]) * grid.res
        offset = agent - centroid
        norm = float(np.linalg.norm(offset))
        pulled = centroid if norm < 1e-6 else centroid + offset / norm * pull_m
        snapped = _snap_free(grid, free, pulled, max_m=max(pull_m, 0.8))
        if snapped is None:
            continue
        points.append(snapped)
    points.sort(key=lambda point: float(np.hypot(point[0] - agent[0], point[1] - agent[1])))
    return points[:limit]


def plan(grid, start_xz, goal_xz, inflate_m=0.15, unknown_cost=5.0, coarse_m=0.2, snap_m=1.0):
    """A* on a coarsened copy of the occupancy grid.

    Occupied cells are dilated by ``inflate_m``. Unknown cells are walkable at
    ``unknown_cost``. The result is a list of world ``(x, z)`` points, or None
    when the start and goal cannot be connected.
    """
    if grid is None:
        return None
    step = max(1, int(round(coarse_m / grid.res)))
    inflate_cells = int(math.ceil(inflate_m / grid.res))
    occupied = grid.logodds > grid.config.occupied_logodds
    if inflate_cells > 0:
        occupied = _dilate(occupied, inflate_cells)
    free = (grid.logodds < grid.config.free_logodds) & ~occupied
    count = grid.logodds.shape[0] // step
    if count < 2:
        return None
    span = count * step
    occupied_c = occupied[:span, :span].reshape(count, step, count, step).any(axis=(1, 3))
    free_c = free[:span, :span].reshape(count, step, count, step).any(axis=(1, 3)) & ~occupied_c
    labels = np.zeros((count, count), dtype=np.uint8)
    labels[free_c] = 1
    labels[occupied_c] = 2

    def cell_of(xz):
        col = int(math.floor((float(xz[0]) - grid.origin[0]) / grid.res / step))
        row = int(math.floor((float(xz[1]) - grid.origin[1]) / grid.res / step))
        if row < 0 or col < 0 or row >= count or col >= count:
            return None
        return row, col

    radius = max(1, int(math.ceil(snap_m / (grid.res * step))))
    start = _nearest(labels, cell_of(start_xz), radius, allow_unknown=False)
    if start is None:
        start = _nearest(labels, cell_of(start_xz), radius, allow_unknown=True)
    goal = _nearest(labels, cell_of(goal_xz), radius, allow_unknown=False)
    if goal is None:
        goal = _nearest(labels, cell_of(goal_xz), radius, allow_unknown=True)
    if start is None or goal is None:
        return None
    route = _astar(labels, start, goal, unknown_cost)
    if route is None:
        return None

    def center(row, col):
        return grid.origin + np.array([col * step + step / 2.0, row * step + step / 2.0]) * grid.res

    return [center(row, col) for row, col in route]


def lookahead(path, agent_xz, distance_m):
    """A point ``distance_m`` ahead of the agent along ``path``."""
    if not path:
        return None
    agent = np.asarray(agent_xz, dtype=np.float64)
    if len(path) == 1:
        return np.asarray(path[0], dtype=np.float64)
    best_index, best_dist, best_t = 0, 1e9, 0.0
    for index in range(len(path) - 1):
        start = np.asarray(path[index], dtype=np.float64)
        end = np.asarray(path[index + 1], dtype=np.float64)
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        t = float(np.clip(((agent - start) @ edge) / (length * length), 0.0, 1.0))
        closest = start + t * edge
        dist = float(np.linalg.norm(agent - closest))
        if dist < best_dist:
            best_index, best_dist, best_t = index, dist, t
    walked = 0.0
    start = np.asarray(path[best_index], dtype=np.float64)
    end = np.asarray(path[best_index + 1], dtype=np.float64)
    point = start + best_t * (end - start)
    for index in range(best_index, len(path) - 1):
        nxt = np.asarray(path[index + 1], dtype=np.float64)
        edge = nxt - point
        length = float(np.linalg.norm(edge))
        if walked + length >= distance_m:
            if length < 1e-9:
                return nxt
            return point + edge / length * (distance_m - walked)
        walked += length
        point = nxt
    return np.asarray(path[-1], dtype=np.float64)


def _walkable(label, allow_unknown):
    if label == 2:
        return False
    if label == 0 and not allow_unknown:
        return False
    return True


def _nearest(labels, cell, radius, allow_unknown):
    if cell is None:
        return None
    row, col = cell
    if _walkable(labels[row, col], allow_unknown):
        return cell
    best = None
    best_dist = 1e9
    height, width = labels.shape
    for r in range(max(0, row - radius), min(height, row + radius + 1)):
        for c in range(max(0, col - radius), min(width, col + radius + 1)):
            if not _walkable(labels[r, c], allow_unknown):
                continue
            dist = (r - row) ** 2 + (c - col) ** 2
            if dist < best_dist:
                best, best_dist = (r, c), dist
    return best


def _astar(labels, start, goal, unknown_cost):
    if start == goal:
        return [start]
    height, width = labels.shape
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbors.append((dr, dc, math.hypot(dr, dc)))
    heap = [(0.0, 0, start[0], start[1])]
    tie = 1
    came = {start: None}
    cost = {start: 0.0}
    while heap:
        _, _, row, col = heapq.heappop(heap)
        if (row, col) == goal:
            route = []
            cursor = goal
            while cursor is not None:
                route.append(cursor)
                cursor = came[cursor]
            route.reverse()
            return route
        for dr, dc, step_cost in neighbors:
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= height or nc >= width:
                continue
            if labels[nr, nc] == 2:
                continue
            if dr != 0 and dc != 0 and (labels[row + dr, col] == 2 or labels[row, col + dc] == 2):
                continue
            extra = 1.0 if labels[nr, nc] == 1 else unknown_cost
            next_cost = cost[(row, col)] + step_cost * extra
            key = (nr, nc)
            if next_cost >= cost.get(key, 1e18):
                continue
            cost[key] = next_cost
            came[key] = (row, col)
            heuristic = math.hypot(nr - goal[0], nc - goal[1])
            heapq.heappush(heap, (next_cost + heuristic, tie, nr, nc))
            tie += 1
    return None


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    rows, cols = np.nonzero(mask)
    members = {(int(r), int(c)) for r, c in zip(rows, cols)}
    groups = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in members and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        groups.append(cells)
    return groups


def _snap_free(grid, free, xz, max_m):
    radius = int(math.ceil(max_m / grid.res))
    col0 = int(math.floor((float(xz[0]) - grid.origin[0]) / grid.res))
    row0 = int(math.floor((float(xz[1]) - grid.origin[1]) / grid.res))
    height, width = free.shape
    best = None
    best_dist = max_m
    for row in range(max(0, row0 - radius), min(height, row0 + radius + 1)):
        for col in range(max(0, col0 - radius), min(width, col0 + radius + 1)):
            if not free[row, col]:
                continue
            point = grid.origin + np.array([col + 0.5, row + 0.5]) * grid.res
            dist = float(np.hypot(point[0] - xz[0], point[1] - xz[1]))
            if dist <= best_dist:
                best, best_dist = point, dist
    return best
