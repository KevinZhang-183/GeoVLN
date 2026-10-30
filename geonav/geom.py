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


def frontiers(
    grid,
    agent_xz,
    min_m=0.5,
    pull_m=0.5,
    limit=4,
    merge_m=1.5,
    gap_m=0.15,
    portals_xz=None,
    portal_m=1.0,
    inflate_m=0.15,
    gain_m=2.0,
):
    """One standable point per opening between free and unknown space.

    Gaps of about ``gap_m`` are closed before clustering, and points closer than
    ``merge_m`` collapse into the larger opening. Obstacle inflation only rejects
    a point the robot cannot stand on. Openings shorter than ``min_m``, and
    openings already covered by a portal, are dropped. The rest are ranked by
    unknown area beyond the opening divided by the A* cost.
    """
    if grid is None or limit <= 0:
        return []
    occupied = grid.logodds > grid.config.occupied_logodds
    free = (grid.logodds < grid.config.free_logodds) & ~occupied
    unknown = ~free & ~occupied
    edge = _dilate(unknown, 1) & free
    if not np.any(edge):
        return []
    agent = np.asarray(agent_xz, dtype=np.float64)
    radius = max(1, int(math.ceil(gap_m / max(grid.res, 1e-6) / 2.0)))
    bridged = _dilate(edge, radius) & ~occupied
    inflate_cells = int(math.ceil(inflate_m / max(grid.res, 1e-6)))
    blocked = _dilate(occupied, inflate_cells) if inflate_cells else occupied
    stand = free & ~blocked
    portals = _as_points(portals_xz)

    clusters = []
    for cells in _components(bridged):
        members = [cell for cell in cells if edge[cell]]
        if not members:
            continue
        boundary = _closest_xz(grid, members, agent)
        pulled = _pull_toward(boundary, agent, pull_m)
        clusters.append({
            "anchor": _cells_xz(grid, members).mean(axis=0),
            "boundary": boundary,
            "point": _snap_free(grid, stand, pulled, max_m=max(pull_m, 0.8)),
            "size": len(members),
            "length": _opening_length(grid, members),
            "dist": float(np.hypot(*(boundary - agent))),
        })
    clusters.sort(key=lambda item: (-item["size"], item["dist"]))
    kept = []
    for item in clusters:
        host, host_gap = None, merge_m
        for other in kept:
            gap = _cluster_gap(item, other)
            if gap < host_gap:
                host, host_gap = other, gap
        if host is None:
            kept.append(item)
            continue
        host["length"] += item["length"]
        host["size"] += item["size"]
        if host["point"] is None and item["point"] is not None:
            host["point"] = item["point"]
            host["boundary"] = item["boundary"]

    ranked = []
    for item in kept:
        if item["point"] is None or item["length"] < min_m:
            continue
        if _near_any(item["boundary"], portals, portal_m):
            continue
        area = _unknown_area(grid, unknown, item["boundary"], agent, gain_m)
        ranked.append((np.asarray(item["point"], dtype=np.float64).copy(), area))
    if not ranked:
        return []
    costs = _route_costs(grid, agent, [point for point, _ in ranked], inflate_m)
    scored = []
    for (point, area), cost in zip(ranked, costs):
        reachable = cost is not None
        travel = max(float(cost), 0.5) if reachable else float(np.hypot(*(point - agent))) + 50.0
        scored.append((reachable, area / travel, travel, point))
    scored.sort(key=lambda item: (not item[0], -item[1], item[2]))
    return [point for _, _, _, point in scored[:limit]]


def _as_points(points_xz):
    if points_xz is None:
        return np.zeros((0, 2))
    points = np.asarray(points_xz, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 2))
    return np.atleast_2d(points).reshape(-1, 2)


def _cells_xz(grid, cells):
    arr = np.asarray(cells, dtype=np.float64)
    return np.column_stack((
        grid.origin[0] + (arr[:, 1] + 0.5) * grid.res,
        grid.origin[1] + (arr[:, 0] + 0.5) * grid.res,
    ))


def _closest_xz(grid, cells, agent):
    points = _cells_xz(grid, cells)
    dist = np.hypot(points[:, 0] - agent[0], points[:, 1] - agent[1])
    return points[int(np.argmin(dist))].copy()


def _pull_toward(origin, toward, distance):
    delta = np.asarray(toward, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.asarray(origin, dtype=np.float64).copy()
    return np.asarray(origin, dtype=np.float64) + delta / norm * min(distance, norm)


def _cluster_gap(left, right):
    """Distance between opening centers. Nearby doors stay apart; fragments of one do not."""
    return float(np.hypot(*(left["anchor"] - right["anchor"])))


def _opening_length(grid, cells):
    total = 0.0
    for run in _runs(cells):
        if len(run) == 1:
            total += grid.res
            continue
        pts = np.asarray(run, dtype=np.float64)
        delta = pts - pts.mean(axis=0)
        _, vecs = np.linalg.eigh(delta.T @ delta)
        proj = delta @ vecs[:, 1]
        total += float(proj.max() - proj.min() + 1.0) * grid.res
    return total


def _runs(cells):
    remaining = set(cells)
    runs = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        run = [start]
        while stack:
            row, col = stack.pop()
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    nxt = (row + d_row, col + d_col)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
                        run.append(nxt)
        runs.append(run)
    return runs


def _near_any(xz, points, radius_m):
    if len(points) == 0:
        return False
    dist = np.hypot(points[:, 0] - xz[0], points[:, 1] - xz[1])
    return bool(np.any(dist < radius_m))


def _unknown_area(grid, unknown, boundary, agent, radius_m):
    """Unknown area within ``radius_m``, on the far side of the opening."""
    direction = np.asarray(boundary, dtype=np.float64) - np.asarray(agent, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6 or radius_m <= 0.0:
        return 0.0
    direction /= norm
    reach = int(math.ceil(radius_m / grid.res))
    col = int(math.floor((float(boundary[0]) - grid.origin[0]) / grid.res))
    row = int(math.floor((float(boundary[1]) - grid.origin[1]) / grid.res))
    row0 = max(0, row - reach)
    row1 = min(unknown.shape[0], row + reach + 1)
    col0 = max(0, col - reach)
    col1 = min(unknown.shape[1], col + reach + 1)
    window = unknown[row0:row1, col0:col1]
    if not window.any():
        return 0.0
    rows, cols = np.nonzero(window)
    points = np.column_stack((
        grid.origin[0] + (cols + col0 + 0.5) * grid.res,
        grid.origin[1] + (rows + row0 + 0.5) * grid.res,
    ))
    delta = points - np.asarray(boundary, dtype=np.float64)
    dist = np.hypot(delta[:, 0], delta[:, 1])
    ahead = delta @ direction
    count = int(np.count_nonzero((dist <= radius_m) & (ahead > 0.0)))
    return count * grid.res * grid.res


def _route_costs(grid, start_xz, goals_xz, inflate_m, unknown_cost=5.0, coarse_m=0.2, snap_m=1.0):
    """A* length of each goal on one coarsened grid. ``None`` when unreachable."""
    goals = list(goals_xz)
    if not goals:
        return []
    built = planner_grid(grid, inflate_m, coarse_m)
    if built is None:
        return [None] * len(goals)
    labels, origin, cell_m = built
    count = labels.shape[0]

    def cell_of(xz):
        col = int(math.floor((float(xz[0]) - origin[0]) / cell_m))
        row = int(math.floor((float(xz[1]) - origin[1]) / cell_m))
        if row < 0 or col < 0 or row >= count or col >= count:
            return None
        return row, col

    radius = max(1, int(math.ceil(snap_m / cell_m)))
    start = _nearest(labels, cell_of(start_xz), radius, allow_unknown=False)
    if start is None:
        start = _nearest(labels, cell_of(start_xz), radius, allow_unknown=True)
    costs = []
    for goal_xz in goals:
        goal = _nearest(labels, cell_of(goal_xz), radius, allow_unknown=False)
        if goal is None:
            goal = _nearest(labels, cell_of(goal_xz), radius, allow_unknown=True)
        if start is None or goal is None:
            costs.append(None)
            continue
        route = _astar(labels, start, goal, unknown_cost)
        if route is None:
            costs.append(None)
            continue
        length = 0.0
        for (row0, col0), (row1, col1) in zip(route, route[1:]):
            length += math.hypot(row1 - row0, col1 - col0) * cell_m
        costs.append(length)
    return costs


def planner_grid(grid, inflate_m=0.15, coarse_m=0.2):
    """The occupancy grid A* searches, after inflation and coarsening.

    Returns ``(labels, origin_xz, cell_m)``. ``labels`` is uint8 with 0 unknown,
    1 free and 2 occupied. A coarse cell is occupied when any inflated fine cell
    inside it is occupied. ``None`` when the grid is too small to coarsen.
    """
    if grid is None:
        return None
    step = max(1, int(round(coarse_m / grid.res)))
    inflate_cells = int(math.ceil(inflate_m / max(grid.res, 1e-6)))
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
    return labels, np.asarray(grid.origin, dtype=np.float64), grid.res * step


def planner_codes(grid, points_xz, inflate_m=0.15, coarse_m=0.2):
    """Classify world ``(x, z)`` points on the grid A* searches."""
    pts = np.atleast_2d(np.asarray(points_xz, dtype=np.float64))
    out = np.zeros(len(pts), dtype=np.uint8)
    built = planner_grid(grid, inflate_m, coarse_m)
    if built is None or len(pts) == 0:
        return out
    labels, origin, cell_m = built
    col = np.floor((pts[:, 0] - origin[0]) / cell_m).astype(np.int64)
    row = np.floor((pts[:, 1] - origin[1]) / cell_m).astype(np.int64)
    height, width = labels.shape
    ok = (row >= 0) & (col >= 0) & (row < height) & (col < width)
    out[ok] = labels[row[ok], col[ok]]
    return out


def plan(grid, start_xz, goal_xz, inflate_m=0.15, unknown_cost=5.0, coarse_m=0.2, snap_m=1.0):
    """A* on a coarsened copy of the occupancy grid.

    Occupied cells are dilated by ``inflate_m``. Unknown cells are walkable at
    ``unknown_cost``. The result is a list of world ``(x, z)`` points, or None
    when the start and goal cannot be connected.
    """
    built = planner_grid(grid, inflate_m, coarse_m)
    if built is None:
        return None
    labels, origin, cell_m = built
    count = labels.shape[0]

    def cell_of(xz):
        col = int(math.floor((float(xz[0]) - origin[0]) / cell_m))
        row = int(math.floor((float(xz[1]) - origin[1]) / cell_m))
        if row < 0 or col < 0 or row >= count or col >= count:
            return None
        return row, col

    radius = max(1, int(math.ceil(snap_m / cell_m)))
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
        return origin + np.array([col + 0.5, row + 0.5]) * cell_m

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
