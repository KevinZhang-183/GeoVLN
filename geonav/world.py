"""One occupancy grid, with portals, landmarks, frontiers and room nodes on it."""

from __future__ import annotations

import numpy as np
from PIL import Image

from geonav._bootstrap import install

install()

from extract_portals import PortalConfig  # noqa: E402
from portal_map import ACTIVE, MapConfig, PortalMap  # noqa: E402
from semantic_map import SemanticConfig, SemanticMap, mask_points, pose_separated  # noqa: E402

from geonav.geom import frontiers as find_frontiers, planner_codes  # noqa: E402

DOOR_WORDS = {"door", "doors", "doorway", "doorways", "arch", "archway"}


class Marker:
    def __init__(self, index, xz, portal_id=None):
        self.id = int(index)
        self.xz = np.asarray(xz, dtype=np.float64)
        self.portal_id = portal_id


class World:
    def __init__(self, sam3=None, key_move_m=0.5, key_turn_deg=30.0):
        self.portals = PortalMap(MapConfig(), PortalConfig())
        self.semantic = SemanticMap(SemanticConfig(), self.portals.portal_config, self.portals.config)
        self.sam3 = sam3
        self.key_move_m = key_move_m
        self.key_turn_deg = key_turn_deg
        self.frontiers = []
        self.nodes = []
        self.new_traversals = []
        self.sam3_calls = 0
        self.sam3_failures = 0
        self._last_sam_pose = None
        self._sam3_warned = False
        self._frontier_seq = 1
        self._node_seq = 1

    @property
    def grid(self):
        return self.portals.grid

    def update(self, depth_m, pose, rgb, step, prompts):
        before = {track.id for track in self.portals.tracks if track.traversed}
        self.portals.update(depth_m, pose, step, frame=f"{step:04d}")
        if self.semantic.grid is None and self.portals.grid is not None:
            self.semantic.attach_grid(self.portals.grid)
        self.semantic.note_agent(pose)
        after = {track.id for track in self.portals.tracks if track.traversed}
        self.new_traversals = sorted(after - before)
        self._add_nodes(pose, step)
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        self.frontiers = _associate(self.frontiers, find_frontiers(self.grid, agent), self._next_frontier_id)
        if prompts and self._due(pose):
            self._segment(rgb, depth_m, pose, step, prompts)

    def active_portals(self):
        hidden = self.portals._hidden_ids()
        return [
            track
            for track in self.portals.tracks
            if track.state in ACTIVE and track.id not in hidden
        ]

    def link_doors(self):
        for instance in self.semantic.instances:
            if not _is_door(instance.label):
                continue
            best, best_dist = None, 1.2
            for track in self.portals.tracks:
                if track.state in ("invalid", "abandoned"):
                    continue
                dist = float(np.hypot(track.center[0] - instance.x, track.center[1] - instance.z))
                if dist < best_dist:
                    best, best_dist = track, dist
            instance.portal_id = None if best is None else best.id

    def render(self, path=None, waypoint=None, coarse=False, inflate_m=0.15, coarse_m=0.2):
        if self.portals.pose is None or self.portals.grid is None:
            return None
        codes_of = None
        if coarse:
            grid = self.portals.grid

            def codes_of(points, grid=grid, inflate_m=inflate_m, coarse_m=coarse_m):
                return planner_codes(grid, points, inflate_m, coarse_m)

        image = self.portals.render(
            egocentric=False,
            px=640,
            codes_of=codes_of,
            overlay=lambda draw, to_px: _overlay(draw, to_px, self, path, waypoint),
        )
        return image

    def _due(self, pose):
        if self._last_sam_pose is None:
            return True
        return pose_separated(self._last_sam_pose, pose, self.key_move_m, self.key_turn_deg)

    def _segment(self, rgb, depth_m, pose, step, prompts):
        if self.sam3 is None:
            return
        result = self.sam3.segment(rgb, prompts)
        if result is None:
            self.sam3_failures += 1
            if not self._sam3_warned:
                print("SAM3 request failed; the map keeps running without new landmarks", flush=True)
                self._sam3_warned = True
            return
        self._last_sam_pose = np.asarray(pose, dtype=np.float64).copy()
        self.sam3_calls += 1
        frame = f"{step:04d}"
        for item in result:
            for mask, score in zip(item["masks"], item["scores"]):
                if mask.shape != depth_m.shape:
                    mask = np.array(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (depth_m.shape[1], depth_m.shape[0]), Image.NEAREST
                        )
                    ) > 0
                points = mask_points(mask, depth_m, self.semantic.config)
                self.semantic.observe_camera(item["prompt"], points, pose, score, frame)
        self.semantic.merge_overlapping()
        self.link_doors()

    def _add_nodes(self, pose, step):
        xz = np.asarray(pose, dtype=np.float64)[[0, 2], 3].copy()
        crossed = self.new_traversals[0] if self.new_traversals else None
        if not self.nodes:
            self.nodes.append(Marker(self._node_seq, xz, crossed))
            self._node_seq += 1
            return
        previous = self.nodes[-1]
        moved = float(np.hypot(xz[0] - previous.xz[0], xz[1] - previous.xz[1]))
        if crossed is None and moved < 2.0:
            return
        self.nodes.append(Marker(self._node_seq, xz, crossed))
        self._node_seq += 1

    def _next_frontier_id(self):
        value = self._frontier_seq
        self._frontier_seq += 1
        return value


def _associate(previous, points, new_id, max_dist=1.0):
    used = set()
    assigned = []
    for xz in points:
        best, best_dist = None, max_dist
        for marker in previous:
            if marker.id in used:
                continue
            dist = float(np.hypot(marker.xz[0] - xz[0], marker.xz[1] - xz[1]))
            if dist < best_dist:
                best, best_dist = marker, dist
        if best is None:
            assigned.append(Marker(new_id(), xz))
        else:
            used.add(best.id)
            assigned.append(Marker(best.id, xz))
    return assigned


def _is_door(label):
    return bool(set(str(label).split()) & DOOR_WORDS)


def _overlay(draw, to_px, world, path, waypoint):
    if path and len(path) > 1:
        draw.line([to_px(point) for point in path], fill=(255, 220, 0), width=2)
    for frontier in world.frontiers:
        x, y = to_px(frontier.xz)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 220, 220))
        draw.text((x + 4, y - 8), f"F{frontier.id}", fill=(0, 180, 180))
    for instance in world.semantic.instances:
        x, y = to_px([instance.x, instance.z])
        radius = 5 if instance.hits >= world.semantic.config.confirm_hits else 3
        color = (0, 180, 70) if instance.hits >= world.semantic.config.confirm_hits else (160, 160, 60)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((x + 4, y - 8), f"L{instance.id}", fill=color)
    for node in world.nodes:
        x, y = to_px(node.xz)
        draw.rectangle((x - 3, y - 3, x + 3, y + 3), outline=(80, 80, 255))
    if waypoint is not None and waypoint.stages:
        x, y = to_px(waypoint.stages[-1].xz)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(220, 30, 30), width=2)
        draw.text((x + 6, y - 10), waypoint.name, fill=(220, 30, 30))
