# 空洞提取

`extract_portals.py` 从一帧深度图里找出能走过去的开口。两侧都有门框的标成黄色，只有一侧门框、另一侧贴着画面边缘的标成橙色。门框离摄像头太近时不出框。

脚本同时给出每个开口在地面上的位置，供占据图使用。相机水平坐标是：x 向右，z 向前，单位米。

## 原理

深度图先按列压缩成一条曲线。取画面高度 18% 到 62% 这一段，每列用 0.55 分位数，再做一次宽度为 7 的中值平滑。分位数能避开门洞里偶发的近处噪声，平滑能去掉单列抖动。

开口是这条曲线上的一个凹槽：某一列比左右邻域至少远 `rise_m`（默认 1.35 米），就从最远的那一列向两边收到深度回落的位置。贴着画面左缘或右缘、先远后突然变近的，单独算一个开口，因为另一侧门框在画面外面。

水平范围确定之后，再沿开口中线逐行往下看。一行的有效深度比门框远 `row_gap_m` 以上，这一行才属于开口。连续的行合成框的上下边界，并继续向下延伸，直到开口中线和旁边的墙落到同一层地面。深度小于 `depth_min` 的像素不参与，仿真里的天空和缺失网格通常是 0，所以那种区域不会被框进去。

通过形状检查的候选分成三类：

- **黄色**：左右都在画面内，是完整门洞。按探索潜力从高到低编号，潜力是开口宽度乘以洞内比门框远出的距离。重叠超过 `overlap` 的只留潜力更高的一个。
- **橙色**：左右有一边贴着图像边缘，标 `E`。它只说明开口在这一侧，宽度和深度都只看到一半，不参与黄色框的排序。
- **丢弃**：写在 `boxes.txt` 的 `drop` 行里，不画框。

## 输出

对一个含 `rgb/` 和 `depth/` 的轨迹目录，结果写到同级的输出目录：

| 文件 | 内容 |
| --- | --- |
| `*.jpg` | 画了黄框和橙框的原图，隔 `stride` 帧一张 |
| `*_1.jpg`、`*_2.jpg` | 黄色开口裁出的 RGB，序号与图上的编号一致 |
| `boxes.txt` | 每个框的潜力、宽度、洞内深度、门框距离、像素框 |
| `portals.json` | 同上，并带地面线段 `segment_xz` |
| `instruction.txt` | 从轨迹目录复制的指令，有则复制 |

`segment_xz` 是门框处的地面线段，两个点都是 `[x, z]`。左右端点分别用 `left_jamb_m`、`right_jamb_m`（开口两侧墙面的深度），斜着看门时线段也能保持真实朝向；贴着画面边缘的一侧没有门框，这两个字段为 `null`，回退到 `jamb_m`。

## 命令

在 `GeoVLN` 目录下：

```bash
python scripts/extract_portals.py --frames /path/to/r2r_frames --out-name portals --stride 4
```

只跑一条轨迹：

```bash
python scripts/extract_portals.py --episode /path/to/episode --near-jamb 1.2
```

`--no-crops` 不保存开口裁图。深度若是 0 到 1 的归一化值（整幅最大值不超过 1.5），会乘以 `depth_max` 还原成米。

## 在代码里调用

```python
from scripts.extract_portals import (
    PortalConfig,
    detect,
    opening_crop,
    doorway_segment,
    opening_points,
    habitat_agent_matrix,
    camera_xz_to_world,
    stamp_occupancy,
    load_depth,
)

config = PortalConfig(near_jamb_m=1.2)
depth = load_depth("depth/028.npy", config)
openings, edges, dropped = detect(depth, config)

crop = opening_crop(rgb, openings[0])
segment = doorway_segment(openings[0].span, openings[0].jamb_m, depth.shape, config)
points = opening_points(depth, openings[0], config, stride=4)
```

`detect` 返回三个列表。黄色在 `openings`，橙色在 `edges`，被过滤的在 `dropped`，原因在 `reason`。

## 投影到占据图

地面上有两种点，都在相机坐标里：

- `doorway_segment` 是门框位置的一条短线段，适合在占据图上标出“这里有一个洞”。
- `opening_points` 是黄框内部有有效深度的像素，投影到地面后的点云，适合标出透过开口能看到的范围。像素过多时用 `stride` 抽稀。

换成世界坐标时，用采集时存下的 `meta.json` 里该帧的 `position` 和 `rotation_wxyz`：

```python
pose = habitat_agent_matrix(frame["position"], frame["rotation_wxyz"])
world_xz = camera_xz_to_world(segment, pose)
stamp_occupancy(grid, origin_xz, resolution, world_xz, value)
```

`camera_xz_to_world` 假定 `pose` 是 Habitat 智能体位姿：局部 x 向右、y 向上、z 向后。函数会先把“向前”的 z 取反，再乘这位姿。返回值是世界坐标的 `(x, z)`，不含高度。

`stamp_occupancy` 的格子约定：`grid[row, col]`，`origin_xz` 是 `grid[0, 0]` 的世界 `(x, z)`，列号沿世界 x 增加，行号沿世界 z 增加，`resolution` 是每格米数。超出地图的点会被丢掉。

## 可调参数

参数都在 `PortalConfig` 里，默认值就是当前这版。命令行目前只暴露 `--near-jamb`，其余在代码里改完再传给 `detect` 或 `process_episode`。

常用的几个：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `hfov` | 79 | 水平视场角，必须和采集相机一致。焦距由它和图像宽度算出 |
| `camera_height` | 1.25 | 相机离地高度，米。用来判断开口有没有落到地面 |
| `depth_min` | 0.2 | 小于这个值的深度视为无效 |
| `depth_max` | 10 | 深度上限，也是归一化深度的还原倍数 |
| `near_jamb_m` | 1.2 | 门框距离小于这个值则不画框。人已经站在门口，画面几乎全是洞内 |
| `min_width_m` | 0.40 | 开口窄于这个值则丢掉 |
| `max_width_m` | 3.15 | 开口宽于这个值则丢掉。空房间的远墙经常更宽 |
| `edge_max_width_m` | 2.45 | 贴边开口宽于这个值则丢掉，避免把半个房间当成门 |
| `min_aspect` | 1.25 | 框高除以框宽。宽于 `squat_min_width_m` 又比这个更扁，视为房间而不是门 |
| `squat_min_width_m` | 1.5 | 只有宽过这个值才检查扁框 |
| `rise_m` | 1.35 | 一列比旁边远多少米，才算跨过门框 |
| `overlap` | 0.55 | 两个黄框重叠面积占较小框的比例超过这个值时，只留潜力更高的 |

过滤原因和对应参数：

| `reason` | 含义 | 相关参数 |
| --- | --- | --- |
| `near` | 门框太近 | `near_jamb_m` |
| `too_wide` | 开口太宽 | `max_width_m` |
| `squat` | 又宽又扁 | `min_aspect`、`squat_min_width_m` |
| `edge_wide` | 贴边且太宽 | `edge_max_width_m` |
| `above_floor` | 框底到不了该距离上的地面 | `camera_height`、`floor_margin` |
| `shallow` | 远处墙面上的浅凹 | `shallow_jamb_m`、`shallow_delta_m`、`shallow_max_width_m` |
| `skew` | 两侧门框深度差太大 | `skew_min_width_m`、`skew_delta_m` |
| `pillar` | 一侧只是细柱，柱外又是远处 | `pillar_min_width_m`、`pillar_near_m`、`pillar_jump_m` |
| `open_side` | 宽开口的一侧没有墙 | `open_side_min_width_m` |

其余参数一般不用改：`body_top`、`body_bottom`、`body_quantile` 决定用画面哪一段估计列深度；`side_px`、`gap_px`、`event_px`、`search_px`、`max_span_px`、`min_span_px` 是按 640×480、79 度画面标定的像素窗口。换分辨率或视场角时，这些像素阈值要一起看。

## 限制

天空、缺失网格和被近裁剪面切掉的表面没有有效深度，开口只会框到还有深度的部分。视场角、图像尺寸和 `depth_max` 要与采集时的 `config/vln_r2r.yaml` 一致，当前是 79 度、640×480、最远 10 米。
