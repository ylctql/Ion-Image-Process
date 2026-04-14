"""椭圆 y 外缘条带（``y_edge_frac``）内 y 向过薄的轴对齐外接矩形与最近邻合并。"""
from __future__ import annotations

import numpy as np

from .blob_components import MinAreaRect, axis_aligned_bounding_rect_xy, rect_component_labels
from .edge_strip import outer_y_edge_strip_masks


def rect_intersects_edge_band(rect: MinAreaRect, band: np.ndarray) -> bool:
    """判断矩形（由 ``corners_xy`` 定义的 AABB）是否与条带掩膜 ``band`` 有像素交集。"""
    c = rect.corners_xy
    h, w = band.shape
    x_lo = int(np.floor(float(c[:, 0].min())))
    x_hi = int(np.ceil(float(c[:, 0].max())))
    y_lo = int(np.floor(float(c[:, 1].min())))
    y_hi = int(np.ceil(float(c[:, 1].max())))
    x_lo = max(0, min(x_lo, w - 1))
    x_hi = max(0, min(x_hi, w - 1))
    y_lo = max(0, min(y_lo, h - 1))
    y_hi = max(0, min(y_hi, h - 1))
    if x_lo > x_hi or y_lo > y_hi:
        return False
    return bool(band[y_lo : y_hi + 1, x_lo : x_hi + 1].any())


def _merge_two_min_area_rects(a: MinAreaRect, b: MinAreaRect) -> MinAreaRect:
    corners = np.vstack((a.corners_xy, b.corners_xy))
    geo = axis_aligned_bounding_rect_xy(corners)
    if geo is None:
        return a
    labs = tuple(sorted(set(rect_component_labels(a)) | set(rect_component_labels(b))))
    return MinAreaRect(
        label=min(a.label, b.label),
        center_xy=(float(geo["center"][0]), float(geo["center"][1])),
        width=float(geo["width"]),
        height=float(geo["height"]),
        angle_deg=float(geo["angle_deg"]),
        corners_xy=np.asarray(geo["corners"], dtype=np.float64),
        area_pixels=int(a.area_pixels + b.area_pixels),
        from_edge_merge=True,
        component_labels=labs,
    )


def merge_edge_band_sliver_rects(
    rects: list[MinAreaRect],
    boundary: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    *,
    y_edge_frac: float = 0.3,
    min_edge_ysize: float = 5.0,
    clip_ellipse: bool = True,
) -> tuple[list[MinAreaRect], int]:
    """
    在与 ``outer_y_edge_strip_masks`` 一致的上下外缘条带并集中，对满足 ``height < min_edge_ysize``
    的轴对齐外接矩形，逐个与其余矩形中中心距最近者合并；合并结果为二者像素坐标的最小 AABB。

    Parameters
    ----------
    y_edge_frac
        外缘带参数 ``F``（默认 0.3 即 30%），几何含义同 :func:`~ion_detect.edge_strip.outer_y_edge_strip_masks`。
    min_edge_ysize
        y 向边长（``MinAreaRect.height``）低于该值且在条带内的矩形才会被合并（默认 5）。
    clip_ellipse
        条带是否再与椭圆内域求交，与 ``outer_y_edge_strip_masks`` 一致。

    Returns
    -------
    rects_out, n_merges
        合并后的矩形列表与发生合并的次数。
    """
    if not rects:
        return [], 0
    h, w = int(image_shape[0]), int(image_shape[1])
    top_m, bot_m, _meta = outer_y_edge_strip_masks(
        boundary, float(y_edge_frac), (h, w), clip_ellipse=clip_ellipse
    )
    band = top_m | bot_m

    work = list(rects)
    n_merges = 0
    thr = float(min_edge_ysize)

    while True:
        cand_ix: list[int] = []
        for i, r in enumerate(work):
            if r.height >= thr - 1e-12:
                continue
            if rect_intersects_edge_band(r, band):
                cand_ix.append(i)
        if not cand_ix:
            break
        cand_ix.sort(key=lambda i: (work[i].height, work[i].label))
        i = cand_ix[0]
        if len(work) < 2:
            break

        cx_i, cy_i = work[i].center_xy

        def _dist_key(j: int) -> tuple[float, int]:
            cx_j, cy_j = work[j].center_xy
            d2 = (cx_j - cx_i) ** 2 + (cy_j - cy_i) ** 2
            return (d2, j)

        others = [j for j in range(len(work)) if j != i]
        best_j = min(others, key=_dist_key)

        merged = _merge_two_min_area_rects(work[i], work[best_j])
        for k in sorted((i, best_j), reverse=True):
            work.pop(k)
        work.append(merged)
        n_merges += 1

    return work, n_merges
