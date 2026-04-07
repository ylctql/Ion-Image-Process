"""合并后轴对齐矩形：按 y 向条带分割并计算离子平衡坐标（矩形中心或条带内前景质心）。"""
from __future__ import annotations

import numpy as np

from .blob_components import MinAreaRect


def _rect_aabb(r: MinAreaRect) -> tuple[float, float, float, float]:
    c = np.asarray(r.corners_xy, dtype=np.float64)
    xmin = float(c[:, 0].min())
    xmax = float(c[:, 0].max())
    ymin = float(c[:, 1].min())
    ymax = float(c[:, 1].max())
    return xmin, xmax, ymin, ymax


def _no_split_aabbs(
    rects: list[MinAreaRect],
    *,
    split: bool,
    max_ysize: float,
    eps: float = 1e-9,
) -> list[tuple[float, float, float, float]]:
    """``split=True`` 时，y 向不需要分割的矩形（``height <= max_ysize``）的 AABB 列表。"""
    if not split:
        return []
    thr = float(max_ysize)
    out: list[tuple[float, float, float, float]] = []
    for r in rects:
        _, _, ymin, ymax = _rect_aabb(r)
        height = ymax - ymin
        if height <= thr + eps:
            out.append(_rect_aabb(r))
    return out


def ion_equilibrium_positions_xy(
    rects: list[MinAreaRect],
    binary: np.ndarray,
    *,
    split: bool = False,
    max_ysize: float = 9.0,
) -> list[tuple[float, float]]:
    """
    对每个矩形给出一个或多个 ``(x, y)`` 识别位置（像素坐标，与 ``imshow`` 一致）。

    - ``split=False``：每个矩形一个点，为 ``MinAreaRect.center_xy``（轴对齐盒几何中心）。
    - ``split=True``：若矩形 y 向跨度 ``height > max_ysize``，令
      ``n = ceil(height / max_ysize)``，将 ``[ymin, ymax]`` 等分为 ``n`` 段；每段内对
      ``binary`` 为真的像素求形心。若该条带与某个**不需要 split** 的矩形（``height <= max_ysize``）
      的 AABB 相交，则先从该条带候选点中去掉落在这些 AABB 并集内的像素，再对剩余点求质心；
      若减去后无点，则退化为该条带几何中心。不需分割的矩形仍为一个点 ``center_xy``。

    合并矩形可能覆盖多个连通域，条带内质心一律用 ``binary`` 前景而非单标签掩膜。
    """
    if not rects:
        return []
    bin_mask = np.asarray(binary, dtype=bool)
    thr = float(max_ysize)
    if thr <= 0:
        raise ValueError("max_ysize must be positive")
    out: list[tuple[float, float]] = []
    eps = 1e-9
    other_boxes = _no_split_aabbs(rects, split=split, max_ysize=max_ysize, eps=eps)

    for r in rects:
        xmin, xmax, ymin, ymax = _rect_aabb(r)
        height = ymax - ymin

        if not split or height <= thr + eps:
            out.append((float(r.center_xy[0]), float(r.center_xy[1])))
            continue

        n = int(np.ceil(height / thr))
        if n < 2:
            out.append((float(r.center_xy[0]), float(r.center_xy[1])))
            continue

        ys_all, xs_all = np.where(bin_mask)
        if ys_all.size == 0:
            for i in range(n):
                y0 = ymin + i * height / n
                y1 = ymin + (i + 1) * height / n
                out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            continue
        xf = xs_all.astype(np.float64)
        yf = ys_all.astype(np.float64)
        m_rect = (
            (xf >= xmin - eps)
            & (xf <= xmax + eps)
            & (yf >= ymin - eps)
            & (yf <= ymax + eps)
        )
        xs_r = xf[m_rect]
        ys_r = yf[m_rect]
        for i in range(n):
            y0 = ymin + i * height / n
            y1 = ymin + (i + 1) * height / n
            m_strip = (ys_r >= y0 - eps) & (ys_r <= y1 + eps)
            m_use = m_strip
            for qx0, qx1, qy0, qy1 in other_boxes:
                m_in_other = (
                    (xs_r >= qx0 - eps)
                    & (xs_r <= qx1 + eps)
                    & (ys_r >= qy0 - eps)
                    & (ys_r <= qy1 + eps)
                )
                m_use = m_use & (~m_in_other)

            if not np.any(m_use):
                out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            else:
                out.append((float(xs_r[m_use].mean()), float(ys_r[m_use].mean())))

    return out
