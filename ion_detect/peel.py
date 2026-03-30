"""y 向外缘条带几何：与 edge_strip / 可视化共用分界阈值。"""
from __future__ import annotations


def y_edge_band_thresholds(boundary, y_edge_frac: float) -> tuple[float, float] | None:
    """在 ``|y-cy|/b >= 1 - y_edge_frac`` 准则下的两条分界线 y 坐标 (像素, 与 ion y0 一致).

    ``y_below = cy - (1-F)*b``, ``y_above = cy + (1-F)*b``, ``F = y_edge_frac``。
    无 boundary 时返回 None.
    """
    if boundary is None:
        return None
    _cx, cy, _a, b = boundary
    f = float(y_edge_frac)
    f = min(max(f, 1e-6), 1.0 - 1e-6)
    thr = 1.0 - f
    bsafe = max(float(b), 1e-9)
    y_below = float(cy - thr * bsafe)
    y_above = float(cy + thr * bsafe)
    return y_below, y_above
