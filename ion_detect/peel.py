"""峰值剥离：合并去重与 y 向边缘带过滤。"""
import numpy as np


def merge_ions_by_distance(primary: list, extra: list, min_sep_px: float) -> list:
    """将 extra 中中心与已有列表距离 ≥ min_sep_px 的离子并入, 再按 (y,x) 排序。"""
    if not extra:
        return primary
    out = list(primary)
    sep2 = float(min_sep_px) ** 2
    for ion in extra:
        x0, y0 = float(ion["x0"]), float(ion["y0"])
        ok = True
        for ex in out:
            dx = float(ex["x0"]) - x0
            dy = float(ex["y0"]) - y0
            if dx * dx + dy * dy < sep2:
                ok = False
                break
        if ok:
            out.append(ion)
    out.sort(key=lambda d: (d["y0"], d["x0"]))
    return out


def y_edge_band_thresholds(boundary, y_edge_frac: float) -> tuple[float, float] | None:
    """与 ``filter_peak_yx_y_edge_bands`` 相同归一化准则下的两条分界线 y 坐标 (像素, 与 ion y0 一致).

    第二轮若仅保留 y 向边缘带, 则保留 ``y <= y_below`` 或 ``y >= y_above`` 的候选
    (``y_below = cy - (1-F)*b``, ``y_above = cy + (1-F)*b``, ``F = y_edge_frac``)。
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


def filter_peak_yx_y_edge_bands(peak_yx, boundary, y_edge_frac: float):
    """仅保留靠近晶格椭圆 y 向两端的候选 (用于峰值剥离第二轮).

    使用与边界过滤相同的 (cx, cy, a, b), 要求 |py - cy| / b >= 1 - y_edge_frac。
    boundary 为 None 时原样返回。
    """
    if boundary is None or peak_yx.size == 0:
        return peak_yx
    _cx, cy, _a, b = boundary
    f = float(y_edge_frac)
    f = min(max(f, 1e-6), 1.0 - 1e-6)
    thr = 1.0 - f
    py = peak_yx[:, 0].astype(np.float64)
    ny = np.abs(py - cy) / max(float(b), 1e-9)
    return peak_yx[ny >= thr]


def filter_ions_y_edge_bands(ions: list, boundary, y_edge_frac: float) -> list:
    """按与 filter_peak_yx_y_edge_bands 相同准则过滤拟合后的离子中心。"""
    if boundary is None or not ions:
        return ions
    _cx, cy, _a, b = boundary
    f = float(y_edge_frac)
    f = min(max(f, 1e-6), 1.0 - 1e-6)
    thr = 1.0 - f
    out = []
    bsafe = max(float(b), 1e-9)
    for ion in ions:
        ny = abs(float(ion["y0"]) - cy) / bsafe
        if ny >= thr:
            out.append(ion)
    return out
