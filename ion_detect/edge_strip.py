"""y 向外缘条带：与 ``|y-cy|/b >= 1-F`` 边缘准则一致的几何，用于按列积分 1D 轮廓。"""
from __future__ import annotations

import numpy as np

from .peel import y_edge_band_thresholds


def outer_y_edge_strip_masks(
    boundary: tuple[float, float, float, float],
    y_edge_frac: float,
    image_shape: tuple[int, int],
    *,
    clip_ellipse: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """构造上下「外缘条带」像素掩膜（轴对齐矩形 ∩ 可选椭圆内域）。

    矩形与 README / ``y_edge_band_thresholds`` 一致：
    - 上条带：y 从椭圆上顶点 ``cy - b`` 到截线 ``y_below = cy - (1-F)*b``；
      x 为截线与椭圆两交点之间的弦长 ``|x-cx| <= x_half``。
    - 下条带：y 从 ``y_above = cy + (1-F)*b`` 到 ``cy + b``；x 同上。

    Parameters
    ----------
    boundary : (cx, cy, a, b)
    y_edge_frac : F，与 peak-peel y 向边缘带相同含义。
    image_shape : (H, W)
    clip_ellipse : 若为 True，只保留落在晶格椭圆内的像素。

    Returns
    -------
    top_mask, bot_mask : (H, W) bool
    meta : dict，含 ``x_half``, ``y_below``, ``y_above`` 等浮点几何量。
    """
    cx, cy, a, b = (float(boundary[0]), float(boundary[1]), float(boundary[2]), float(boundary[3]))
    h, w = int(image_shape[0]), int(image_shape[1])
    edges = y_edge_band_thresholds(boundary, y_edge_frac)
    if edges is None:
        raise ValueError("boundary 无效")
    y_below, y_above = float(edges[0]), float(edges[1])
    asafe = max(a, 1e-9)
    bsafe = max(b, 1e-9)

    def chord_half_width(y_line: float) -> float:
        t = (y_line - cy) / bsafe
        if abs(t) > 1.0:
            return 0.0
        inner = 1.0 - t * t
        return float(asafe * np.sqrt(max(0.0, inner)))

    x_half = chord_half_width(y_below)
    if x_half <= 0.0:
        x_half = chord_half_width(y_above)

    y_top_v = cy - bsafe
    y_bot_v = cy + bsafe

    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]

    top_strip = (ys >= y_top_v) & (ys <= y_below)
    top_rect = top_strip & (np.abs(xs - cx) <= x_half)
    bot_strip = (ys >= y_above) & (ys <= y_bot_v)
    bot_rect = bot_strip & (np.abs(xs - cx) <= x_half)

    if clip_ellipse:
        ell = ((xs - cx) / asafe) ** 2 + ((ys - cy) / bsafe) ** 2 <= 1.0
        top_mask = top_rect & ell
        bot_mask = bot_rect & ell
    else:
        top_mask = top_rect
        bot_mask = bot_rect

    meta = {
        "cx": cx,
        "cy": cy,
        "a": a,
        "b": b,
        "y_edge_frac": float(y_edge_frac),
        "y_below": y_below,
        "y_above": y_above,
        "x_half": x_half,
        "y_top_vertex": y_top_v,
        "y_bot_vertex": y_bot_v,
        "clip_ellipse": clip_ellipse,
    }
    return top_mask, bot_mask, meta


def _profile_argmax_index(profile: np.ndarray) -> int:
    """Argmax ignoring NaNs; if none finite, return 0."""
    prof = np.asarray(profile, dtype=np.float64)
    if prof.size == 0:
        return 0
    if not np.any(np.isfinite(prof)):
        return 0
    return int(np.nanargmax(prof))


def _parabola_peak_refine(profile: np.ndarray, idx: int) -> tuple[float, float]:
    """三点抛物线拟合，返回 (亚像素峰位, 峰处插值强度)。"""
    prof = np.asarray(profile, dtype=np.float64)
    n = prof.size
    idx = int(np.clip(idx, 0, max(0, n - 1)))
    if n < 3 or idx <= 0 or idx >= n - 1:
        return float(idx), float(prof[idx])
    y0, y1, y2 = prof[idx - 1], prof[idx], prof[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(idx), float(y1)
    dx = 0.5 * (y0 - y2) / denom
    dx = float(np.clip(dx, -0.5, 0.5))
    x_peak = idx + dx
    y_peak = y1 - 0.25 * (y0 - y2) * dx
    return x_peak, float(y_peak)


def outer_y_edge_column_profiles(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    y_edge_frac: float,
    *,
    clip_ellipse: bool = True,
    col_metric: str = "mean",
) -> dict:
    """对上下外缘条带按列聚合，得到 1D 曲线并定位峰值。

    Parameters
    ----------
    col_metric
        ``sum``：每列掩膜内像素求和；``mean``：除以该列掩膜像素数（y 向均值）；
        ``max``：每列掩膜内像素最大值。椭圆裁剪导致各列行数不同时，``mean`` 更利于列间比对。
        ``max`` 在无掩膜像素列为 NaN。对 bgsub/peel 等带符号量，均值仍按掩膜像素数作除数
        （不用「非零个数」，避免阈值敏感）。
    """
    if col_metric not in ("sum", "mean", "max"):
        raise ValueError(f"col_metric must be 'sum', 'mean', or 'max', got {col_metric!r}")

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    top_m, bot_m, meta = outer_y_edge_strip_masks(
        boundary, y_edge_frac, (h, w), clip_ellipse=clip_ellipse
    )
    top_sum = np.sum(img * top_m, axis=0)
    bot_sum = np.sum(img * bot_m, axis=0)
    top_cnt = np.sum(top_m, axis=0).astype(np.float64)
    bot_cnt = np.sum(bot_m, axis=0).astype(np.float64)

    if col_metric == "sum":
        top_prof = top_sum
        bot_prof = bot_sum
    elif col_metric == "mean":
        top_prof = np.divide(
            top_sum, top_cnt, out=np.zeros_like(top_sum), where=top_cnt > 0.0
        )
        bot_prof = np.divide(
            bot_sum, bot_cnt, out=np.zeros_like(bot_sum), where=bot_cnt > 0.0
        )
    else:
        t_big = np.where(top_m, img, -np.inf)
        b_big = np.where(bot_m, img, -np.inf)
        top_prof = np.max(t_big, axis=0)
        bot_prof = np.max(b_big, axis=0)
        top_prof = np.where(top_cnt > 0, top_prof, np.nan)
        bot_prof = np.where(bot_cnt > 0, bot_prof, np.nan)

    meta = {**meta, "col_metric": col_metric}

    x_grid = np.arange(w, dtype=np.float64)

    itop = _profile_argmax_index(top_prof)
    ibot = _profile_argmax_index(bot_prof)
    peak_top_x, peak_top_val = _parabola_peak_refine(top_prof, itop)
    peak_bot_x, peak_bot_val = _parabola_peak_refine(bot_prof, ibot)

    return {
        "meta": meta,
        "x": x_grid,
        "top_profile": top_prof,
        "bot_profile": bot_prof,
        "top_col_counts": top_cnt,
        "bot_col_counts": bot_cnt,
        "top_mask": top_m,
        "bot_mask": bot_m,
        "top_peak_ix": itop,
        "bot_peak_ix": ibot,
        "top_peak_x": peak_top_x,
        "top_peak_value": peak_top_val,
        "bot_peak_x": peak_bot_x,
        "bot_peak_value": peak_bot_val,
    }
