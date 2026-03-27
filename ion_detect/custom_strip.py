"""轴对齐矩形条带：按列聚合（y 向积分/均值/最大）→ x 向辅助峰 → 列内 y 中心（COM / 与 edge_strip 相同链路）。"""
from __future__ import annotations

import numpy as np

from .edge_strip import _parabola_peak_refine, _profile_argmax_index
from .edge_strip_profile_analysis import (
    column_y_profile_in_strip,
    strip_profile_peaks_ixy,
    y_center_from_y_profile_fit,
    y_center_of_mass_from_profile,
    y_marked_com_nearest_local_peak,
)


def axis_aligned_rect_mask(
    image_shape: tuple[int, int],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> tuple[np.ndarray, dict]:
    """构造轴对齐矩形条带掩膜；**x 为列、y 为行**（行 0 在图像顶部）。

    边界为 **含端点的整数像素索引**，会裁剪到 ``[0, W-1]`` / ``[0, H-1]``。
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    x0 = int(np.floor(min(float(x_lo), float(x_hi))))
    x1 = int(np.ceil(max(float(x_lo), float(x_hi))))
    y0 = int(np.floor(min(float(y_lo), float(y_hi))))
    y1 = int(np.ceil(max(float(y_lo), float(y_hi))))
    x0 = int(np.clip(x0, 0, max(0, w - 1)))
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y0 = int(np.clip(y0, 0, max(0, h - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    mask = np.zeros((h, w), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    meta = {
        "x_lo": x0,
        "x_hi": x1,
        "y_lo": y0,
        "y_hi": y1,
        "h": h,
        "w": w,
    }
    return mask, meta


def pixel_y_range_from_crystal_u(
    boundary: tuple[float, float, float, float],
    u_lo: float,
    u_hi: float,
    image_height: int,
) -> tuple[float, float]:
    """由晶格相对纵坐标得到像素行范围（含端点，裁剪到图像）。

    定义 **u = (cy - y_row) / b**（与竖直半轴 ``b`` 归一化）：
    **u > 0** 表示行 **y < cy**，即图像上在中心 **上方**；**u < 0** 表示在中心 **下方**。
    给定 ``u`` 闭区间（两参数不必有序），条带覆盖的像素行满足
    ``y in [cy - u_max*b, cy - u_min*b]``（整数化前），再与 ``[0, H-1]`` 求交。
    """
    _cx, cy, _a, b = boundary
    h = int(image_height)
    bsafe = max(float(b), 1e-9)
    u0 = float(min(u_lo, u_hi))
    u1 = float(max(u_lo, u_hi))
    y_rows_small = cy - u1 * bsafe
    y_rows_large = cy - u0 * bsafe
    y_lo = float(min(y_rows_small, y_rows_large))
    y_hi = float(max(y_rows_small, y_rows_large))
    y_lo = float(np.clip(y_lo, 0, max(0, h - 1)))
    y_hi = float(np.clip(y_hi, 0, max(0, h - 1)))
    return y_lo, y_hi


def pixel_y_range_from_crystal_tau(
    boundary: tuple[float, float, float, float],
    tau_lo: float,
    tau_hi: float,
    image_height: int,
) -> tuple[float, float]:
    """由 **竖直直径** 归一化坐标得到像素行范围（含端点，裁剪到图像）。

    与 ``outer_y_edge_strip_masks`` 文档中「从上顶点 ``cy-b`` 起算」的零点一致：

    .. math:: \\tau = (y_{row} - (c_y - b)) / (2b)

    - **τ = 0**：椭圆 **上** 竖直外顶点（``y = cy - b``）；
    - **τ = 0.5**：晶格中心（``y = cy``）；
    - **τ = 1**：**下** 竖直外顶点（``y = cy + b``）。

    与 ``u = (c_y - y)/b``（CLI ``--y-norm-style center_u``）的换算为 **u = 1 - 2τ**。上外缘条带
    ``y \\in [cy-b,\\; cy-(1-F)b]`` 对应 **τ ∈ [0, F/2]**。

    两参数 ``tau_lo, tau_hi`` 不必有序；``τ`` 可在 [0,1] 外（几何延长），``y`` 仍裁到图像内。
    """
    _cx, cy, _a, b = boundary
    h = int(image_height)
    bsafe = max(float(b), 1e-9)
    t0 = float(min(tau_lo, tau_hi))
    t1 = float(max(tau_lo, tau_hi))
    y_a = (cy - bsafe) + 2.0 * bsafe * t0
    y_b = (cy - bsafe) + 2.0 * bsafe * t1
    y_lo = float(min(y_a, y_b))
    y_hi = float(max(y_a, y_b))
    y_lo = float(np.clip(y_lo, 0, max(0, h - 1)))
    y_hi = float(np.clip(y_hi, 0, max(0, h - 1)))
    return y_lo, y_hi


def pixel_y_range_from_y_edge_frac_interval(
    boundary: tuple[float, float, float, float],
    f_lo: float,
    f_hi: float,
    image_height: int,
    *,
    top: bool,
) -> tuple[float, float]:
    """与 ``outer_y_edge_strip_masks(..., y_edge_frac=F)`` 同一套 **F** 参数命名的纵带。

    在上带，行 ``y`` 到外顶 ``cy-b`` 的参量 **F(y) = (y - (cy - b)) / b**（``F=0`` 上外顶，
    ``F=1`` 到中心）；给定 **F ∈ [f_lo, f_hi]**（会先排序并裁剪到 ``[0,1]``），条带为
    ``y ∈ [(cy-b)+f_lo·b,\\; (cy-b)+f_hi·b]``。

    在下带，**F(y) = ((cy+b) - y) / b**（``F=0`` 下外顶）；**F ∈ [f_lo, f_hi]** 对应
    ``y ∈ [(cy+b)-f_hi·b,\\; (cy+b)-f_lo·b]``。

    Parameters
    ----------
    top
        ``True`` 为上带，``False`` 为下带（由 CLI 对 ``--y-norm-range`` 两数正负判定）。
    """
    _cx, cy, _a, b = boundary
    h = int(image_height)
    bsafe = max(float(b), 1e-9)
    fa = float(min(f_lo, f_hi))
    fb = float(max(f_lo, f_hi))
    fa = float(np.clip(fa, 0.0, 1.0))
    fb = float(np.clip(fb, 0.0, 1.0))
    if top:
        y_a = (cy - bsafe) + fa * bsafe
        y_b = (cy - bsafe) + fb * bsafe
    else:
        y_a = (cy + bsafe) - fb * bsafe
        y_b = (cy + bsafe) - fa * bsafe
    y_lo = float(min(y_a, y_b))
    y_hi = float(max(y_a, y_b))
    y_lo = float(np.clip(y_lo, 0, max(0, h - 1)))
    y_hi = float(np.clip(y_hi, 0, max(0, h - 1)))
    return y_lo, y_hi


def rect_strip_column_profiles(
    image: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    *,
    col_metric: str = "mean",
) -> dict:
    """矩形条带内按列聚合，得到全长 ``W`` 的 1D 曲线（与 ``outer_y_edge_column_profiles`` 统计口径一致）。

    条带外列的 ``col_counts`` 为 0，``mean``/``max`` 模式下该列剖面为 0 或 NaN。
    """
    if col_metric not in ("sum", "mean", "max"):
        raise ValueError(f"col_metric must be 'sum', 'mean', or 'max', got {col_metric!r}")

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    mask, rmeta = axis_aligned_rect_mask((h, w), x_lo, x_hi, y_lo, y_hi)
    col_sum = np.sum(img * mask, axis=0)
    col_cnt = np.sum(mask, axis=0).astype(np.float64)

    if col_metric == "sum":
        prof = col_sum
    elif col_metric == "mean":
        prof = np.divide(
            col_sum, col_cnt, out=np.zeros_like(col_sum), where=col_cnt > 0.0
        )
    else:
        t_big = np.where(mask, img, -np.inf)
        prof = np.max(t_big, axis=0)
        prof = np.where(col_cnt > 0, prof, np.nan)

    meta = {**rmeta, "col_metric": col_metric}
    x_grid = np.arange(w, dtype=np.float64)
    imax = _profile_argmax_index(prof)
    peak_x, peak_val = _parabola_peak_refine(prof, imax)

    return {
        "meta": meta,
        "mask": mask,
        "x": x_grid,
        "profile": prof,
        "col_counts": col_cnt,
        "peak_ix": imax,
        "peak_x": peak_x,
        "peak_value": peak_val,
    }


def rect_strip_profile_for_peak_search(
    result: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """与 ``masked_strip_profiles_for_plot`` 同理：mean/max 在无掩膜列置 NaN 供峰搜索。"""
    cm = result["meta"].get("col_metric", "mean")
    prof = np.asarray(result["profile"], dtype=np.float64).copy()
    cnt = result.get("col_counts")
    if cm in ("mean", "max") and cnt is not None:
        prof[cnt <= 0] = np.nan
    return prof, cnt


def fitted_xy_for_rect_strip_peaks(
    strip_map: np.ndarray,
    result: dict,
    *,
    peak_dist: float,
    add_neighbor_x: bool = False,
    center_mode: str = "com",
    double_peak_fit: bool = False,
    prominence_min: float | None = None,
) -> list[tuple[float, float]]:
    """对每个 x 向辅助峰列算 y 中心，返回 ``(x_px, y_row)``（与 ``fitted_xy_for_auxiliary_strip_peaks`` 单条带分支相同）。"""
    x = result["x"]
    plot_prof, cnt = rect_strip_profile_for_peak_search(result)
    peaks = strip_profile_peaks_ixy(x, plot_prof, cnt, peak_dist)
    mask = result["mask"]
    out: list[tuple[float, float]] = []
    for col_ix, xp, _py in peaks:
        y_row, vals = column_y_profile_in_strip(
            strip_map, mask, col_ix, add_neighbor_x=add_neighbor_x,
        )
        if center_mode == "com":
            y_c = y_center_of_mass_from_profile(y_row, vals)
        elif center_mode == "com_fit":
            y_c, _y_com = y_marked_com_nearest_local_peak(y_row, vals)
        else:
            y_c, _popt, _info, _ypr = y_center_from_y_profile_fit(
                y_row,
                vals,
                double_peak_fit=double_peak_fit,
                prominence_min=prominence_min,
            )
        if y_c is None:
            continue
        out.append((float(xp), float(y_c)))
    return out
