"""条带 1D 列轮廓上的辅助峰检测、掩膜列 profile 与一维高斯拟合（供 CLI / 可视化复用）。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import peak_prominences

from .edge_strip import outer_y_edge_strip_masks


def y_profile_for_peak_prominence(y_prof: np.ndarray) -> np.ndarray:
    """Replace NaN so ``peak_prominences`` can use the full length of the profile."""
    y = np.asarray(y_prof, dtype=np.float64)
    if np.all(np.isfinite(y)):
        return y
    y2 = y.copy()
    lo = np.nanmin(y2)
    fill = float(lo) if np.isfinite(lo) else 0.0
    y2[~np.isfinite(y2)] = fill
    return y2


def strip_profile_local_maxima_ixy(
    x_grid: np.ndarray,
    y_prof: np.ndarray,
    col_counts: np.ndarray | None,
) -> list[tuple[int, float, float]]:
    """Strict 1D local maxima (i..n-2): y[i] > y[i-1] and y[i] > y[i+1].

    Skips columns with non-finite values or col_counts[i] <= 0 when counts are given.
    Returns ``(sample_index, x, y)`` sorted by x.
    """
    xg = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_prof, dtype=np.float64)
    n = int(y.size)
    cnt = np.asarray(col_counts, dtype=np.float64) if col_counts is not None else None
    out: list[tuple[int, float, float]] = []
    for i in range(1, n - 1):
        if cnt is not None and cnt[i] <= 0.0:
            continue
        a, b, c = y[i - 1], y[i], y[i + 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        if b > a and b > c:
            out.append((i, float(xg[i]), float(b)))
    return sorted(out, key=lambda t: t[1])


def strip_profile_peaks_min_distance_ixy(
    peaks_ixy: list[tuple[int, float, float]],
    y_fin: np.ndarray,
    peak_dist: float,
) -> list[tuple[int, float, float]]:
    """Require spacing > peak_dist between adjacent kept peaks (in x).

    If two peaks violate that, drop the one with lower `scipy.signal.peak_prominences`
    (recomputed on the current survivor set each time). Tie: higher y.
    ``peak_dist <= 0`` skips filtering.
    """
    if peak_dist <= 0.0 or len(peaks_ixy) <= 1:
        return sorted(peaks_ixy, key=lambda t: t[1])

    pts = sorted(peaks_ixy, key=lambda t: t[1])
    y_fin = np.asarray(y_fin, dtype=np.float64)
    while True:
        j = -1
        for i in range(len(pts) - 1):
            if pts[i + 1][1] - pts[i][1] <= peak_dist:
                j = i
                break
        if j < 0:
            break
        k0, k1 = j, j + 1
        ix_arr = np.array([p[0] for p in pts], dtype=np.intp)
        prom, _, _ = peak_prominences(y_fin, ix_arr, wlen=None)
        p0 = float(prom[k0])
        p1 = float(prom[k1])
        _ix0, _x0, y0 = pts[k0]
        _ix1, _x1, y1 = pts[k1]
        if p0 > p1:
            del pts[k1]
        elif p1 > p0:
            del pts[k0]
        elif y0 >= y1:
            del pts[k1]
        else:
            del pts[k0]
    return pts


def strip_profile_peaks_ixy(
    x_grid: np.ndarray,
    y_prof: np.ndarray,
    col_counts: np.ndarray | None,
    peak_dist: float,
) -> list[tuple[int, float, float]]:
    peaks = strip_profile_local_maxima_ixy(x_grid, y_prof, col_counts)
    y_fin = y_profile_for_peak_prominence(y_prof)
    return strip_profile_peaks_min_distance_ixy(peaks, y_fin, peak_dist)


def strip_profile_peak_xs(
    x_grid: np.ndarray,
    y_prof: np.ndarray,
    col_counts: np.ndarray | None,
    peak_dist: float,
) -> list[float]:
    return [p[1] for p in strip_profile_peaks_ixy(x_grid, y_prof, col_counts, peak_dist)]


def masked_strip_profiles_for_plot(
    result: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Same masking as strip plots: NaN where column has no mask pixels for mean/max."""
    cm = result["meta"].get("col_metric", "mean")
    top_p = result["top_profile"]
    bot_p = result["bot_profile"]
    tc = result.get("top_col_counts")
    bc = result.get("bot_col_counts")
    if cm in ("mean", "max"):
        top_plot = np.asarray(top_p, dtype=np.float64).copy()
        bot_plot = np.asarray(bot_p, dtype=np.float64).copy()
        if tc is not None:
            top_plot[tc <= 0] = np.nan
        if bc is not None:
            bot_plot[bc <= 0] = np.nan
    else:
        top_plot = np.asarray(top_p, dtype=np.float64)
        bot_plot = np.asarray(bot_p, dtype=np.float64)
    return top_plot, bot_plot, tc, bc


def column_y_profile_in_strip(
    img: np.ndarray,
    mask_strip: np.ndarray,
    col_ix: int,
    *,
    add_neighbor_x: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Within one strip mask, return (y pixel index, pixel values) for column ``col_ix``.

    If ``add_neighbor_x`` is True, each value is ``I[y,x-1]+I[y,x]+I[y,x+1]`` (在图像边界处
    缺失的邻列不列入求和)，行集合仍由中心列 ``col_ix`` 在 ``mask_strip`` 内的 True 决定。
    """
    w = int(img.shape[1])
    col_ix = int(np.clip(col_ix, 0, max(0, w - 1)))
    rows = np.flatnonzero(mask_strip[:, col_ix])
    if rows.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    if not add_neighbor_x:
        vals = img[rows, col_ix]
        return rows.astype(np.float64), vals
    vals = img[rows, col_ix].copy()
    if col_ix - 1 >= 0:
        vals += img[rows, col_ix - 1]
    if col_ix + 1 < w:
        vals += img[rows, col_ix + 1]
    return rows.astype(np.float64), vals


def y_center_of_mass_from_profile(y_row: np.ndarray, vals: np.ndarray) -> float | None:
    """1D center of mass ``sum(y * w) / sum(w)`` with weights ``w = max(intensity, 0)``.

    If ``sum(max(I,0)) == 0``, use ``w = max(I - min(I), 0)`` on finite samples (handles negative baselines).
    Returns None when no positive total weight.
    """
    m = np.isfinite(y_row) & np.isfinite(vals)
    yv = np.asarray(y_row[m], dtype=np.float64)
    vv = np.asarray(vals[m], dtype=np.float64)
    if yv.size < 1:
        return None
    w = np.maximum(vv, 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0:
        off = float(np.min(vv))
        w = np.maximum(vv - off, 0.0)
        sw = float(np.sum(w))
        if sw <= 0.0:
            return None
    return float(np.sum(yv * w) / sw)


def y_marked_com_nearest_local_peak(
    y_row: np.ndarray,
    vals: np.ndarray,
) -> tuple[float | None, float | None]:
    """先算亮度加权 COM，再在沿 y 递增的 profile 上找严格局部极大（邻域两点更弱），取与 COM 最近的峰 y 作为标记中心。

    若无局部极大（点数不足或无上凸），退回 COM。返回 ``(y_marked, y_com)``；COM 不可算时 ``(None, None)``。
    """
    y_com = y_center_of_mass_from_profile(y_row, vals)
    if y_com is None:
        return None, None
    m = np.isfinite(y_row) & np.isfinite(vals)
    yv = np.asarray(y_row[m], dtype=np.float64)
    vv = np.asarray(vals[m], dtype=np.float64)
    if yv.size < 3:
        return float(y_com), float(y_com)
    order = np.argsort(yv)
    yv, vv = yv[order], vv[order]
    peak_ys: list[float] = []
    n = int(yv.size)
    for i in range(1, n - 1):
        if vv[i] > vv[i - 1] and vv[i] > vv[i + 1]:
            peak_ys.append(float(yv[i]))
    if not peak_ys:
        return float(y_com), float(y_com)
    y_com_f = float(y_com)
    y_marked = min(peak_ys, key=lambda yp: abs(yp - y_com_f))
    return y_marked, float(y_com)


def gaussian_1d_profile(y: np.ndarray, c0: float, amp: float, mu: float, sigma: float) -> np.ndarray:
    return c0 + amp * np.exp(-0.5 * ((y - mu) / sigma) ** 2)


def two_gaussian_1d_profile(
    y: np.ndarray,
    c0: float,
    a1: float,
    mu1: float,
    s1: float,
    a2: float,
    mu2: float,
    s2: float,
) -> np.ndarray:
    return (
        c0
        + a1 * np.exp(-0.5 * ((y - mu1) / s1) ** 2)
        + a2 * np.exp(-0.5 * ((y - mu2) / s2) ** 2)
    )


def fit_y_profile_gaussian(y: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray | None, str]:
    m = np.isfinite(y) & np.isfinite(vals)
    yv = np.asarray(y[m], dtype=np.float64)
    vv = np.asarray(vals[m], dtype=np.float64)
    if yv.size < 4:
        return None, "need >= 4 points"
    order = np.argsort(yv)
    yv, vv = yv[order], vv[order]
    c0 = float(np.percentile(vv, 10))
    amp = float(np.max(vv) - c0)
    if not np.isfinite(amp) or amp == 0.0:
        amp = float(np.max(vv))
    mu = float(yv[np.argmax(vv)])
    span = float(max(yv[-1] - yv[0], 1.0))
    sigma = float(max(span * 0.15, 0.5))
    p0 = np.array([c0, amp, mu, sigma], dtype=np.float64)
    lo = np.array([-np.inf, -np.inf, float(np.min(yv)), 1e-6], dtype=np.float64)
    hi = np.array([np.inf, np.inf, float(np.max(yv)), np.inf], dtype=np.float64)
    try:
        popt, _ = curve_fit(
            gaussian_1d_profile, yv, vv, p0=p0, bounds=(lo, hi), maxfev=20000,
        )
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)
    return popt, ""


def fit_y_profile_double_gaussian(y: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray | None, str]:
    """两峰高斯 + 公共偏置；点数过少或拟合失败返回 None。"""
    m = np.isfinite(y) & np.isfinite(vals)
    yv = np.asarray(y[m], dtype=np.float64)
    vv = np.asarray(vals[m], dtype=np.float64)
    if yv.size < 10:
        return None, "need >= 10 points for double Gaussian"
    order = np.argsort(yv)
    yv, vv = yv[order], vv[order]
    ymin, ymax = float(np.min(yv)), float(np.max(yv))
    n = int(yv.size)
    mid = max(2, n // 2)
    i1 = int(np.argmax(vv[:mid]))
    i2 = mid + int(np.argmax(vv[mid:]))
    mu1, mu2 = float(yv[i1]), float(yv[i2])
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
    span = float(max(ymax - ymin, 1.0))
    if abs(mu2 - mu1) < 1e-3 * span:
        mu1 = float(ymin + 0.22 * span)
        mu2 = float(ymin + 0.78 * span)
    c0 = float(np.percentile(vv, 8))
    a1 = float(max(vv[max(0, i1)] - c0, 1e-9))
    a2 = float(max(vv[min(n - 1, i2)] - c0, 1e-9))
    s_init = float(max(span * 0.12, 0.45))
    p0 = np.array([c0, a1, mu1, s_init, a2, mu2, s_init], dtype=np.float64)
    lo = np.array(
        [-np.inf, -np.inf, ymin, 1e-6, -np.inf, ymin, 1e-6],
        dtype=np.float64,
    )
    hi = np.array(
        [np.inf, np.inf, ymax, np.inf, np.inf, ymax, np.inf],
        dtype=np.float64,
    )
    try:
        popt, _ = curve_fit(
            two_gaussian_1d_profile, yv, vv, p0=p0, bounds=(lo, hi), maxfev=50000,
        )
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)
    return popt, ""


def y_center_from_profile_prominence_top2(
    y_row: np.ndarray,
    vals: np.ndarray,
    *,
    min_prominence: float,
) -> tuple[float | None, tuple[float, float] | None, str]:
    """在排序后的 y-profile 上找严格局部极大，计算 ``peak_prominence``，取 prominence 最大的两个峰 y 求平均。

    仅保留 ``prominence >= min_prominence`` 的候选；若过滤后只有一个，中心即该峰 y；
    若无候选则返回 None。若无严格局部极大则退化为全局最大索引作为单一候选。
    """
    m = np.isfinite(y_row) & np.isfinite(vals)
    yv = np.asarray(y_row[m], dtype=np.float64)
    vv = np.asarray(vals[m], dtype=np.float64)
    order = np.argsort(yv)
    yv, vv = yv[order], vv[order]
    n = int(yv.size)
    if n < 3:
        return None, None, "need >= 3 points"
    idx_candidates: list[int] = []
    for i in range(1, n - 1):
        if vv[i] > vv[i - 1] and vv[i] > vv[i + 1]:
            idx_candidates.append(i)
    if not idx_candidates:
        idx_candidates = [int(np.argmax(vv))]

    v_fin = y_profile_for_peak_prominence(vv)
    ix_arr = np.array(idx_candidates, dtype=np.intp)
    prom, _, _ = peak_prominences(v_fin, ix_arr, wlen=None)
    scored: list[tuple[int, float]] = []
    for j, ix in enumerate(idx_candidates):
        pj = float(prom[j])
        if np.isfinite(pj) and pj >= float(min_prominence):
            scored.append((ix, pj))
    if not scored:
        return None, None, "no candidate with prominence>=min"

    scored.sort(key=lambda t: t[1], reverse=True)
    if len(scored) == 1:
        i0 = scored[0][0]
        y0 = float(yv[i0])
        return y0, (y0, y0), "one_peak"

    i1, i2 = scored[0][0], scored[1][0]
    y1, y2 = float(yv[i1]), float(yv[i2])
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    return 0.5 * (y1 + y2), (ya, yb), "top2"


def y_center_from_y_profile_fit(
    y_row: np.ndarray,
    vals: np.ndarray,
    *,
    double_peak_fit: bool = False,
    prominence_min: float | None = None,
) -> tuple[float | None, np.ndarray | None, str, tuple[float, float] | None]:
    """返回 (y_center, popt_or_None, info, prominence_peak_ys_or_None)。

    若给定 ``prominence_min``（含 0），优先用 prominence 最大的两个局部峰的平均；失败则回退下方高斯逻辑。
    ``double_peak_fit``：双高斯两 ``mu`` 平均；失败回退单高斯。与 prominence 同时开启时 prominence 优先。
    """
    y_pair: tuple[float, float] | None = None
    if prominence_min is not None:
        yc, ypr, msg = y_center_from_profile_prominence_top2(
            y_row, vals, min_prominence=prominence_min,
        )
        if yc is not None:
            return yc, None, f"prominence:{msg}", ypr

    if not double_peak_fit:
        popt, err = fit_y_profile_gaussian(y_row, vals)
        if popt is None:
            return None, None, err, None
        return float(popt[2]), popt, "", None

    p2, err2 = fit_y_profile_double_gaussian(y_row, vals)
    if p2 is not None:
        mu_a, mu_b = float(p2[2]), float(p2[5])
        if mu_a > mu_b:
            mu_a, mu_b = mu_b, mu_a
        y_mid = 0.5 * (mu_a + mu_b)
        return y_mid, p2, "", (mu_a, mu_b)

    p1, err1 = fit_y_profile_gaussian(y_row, vals)
    if p1 is None:
        return None, None, f"double failed ({err2}); single: {err1}", None
    return float(p1[2]), p1, "single_fallback", None


def strip_profile_fit_masks(
    result: dict,
    boundary: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    *,
    clip_ellipse: bool,
    y_fit_frac: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """与 ``--peak-col-gallery`` / 列向拟合一致：用于 y-profile 采样的上下条带掩膜。"""
    meta = result["meta"]
    peak_f = float(meta["y_edge_frac"])
    if y_fit_frac is None or abs(float(y_fit_frac) - peak_f) <= 1e-9:
        return result["top_mask"], result["bot_mask"]
    top_m, bot_m, _ = outer_y_edge_strip_masks(
        boundary, float(y_fit_frac), image_shape, clip_ellipse=clip_ellipse,
    )
    return top_m, bot_m


def fitted_xy_for_auxiliary_strip_peaks(
    strip_map: np.ndarray,
    result: dict,
    boundary: tuple[float, float, float, float],
    *,
    peak_dist: float,
    clip_ellipse: bool,
    y_fit_frac: float | None,
    add_neighbor_x: bool = True,
    double_peak_fit: bool = False,
    prominence_min: float | None = None,
    center_mode: str = "fit",
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """对每个辅助条带峰所在列估计 y 中心，得到 ``(x_px, y_row)``；失败则跳过该峰。

    ``center_mode=="com"``：仅用 ``y_center_of_mass_from_profile``（亮度加权质心），忽略
    ``prominence_min`` / ``double_peak_fit``。
    ``center_mode=="com_fit"``：``y_marked_com_nearest_local_peak``（COM + 沿 y 邻域局部峰，取距 COM 最近者）。
    否则 ``prominence_min`` 非 None 时优先 prominence 最大两个局部峰取平均；再按 ``double_peak_fit`` 用双/单高斯。
    """
    x = result["x"]
    top_plot, bot_plot, tc, bc = masked_strip_profiles_for_plot(result)
    top_peaks = strip_profile_peaks_ixy(x, top_plot, tc, peak_dist)
    bot_peaks = strip_profile_peaks_ixy(x, bot_plot, bc, peak_dist)
    h, w = int(strip_map.shape[0]), int(strip_map.shape[1])
    top_m, bot_m = strip_profile_fit_masks(
        result, boundary, (h, w), clip_ellipse=clip_ellipse, y_fit_frac=y_fit_frac,
    )

    def collect(
        peaks: list[tuple[int, float, float]],
        mask: np.ndarray,
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for col_ix, xp, _strip_y in peaks:
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

    return collect(top_peaks, top_m), collect(bot_peaks, bot_m)
