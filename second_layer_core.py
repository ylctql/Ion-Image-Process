"""
Second-layer ion detection: y-histogram peak row → three-row x profile → column COM.

Shared by ``second_layer_ion_peaks.py`` and ``merge_ion_centers.py`` (optional slab replace).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def _peak_indices_with_padded_ends(
    signal: np.ndarray,
    *,
    prominence: float,
    distance: int | None = None,
) -> np.ndarray:
    """``find_peaks`` does not treat endpoints as maxima; pad with zeros so first/last bins are detectable."""
    s = np.asarray(signal, dtype=np.float64).ravel()
    if s.size == 0:
        return np.array([], dtype=np.int64)
    padded = np.concatenate([[0.0], s, [0.0]])
    kw: dict[str, Any] = {"prominence": float(prominence)}
    if distance is not None:
        kw["distance"] = max(1, int(distance))
    ixp, _ = find_peaks(padded, **kw)
    ix = ixp.astype(np.int64) - 1
    return ix[(ix >= 0) & (ix < s.size)]


def second_histogram_peak_y_row(
    y_arr: np.ndarray,
    bin_edges: np.ndarray,
    hist_prominence: float,
    line_id: int,
) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
    """``line_id``-th peak when sorting histogram peaks by y (1-based). Returns y0, k, y_a, y_b, counts, peak_ix."""
    if line_id < 1:
        raise ValueError("line_id must be >= 1")
    counts, _ = np.histogram(y_arr, bins=bin_edges)
    peak_ix = _peak_indices_with_padded_ends(
        counts, prominence=float(hist_prominence),
    )
    if peak_ix.size < line_id:
        raise RuntimeError(
            f"y histogram: {peak_ix.size} peaks (prominence>{hist_prominence:g}); "
            f"need >= {line_id} for line_id={line_id}",
        )
    peak_y_centers = 0.5 * (bin_edges[peak_ix] + bin_edges[peak_ix + 1])
    sort_order = np.argsort(peak_y_centers)
    k = int(peak_ix[sort_order][line_id - 1])
    y_a, y_b = float(bin_edges[k]), float(bin_edges[k + 1])
    y0 = int(round(0.5 * (y_a + y_b)))
    return y0, k, y_a, y_b, counts, peak_ix


def three_row_sum_profile(
    im: np.ndarray,
    yr0: int,
    x_lo: int,
    x_hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = im.shape
    yr = int(np.clip(yr0, 0, h - 1))
    x_lo = max(0, int(x_lo))
    x_hi = min(w - 1, int(x_hi))
    xs = np.arange(x_lo, x_hi + 1, dtype=np.int64)
    y_top = max(0, yr - 1)
    y_bot = min(h, yr + 2)
    prof = np.sum(im[y_top:y_bot, xs], axis=0).astype(np.float64)
    return xs, prof


def com_y_column(
    im: np.ndarray,
    x_px: int,
    yr0: int,
    halfwin: int,
    *,
    neighbor_cols: int = 0,
) -> tuple[float, int, int, int, int]:
    """y 向亮度质心：在 y∈[yr0−halfwin, yr0+halfwin]、x∈[x_px−n, x_px+n]（截断到图像内）上求 COM。

    返回 ``(y_com, r0, r1, x_lo, x_hi)``；``r0,r1`` 为 y 半窗对应的行切片 ``[r0,r1)``，
    ``x_lo,x_hi`` 为参与求和的列下标（闭区间）。
    """
    h, w = im.shape
    x_px = int(np.clip(x_px, 0, w - 1))
    n = max(0, int(neighbor_cols))
    x_lo = max(0, x_px - n)
    x_hi = min(w - 1, x_px + n)
    r0 = max(0, yr0 - halfwin)
    r1 = min(h, yr0 + halfwin + 1)
    patch = np.maximum(im[r0:r1, x_lo : x_hi + 1].astype(np.float64), 0.0)
    s = float(np.sum(patch))
    if s <= 0.0:
        return float("nan"), r0, r1, x_lo, x_hi
    y_idx = np.arange(r0, r1, dtype=np.float64)[:, np.newaxis]
    y_com = float(np.sum(y_idx * patch) / s)
    return y_com, r0, r1, x_lo, x_hi


def ions_from_second_layer_row(
    image: np.ndarray,
    y0: int,
    px_lo: float,
    px_hi: float,
    *,
    halfwin: int,
    prof_prominence_frac: float,
    prof_peak_distance: int,
    source: str,
    com_neighbor_cols: int = 0,
) -> list[dict[str, Any]]:
    """One horizontal layer: sum I(y0-1:y0+1,x), find x peaks, COM y in ±halfwin rows × (x±com_neighbor_cols) cols."""
    h, w = image.shape
    im = np.asarray(image, dtype=np.float64)
    yr = int(np.clip(int(y0), 0, h - 1))
    a, b = (float(px_lo), float(px_hi)) if px_lo <= px_hi else (float(px_hi), float(px_lo))
    xlo = max(0, int(np.floor(a)))
    xhi = min(w - 1, int(np.ceil(b)))
    xs, prof = three_row_sum_profile(im, yr, xlo, xhi)
    if prof.size == 0:
        return []
    pmax = float(np.max(prof))
    prom = max(pmax * float(prof_prominence_frac), 1e-9)
    dist_px = max(1, int(prof_peak_distance))
    peaks_ix = _peak_indices_with_padded_ends(
        prof, prominence=prom, distance=dist_px,
    )
    out: list[dict[str, Any]] = []
    for ix in peaks_ix:
        x_px = int(xs[ix])
        y_com, _r0, _r1, _xl, _xh = com_y_column(
            im, x_px, yr, int(halfwin), neighbor_cols=int(com_neighbor_cols),
        )
        if not np.isfinite(y_com):
            continue
        out.append({"x0": float(x_px), "y0": float(y_com), "source": source})
    return out


def second_layer_y0_pair_and_slab_hi_mid23(
    y_arr: np.ndarray,
    bin_edges: np.ndarray,
    hist_prominence: float,
    line_id_first: int,
    line_id_second: int,
    line_id_third: int,
    mid_margin_px: float,
) -> tuple[int, int, float, float, float, float]:
    """Histogram-driven slab upper bound between 2nd and 3rd y-histogram peaks.

    Returns ``y0_1``, ``y0_2`` (rows for three-line sum profiles), ``yc2``, ``yc3`` (bin centers),
    ``y_mid`` = (yc2+yc3)/2, ``y_replace_hi`` = max(0, y_mid - mid_margin_px). Merge points in the
    x slab with ``y <= y_replace_hi`` are replaced by second-layer L1/L2; third-peak ions (larger y)
    stay as merge if they lie below the midpoint band.
    """
    for lid in (line_id_first, line_id_second, line_id_third):
        if lid < 1:
            raise ValueError("line_id_* must be >= 1")
    n_need = max(line_id_first, line_id_second, line_id_third)
    counts, _ = np.histogram(y_arr, bins=bin_edges)
    peak_ix = _peak_indices_with_padded_ends(
        counts, prominence=float(hist_prominence),
    )
    if peak_ix.size < n_need:
        raise RuntimeError(
            f"y histogram: {peak_ix.size} peaks (prominence>{hist_prominence:g}); need >= {n_need}",
        )
    peak_centers = 0.5 * (bin_edges[peak_ix] + bin_edges[peak_ix + 1])
    order = np.argsort(peak_centers)
    sorted_c = peak_centers[order].astype(np.float64)

    y0_1 = int(round(float(sorted_c[line_id_first - 1])))
    y0_2 = int(round(float(sorted_c[line_id_second - 1])))
    yc2 = float(sorted_c[line_id_second - 1])
    yc3 = float(sorted_c[line_id_third - 1])
    y_mid = 0.5 * (yc2 + yc3)
    y_hi = float(y_mid) - float(mid_margin_px)
    if y_hi < 0.0:
        y_hi = 0.0
    return y0_1, y0_2, yc2, yc3, float(y_mid), y_hi


def replace_merge_in_xy_slab(
    merged: list[dict[str, Any]],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    replacement: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop centers inside [x_lo,x_hi]×[y_lo,y_hi]; append replacement; sort by (y0,x0)."""
    xa, xb = (x_lo, x_hi) if x_lo <= x_hi else (x_hi, x_lo)
    kept = [
        p
        for p in merged
        if not (xa <= float(p["x0"]) <= xb and y_lo <= float(p["y0"]) <= y_hi)
    ]
    kept.extend(replacement)
    kept.sort(key=lambda d: (float(d["y0"]), float(d["x0"])))
    return kept
