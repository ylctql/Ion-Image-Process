"""滤波降噪与晶格 boundary：与 ``detect_ions`` 前段一致，供二值化 / 连通域工作流复用。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .boundary import estimate_crystal_boundary
from .preprocess import build_matched_kernel


@dataclass(frozen=True)
class BlobPreprocessResult:
    """``image`` 经可选减背景、可选匹配滤波后的量，以及由 **signal** 估计的 boundary。"""

    signal: np.ndarray
    """已减背景时同 ``detect_ions`` 首轮 ``signal``；未减背景时为 ``image`` 的浮点副本。"""
    denoised_map: np.ndarray
    """二值化所用的浮点图：开启匹配滤波时为滤波结果，否则与 ``signal`` 相同。"""
    boundary: tuple[float, float, float, float] | None
    """``(cx, cy, a, b)`` 椭圆；估计失败时为 ``None``。"""


def subtract_gaussian_background(
    image: np.ndarray,
    bg_sigma: float | tuple[float, ...] = (10, 30),
) -> np.ndarray:
    """``image - gaussian_filter(image, bg_sigma)``。"""
    img = np.asarray(image, dtype=np.float64)
    bg = gaussian_filter(img, sigma=bg_sigma)
    return img - bg


def matched_filter_same(signal: np.ndarray) -> np.ndarray:
    """与 ``detect_ions`` 相同的匹配滤波（``fftconvolve`` + ``build_matched_kernel``）。"""
    s = np.asarray(signal, dtype=np.float64)
    k = build_matched_kernel()
    return fftconvolve(s, k, mode="same")


def preprocess_for_blob_analysis(
    image: np.ndarray,
    *,
    bg_sigma: float | tuple[float, ...] = (10, 30),
    use_bgsub: bool = True,
    use_matched_filter: bool = False,
) -> BlobPreprocessResult:
    """
    可选减高斯背景 → 可选匹配滤波；boundary 始终在 **signal**（未卷积）上估计。

    Parameters
    ----------
    use_bgsub
        为 True 时 ``signal = image - GaussianBackground``；为 False 时 ``signal`` 为 ``image`` 的浮点数组。
    use_matched_filter
        默认 False（仅 bgsub）。为 True 时 ``denoised_map`` 为与 ``detect_ions`` 相同的匹配滤波图，否则 ``denoised_map is signal``。
    """
    img = np.asarray(image, dtype=np.float64)
    if use_bgsub:
        signal = subtract_gaussian_background(img, bg_sigma=bg_sigma)
    else:
        signal = img
    denoised = matched_filter_same(signal) if use_matched_filter else signal
    boundary = estimate_crystal_boundary(signal)
    return BlobPreprocessResult(
        signal=signal,
        denoised_map=np.asarray(denoised, dtype=np.float64),
        boundary=boundary,
    )


def ellipse_interior_mask(
    shape: tuple[int, int],
    boundary: tuple[float, float, float, float],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    与 ``blob_cli`` / ``blob_viz`` 椭圆 patch 一致：``(cx, cy, a, b)`` 为半轴，返回 (H, W) bool，True 为椭圆内。
    """
    h, w = int(shape[0]), int(shape[1])
    cx, cy, a, b = boundary
    yy, xx = np.indices((h, w), dtype=np.float64)
    aa = max(float(a), eps)
    bb = max(float(b), eps)
    t = ((xx - cx) / aa) ** 2 + ((yy - cy) / bb) ** 2
    return np.less_equal(t, 1.0 + eps)


def _thr_norm_scale_positive_percentile(
    z: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    eps: float,
) -> float:
    """在 ``mask`` 内用正值子集的 ``percentile`` 作尺度；无正值时用 ``|z|`` 分位数，避免减背景后尺度为负。"""
    vals = z[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    pct = float(np.clip(percentile, 1.0, 100.0))
    pos = vals[vals > 0]
    if pos.size > 0:
        s = float(np.percentile(pos, pct))
    else:
        s = float(np.percentile(np.abs(vals), pct))
    if not np.isfinite(s) or s <= eps:
        s = max(float(np.nanmax(np.abs(vals))), eps)
    return s


def _thr_norm_scale_roi_signed_percentile(
    z: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    eps: float,
) -> float:
    """
    在 ``mask`` 内 **全体有限像素** 的 **有符号** 取值上取 ``percentile``（从最小到最大排序后的分位点），
    **不是** ``|z|``。用于除法的 ``scale`` 必须为正；若该分位落在非正侧则回退为正值子集分位，再回退 ``|z|``。
    """
    vals = z[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    pct = float(np.clip(percentile, 1.0, 100.0))
    s = float(np.percentile(vals, pct))
    if not np.isfinite(s) or s <= eps:
        pos = vals[vals > 0]
        if pos.size > 0:
            s = float(np.percentile(pos, pct))
        else:
            s = float(np.percentile(np.abs(vals), pct))
        if not np.isfinite(s) or s <= eps:
            s = max(float(np.nanmax(np.abs(vals))), eps)
    return s


def denoised_map_thr_norm_p95(
    denoised_map: np.ndarray,
    boundary: tuple[float, float, float, float] | None,
    *,
    percentile: float = 95.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    每帧统一阈值用：在晶体椭圆内（无 boundary 时用全图有限像素）取正值鲁棒分位作尺度，
    ``z_norm = z / scale``。``--threshold`` 作用在 ``z_norm`` 上（例如 0.8 表示约原图 0.8×P95(+)）。
    """
    z = np.asarray(denoised_map, dtype=np.float64)
    if boundary is not None:
        m = ellipse_interior_mask(z.shape, boundary)
    else:
        m = np.isfinite(z)
    scale = _thr_norm_scale_positive_percentile(z, m, percentile=percentile, eps=eps)
    return (z / scale), float(scale)


def denoised_map_thr_norm_p95_all(
    denoised_map: np.ndarray,
    boundary: tuple[float, float, float, float] | None,
    *,
    percentile: float = 95.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    与 :func:`denoised_map_thr_norm_p95` 相同的几何掩膜，但尺度取 **ROI 内全体有限像素的有符号** ``P`` 分位
    （非绝对值）。``z_norm = z / scale``；建议 ``percentile`` 足够高（如 95）使分位落在正尾，否则将触发回退逻辑。
    """
    z = np.asarray(denoised_map, dtype=np.float64)
    if boundary is not None:
        m = ellipse_interior_mask(z.shape, boundary)
    else:
        m = np.isfinite(z)
    scale = _thr_norm_scale_roi_signed_percentile(z, m, percentile=percentile, eps=eps)
    return (z / scale), float(scale)


def map_for_binarize(
    pre: BlobPreprocessResult,
    *,
    source: str = "denoised_map",
) -> np.ndarray:
    """返回用于阈值化的数组：``\"denoised_map\"`` 或 ``\"signal\"``。"""
    if source == "denoised_map":
        return pre.denoised_map
    if source == "signal":
        return pre.signal
    raise ValueError(f"source must be 'denoised_map' or 'signal', got {source!r}")
