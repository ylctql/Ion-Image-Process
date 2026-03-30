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
    """用于 ``binarize_on='denoised_map'`` 的浮点图：开启匹配滤波时为滤波结果，否则与 ``signal`` 相同。"""
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
