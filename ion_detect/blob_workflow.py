"""二值连通域 + 最小外接矩形工作流编排。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .blob_components import MinAreaRect, binarize_foreground, label_connected_components, rects_from_labeled
from .blob_preprocess import BlobPreprocessResult, preprocess_for_blob_analysis


@dataclass(frozen=True)
class BlobWorkflowResult:
    preprocess: BlobPreprocessResult
    threshold: float
    binary: np.ndarray
    labeled: np.ndarray
    n_components: int
    rects: list[MinAreaRect]


def run_blob_workflow(
    image: np.ndarray,
    threshold: float,
    *,
    bg_sigma: float | tuple[float, ...] = (10, 30),
    use_bgsub: bool = True,
    use_matched_filter: bool = False,
    ge: bool = True,
    connectivity: Literal[4, 8] = 8,
    min_area_pixels: int = 1,
) -> BlobWorkflowResult:
    """
    可选减高斯背景 → 可选匹配滤波得 ``denoised_map`` → 估计 boundary → 在 ``denoised_map`` 上阈值二值化
    → 连通域 → 每域轴对齐最小外接矩形。

    ``use_matched_filter=True`` 时与 ``detect_ions`` 使用相同的匹配滤波核；否则 ``denoised_map`` 与
    ``signal``（减背景后或原图浮点）相同。
    """
    pre = preprocess_for_blob_analysis(
        image,
        bg_sigma=bg_sigma,
        use_bgsub=use_bgsub,
        use_matched_filter=use_matched_filter,
    )
    src = np.asarray(pre.denoised_map, dtype=np.float64)
    binary = binarize_foreground(src, threshold, ge=ge)
    labeled, n = label_connected_components(binary, connectivity=connectivity)
    rects = rects_from_labeled(labeled, min_area_pixels=min_area_pixels)
    return BlobWorkflowResult(
        preprocess=pre,
        threshold=float(threshold),
        binary=binary,
        labeled=labeled,
        n_components=n,
        rects=rects,
    )
