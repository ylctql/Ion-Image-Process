"""二值连通域 + 最小外接矩形工作流编排。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .blob_components import MinAreaRect, binarize_foreground, label_connected_components, rects_from_labeled
from .blob_preprocess import BlobPreprocessResult, map_for_binarize, preprocess_for_blob_analysis


@dataclass(frozen=True)
class BlobWorkflowResult:
    preprocess: BlobPreprocessResult
    binarize_source: str
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
    binarize_on: Literal["denoised_map", "signal"] = "denoised_map",
    ge: bool = True,
    connectivity: Literal[4, 8] = 8,
    min_area_pixels: int = 1,
) -> BlobWorkflowResult:
    """
    默认仅减背景（不做匹配滤波）→ 估计 boundary → 阈值二值化 → 连通域 → 每域轴对齐最小外接矩形。
    可将 ``use_matched_filter=True`` 以与 ``detect_ions`` 的 ``detect_map`` 一致。

    Parameters
    ----------
    use_bgsub
        为 True（默认）时做高斯减背景；为 False 时后续均在原始 ``image`` 上计算。
    binarize_on
        ``\"denoised_map\"``：在 ``preprocess.denoised_map`` 上阈值（无匹配滤波时等于 ``signal``）；
        ``\"signal\"``：仅在 ``signal`` 上阈值（已减背景时为 bgsub，否则为原图浮点）。
    """
    pre = preprocess_for_blob_analysis(
        image,
        bg_sigma=bg_sigma,
        use_bgsub=use_bgsub,
        use_matched_filter=use_matched_filter,
    )
    src = map_for_binarize(pre, source=binarize_on)
    binary = binarize_foreground(src, threshold, ge=ge)
    labeled, n = label_connected_components(binary, connectivity=connectivity)
    rects = rects_from_labeled(labeled, min_area_pixels=min_area_pixels)
    return BlobWorkflowResult(
        preprocess=pre,
        binarize_source=binarize_on,
        threshold=float(threshold),
        binary=binary,
        labeled=labeled,
        n_components=n,
        rects=rects,
    )
