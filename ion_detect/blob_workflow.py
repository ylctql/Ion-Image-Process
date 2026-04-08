"""二值连通域 + 最小外接矩形工作流编排。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .blob_components import (
    MinAreaRect,
    binarize_foreground,
    drop_rects_both_axis_spans_at_most,
    label_connected_components,
    rects_from_labeled,
)
from .blob_edge_rect_merge import merge_edge_band_sliver_rects
from .blob_preprocess import BlobPreprocessResult, preprocess_for_blob_analysis


@dataclass(frozen=True)
class BlobWorkflowResult:
    preprocess: BlobPreprocessResult
    threshold: float
    binary: np.ndarray
    labeled: np.ndarray
    n_components: int
    rects: list[MinAreaRect]
    n_edge_sliver_merges: int = 0
    n_rects_dropped_pre_merge: int = 0
    # 连通域 → rects_from_labeled 之后、pre-merge drop / edge merge 之前
    n_rects_after_labeling: int = 0


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
    merge_small_rects: bool = True,
    y_edge_frac: float = 0.3,
    min_edge_ysize: float = 5.0,
    merge_band_clip_ellipse: bool = True,
    pre_merge_drop_max_span: float | None = 1.0,
) -> BlobWorkflowResult:
    """
    可选减高斯背景 → 可选匹配滤波得 ``denoised_map`` → 估计 boundary → 在 ``denoised_map`` 上阈值二值化
    → 连通域 → 每域轴对齐最小外接矩形。

    在条带 merge 之前，默认剔除 ``width`` 与 ``height`` 均 ≤ ``pre_merge_drop_max_span`` 的矩形
    （两向跨度都不超过 1 像素格意义下的外接盒，见 :func:`~ion_detect.blob_components.drop_rects_both_axis_spans_at_most`）。
    设 ``pre_merge_drop_max_span=None`` 可关闭该步。

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
    n_rects_after_labeling = len(rects)
    n_drop_pre = 0
    if pre_merge_drop_max_span is not None:
        rects, n_drop_pre = drop_rects_both_axis_spans_at_most(
            rects,
            float(pre_merge_drop_max_span),
        )
    n_edge = 0
    if (
        merge_small_rects
        and pre.boundary is not None
        and rects
    ):
        rects, n_edge = merge_edge_band_sliver_rects(
            rects,
            pre.boundary,
            src.shape,
            y_edge_frac=float(y_edge_frac),
            min_edge_ysize=float(min_edge_ysize),
            clip_ellipse=merge_band_clip_ellipse,
        )
    return BlobWorkflowResult(
        preprocess=pre,
        threshold=float(threshold),
        binary=binary,
        labeled=labeled,
        n_components=n,
        rects=rects,
        n_edge_sliver_merges=n_edge,
        n_rects_dropped_pre_merge=n_drop_pre,
        n_rects_after_labeling=n_rects_after_labeling,
    )
