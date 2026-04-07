"""
离子晶格图像中单离子的检测与椭圆拟合。

子模块
--------
gaussian, boundary, preprocess, fitting, peel — 算法组件
binarize — 减背景图二值化
blob_preprocess / blob_components / blob_workflow / blob_viz — 二值连通域与最小外接矩形工作流
pipeline — ``detect_ions`` 端到端流程
viz — 可视化与摘要
cli_helpers — 批处理索引解析

算法流程见 ``pipeline.detect_ions`` 文档字符串。
"""
from .binarize import bgsub_binarize, bgsub_binarize_u8
from .blob_components import (
    MinAreaRect,
    axis_aligned_bounding_rect_xy,
    binarize_foreground,
    drop_rects_both_axis_spans_at_most,
    label_connected_components,
    rects_from_labeled,
)
from .blob_edge_rect_merge import merge_edge_band_sliver_rects
from .blob_ion_positions import ion_equilibrium_positions_xy
from .blob_preprocess import BlobPreprocessResult, map_for_binarize, preprocess_for_blob_analysis
from .blob_workflow import BlobWorkflowResult, run_blob_workflow
from .edge_strip import outer_y_edge_column_profiles, outer_y_edge_strip_masks
from .blob_viz import visualize_blob_workflow, visualize_blob_rects
from .pipeline import detect_ions
from .viz import print_summary, visualize, visualize_bgsub, visualize_bgsub_binarized

__all__ = [
    "BlobPreprocessResult",
    "BlobWorkflowResult",
    "MinAreaRect",
    "axis_aligned_bounding_rect_xy",
    "bgsub_binarize",
    "bgsub_binarize_u8",
    "binarize_foreground",
    "detect_ions",
    "drop_rects_both_axis_spans_at_most",
    "label_connected_components",
    "merge_edge_band_sliver_rects",
    "ion_equilibrium_positions_xy",
    "map_for_binarize",
    "preprocess_for_blob_analysis",
    "rects_from_labeled",
    "run_blob_workflow",
    "visualize_blob_rects",
    "visualize_blob_workflow",
    "visualize",
    "visualize_bgsub",
    "visualize_bgsub_binarized",
    "print_summary",
    "outer_y_edge_column_profiles",
    "outer_y_edge_strip_masks",
]
