"""
离子晶格图像中单离子的检测与椭圆拟合。

子模块
--------
gaussian, boundary, preprocess, fitting, peel — 算法组件
pipeline — ``detect_ions`` 端到端流程
viz — 可视化与摘要
cli_helpers — 批处理索引解析

算法流程见 ``pipeline.detect_ions`` 文档字符串。
"""
from .edge_strip import outer_y_edge_column_profiles, outer_y_edge_strip_masks
from .pipeline import detect_ions
from .viz import print_summary, visualize

__all__ = [
    "detect_ions",
    "visualize",
    "print_summary",
    "outer_y_edge_column_profiles",
    "outer_y_edge_strip_masks",
]
