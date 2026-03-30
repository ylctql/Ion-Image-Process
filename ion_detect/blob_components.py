"""二值化、连通域标号与轴对齐最小外接矩形（边平行于 x、y 轴）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy.ndimage import label as nd_label

from .binarize import bgsub_binarize


def binarize_foreground(
    float_map: np.ndarray,
    threshold: float,
    *,
    ge: bool = True,
) -> np.ndarray:
    """对浮点图做阈值分割；规则同 :func:`ion_detect.binarize.bgsub_binarize`。"""
    return bgsub_binarize(float_map, threshold, ge=ge)


def label_connected_components(
    binary: np.ndarray,
    *,
    connectivity: Literal[4, 8] = 8,
) -> tuple[np.ndarray, int]:
    """
    对二值前景（True 为前景）做连通域标号。

    Returns
    -------
    labels : int64 (H, W), 0 为背景，1…n 为各连通域
    n : 连通域个数
    """
    m = np.asarray(binary, dtype=bool)
    if connectivity == 4:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        struct = np.ones((3, 3), dtype=int)
    labeled_arr, num_features = cast(
        tuple[np.ndarray, int],
        nd_label(m, structure=struct),
    )
    return labeled_arr.astype(np.int32), int(num_features)


@dataclass
class MinAreaRect:
    """轴对齐最小外接矩形（像素坐标，x 横、y 纵，与 ``imshow`` 一致）；``angle_deg`` 恒为 0。"""

    label: int
    center_xy: tuple[float, float]
    width: float
    height: float
    angle_deg: float
    corners_xy: np.ndarray
    """形状 (4, 2)，列为 ``(x, y)``，逆时针或顺时针闭合。"""
    area_pixels: int


def _axis_aligned_rect_xy(xy: np.ndarray) -> dict[str, Any]:
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    corners = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=np.float64,
    )
    return {
        "center": corners.mean(axis=0),
        "width": float(xmax - xmin),
        "height": float(ymax - ymin),
        "angle_deg": 0.0,
        "corners": corners,
    }


def axis_aligned_bounding_rect_xy(xy: np.ndarray) -> dict[str, Any] | None:
    """
    对点集 ``xy``（形状 ``(N,2)``，列 ``x,y``）求轴对齐外接盒（AABB）。

    单点退化为零宽高矩形（四角重合）。
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return None
    if xy.shape[0] == 1:
        c = xy[0]
        z = np.repeat(c[None, :], 4, axis=0)
        return {"center": c, "width": 0.0, "height": 0.0, "angle_deg": 0.0, "corners": z}
    return _axis_aligned_rect_xy(xy)


def rects_from_labeled(
    labeled: np.ndarray,
    *,
    min_area_pixels: int = 1,
) -> list[MinAreaRect]:
    """
    对每个连通域标签求轴对齐最小外接矩形（像素 ``(x,y)``）。

    ``min_area_pixels``：面积（像素数）小于该值的区域丢弃。
    """
    lab = np.asarray(labeled, dtype=np.int32)
    nlab = int(lab.max())
    out: list[MinAreaRect] = []
    for k in range(1, nlab + 1):
        ys, xs = np.where(lab == k)
        count = int(xs.size)
        if count < min_area_pixels:
            continue
        xy = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        geo = axis_aligned_bounding_rect_xy(xy)
        if geo is None:
            continue
        out.append(
            MinAreaRect(
                label=k,
                center_xy=(float(geo["center"][0]), float(geo["center"][1])),
                width=float(geo["width"]),
                height=float(geo["height"]),
                angle_deg=float(geo["angle_deg"]),
                corners_xy=np.asarray(geo["corners"], dtype=np.float64),
                area_pixels=count,
            ),
        )
    return out
