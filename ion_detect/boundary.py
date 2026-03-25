"""晶格边界估计与候选过滤。"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion


def estimate_crystal_boundary(signal, smooth_sigma=8,
                              mask_threshold_pct=0.15, margin=1.05):
    """从信号图估计离子晶格的椭圆边界。

    对背景减除后的信号做平滑, 取 15% 阈值确定晶格区域,
    从区域边界拟合椭圆, 加 5% 余量作为候选过滤边界。

    Returns
    -------
    (cx, cy, a, b) 或 None
        cx, cy – 椭圆中心 (像素)
        a      – x 方向半轴
        b      – y 方向半轴
    """
    smoothed = gaussian_filter(np.clip(signal, 0, None), sigma=smooth_sigma)
    mask = smoothed > mask_threshold_pct * smoothed.max()
    edge = mask & ~binary_erosion(mask, iterations=2)
    by, bx = np.where(edge)

    if len(bx) < 10:
        return None

    cx = (bx.max() + bx.min()) / 2.0
    cy = (by.max() + by.min()) / 2.0
    a = (bx.max() - bx.min()) / 2.0 * margin
    b = (by.max() - by.min()) / 2.0 * margin
    return cx, cy, a, b


def apply_boundary_filter(peak_yx, cx, cy, a, b):
    """保留椭圆边界内的候选峰, 移除外部噪声。"""
    px = peak_yx[:, 1].astype(np.float64)
    py = peak_yx[:, 0].astype(np.float64)
    inside = ((px - cx) / a)**2 + ((py - cy) / b)**2 <= 1.0
    return peak_yx[inside]
