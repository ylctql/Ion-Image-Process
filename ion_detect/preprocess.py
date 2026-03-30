"""背景/匹配滤波与峰候选提取。"""
import numpy as np
from scipy.ndimage import maximum_filter

from .boundary import apply_boundary_filter


def build_matched_kernel(sigma_x=1.2, sigma_y=1.8, half_size=3):
    """构造与离子 PSF 匹配的高斯核，用于增强信噪比。"""
    ky, kx = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    kernel = np.exp(-0.5 * (kx**2 / sigma_x**2 + ky**2 / sigma_y**2))
    kernel /= kernel.sum()
    return kernel


def peaks_from_detect_map(
    detect_map: np.ndarray,
    peak_size,
    rel_threshold: float,
    boundary,
):
    """在检测图 detect_map 上做局部极大 + 阈值, 返回 peak_yx (N,2) [row, col]。"""
    local_max = maximum_filter(detect_map, size=peak_size)
    thresh = rel_threshold * detect_map.max()
    peak_mask = (detect_map == local_max) & (detect_map > thresh)

    peak_yx = np.argwhere(peak_mask)
    if boundary is not None:
        peak_yx = apply_boundary_filter(peak_yx, *boundary)
    return peak_yx
