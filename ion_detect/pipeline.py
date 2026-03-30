"""端到端检测流程: 预处理 → 找峰 → 拟合。"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .boundary import estimate_crystal_boundary
from .fitting import fit_all_peaks
from .preprocess import build_matched_kernel, peaks_from_detect_map


def detect_ions(image, bg_sigma=(10, 30), peak_size=(5, 9),
                rel_threshold=0.025, fit_hw=(3, 4),
                sigma_range=(0.3, 3.5),
                refine=True, fix_theta_zero=True,
                return_bgsub=False):
    """
    检测离子并拟合椭圆参数。

    Parameters
    ----------
    image : ndarray (H, W), float
    bg_sigma : float or tuple – 背景估计的高斯 sigma (y, x)
    peak_size : int or tuple  – 局部极大值检测窗口尺寸 (y, x)
    rel_threshold : float – 信号阈值 (相对于最大信号)
    fit_hw : int or (int, int) – 高斯拟合的半窗口 (hw_y, hw_x)
    sigma_range : tuple – 允许的 sigma 范围 (min, max)
    refine : bool – 是否启用两阶段精修 (计数模式开启; 形变测量可关闭)
    fix_theta_zero : bool – 默认为 True, 高斯拟合固定转角 θ=0 (轴对齐 x/y)；仅代码中可改为 False 以拟合旋转椭圆
    return_bgsub : bool – 为 True 时额外返回首轮 ``image - GaussianBackground`` (与边界估计、
        拟合所用 ``signal`` 一致; 有符号, 可为负)。

    Returns
    -------
    ions : list[dict]  每个离子的参数:
        x0, y0       – 中心坐标 (像素, 亚像素精度)
        sigma_minor  – 短轴 sigma (像素)
        sigma_major  – 长轴 sigma (像素)
        theta_deg    – 长轴相对 x 轴的旋转角 (度)
        amplitude    – 高斯峰值强度

    默认 ``(ions, boundary)``；若 ``return_bgsub`` 为真则 ``(ions, boundary, bgsub)``。
    """
    img = image.astype(np.float64)
    h, w = img.shape
    s_lo, s_hi = sigma_range

    if isinstance(fit_hw, (list, tuple)):
        hw_y, hw_x = fit_hw
    else:
        hw_y = hw_x = fit_hw

    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg
    kernel = build_matched_kernel()
    detect_map = fftconvolve(signal, kernel, mode="same")

    boundary = estimate_crystal_boundary(signal)

    peak_yx = peaks_from_detect_map(
        detect_map,
        peak_size,
        rel_threshold,
        boundary,
    )

    ions = fit_all_peaks(
        img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, refine=refine, fix_theta_zero=fix_theta_zero,
    )
    ions.sort(key=lambda d: (d["y0"], d["x0"]))

    if return_bgsub:
        return ions, boundary, signal
    return ions, boundary
