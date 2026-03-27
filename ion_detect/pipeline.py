"""端到端检测流程: 预处理 → 找峰 → 拟合 → 可选峰值剥离。"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .boundary import estimate_crystal_boundary
from .fitting import fit_all_peaks
from .gaussian import _accumulate_peel_model
from .peel import (
    filter_ions_y_edge_bands,
    filter_peak_yx_y_edge_bands,
    merge_ions_by_distance,
)
from .preprocess import build_matched_kernel, peaks_from_detect_map


def detect_ions(image, bg_sigma=(10, 30), peak_size=(5, 9),
                rel_threshold=0.025, fit_hw=(4, 3),
                sigma_range=(0.3, 3.5), use_matched_filter=True,
                refine=True, fix_theta_zero=False,
                use_y_threshold_comp=False,
                amp_y_coef=None, amp_y_coef_path=None,
                amp_y_coef_mode="even", comp_floor=0.2,
                joint_pair_y_gap=None, joint_pair_x_gap=None,
                peak_peel=False, peak_peel_min_sep=2.0,
                peak_peel_margin_sigma=4.5,
                peak_peel_y_edges_only=False, peak_peel_y_edge_frac=0.25,
                peak_peel_rel_threshold=None,
                peak_peel_min_amp_frac=None,
                return_peel_residual=False):
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
    use_matched_filter : bool – 是否用匹配滤波器增强峰检测
    refine : bool – 是否启用两阶段精修 (计数模式开启; 形变测量可关闭)
    fix_theta_zero : bool – 若为 True, 高斯拟合固定转角 θ=0 (轴对齐 x/y, 不拟合旋转)
    joint_pair_y_gap : float or None – 若给定, 对 |Δy|≤该值且 |Δx|≤joint_pair_x_gap 的近邻峰
        在合并 ROI 内做 N=2、θ=0 联合高斯拟合 (失败则回退为两次单峰拟合)。
    joint_pair_x_gap : float or None – 联合配对允许的 |Δx| (像素); 默认 max(4, hw_x)。
    peak_peel : bool – 为 True 时在首轮拟合后从原图减去各峰高斯核, 在残差上再检一轮并去重合并。
    peak_peel_min_sep : float – 第二轮新峰与已有峰中心的最小距离 (像素), 低于此视为重复。
    peak_peel_margin_sigma : float – 叠加剥离模型时每个峰在 ROI 上覆盖的半宽倍数 (相对 σ_major)。
    peak_peel_y_edges_only : bool – 为 True 时第二轮只在晶格边界椭圆的 y 向边缘带内取候选并保留结果,
        减轻中间区域因减峰不彻底产生的「傍大峰」假检测 (需 boundary 非 None 才生效)。
    peak_peel_y_edge_frac : float – y 向边缘带厚度, 0.25 表示仅 |y-cy|/b ≥ 0.75 的候选 (靠上下缘)。
    peak_peel_rel_threshold : float or None – 第二轮检测的相对阈值; None 则与首轮 rel_threshold 相同,
        可设为略高于首轮 (如 0.04) 以抑制残差上的小峰。
    peak_peel_min_amp_frac : float or None – 第二轮保留的离子振幅须 ≥ 该比例 × 首轮振幅中位数;
        如 0.35 可去掉很弱的剥离伪峰。
    return_peel_residual : bool – 为 True 时额外返回首轮剥离后的残差图 ``image - Σ拟合峰`` (仅当
        ``peak_peel`` 且首轮 ``ions`` 非空时非 None; 否则为 None)。

    Returns
    -------
    ions : list[dict]  每个离子的参数:
        x0, y0       – 中心坐标 (像素, 亚像素精度)
        sigma_minor  – 短轴 sigma (像素)
        sigma_major  – 长轴 sigma (像素)
        theta_deg    – 长轴相对 x 轴的旋转角 (度)
        amplitude    – 高斯峰值强度

    若 ``return_peel_residual`` 为 True, 返回 ``(ions, boundary, peel_residual)``,
    其中 ``peel_residual`` 为 ndarray 或 None; 否则返回 ``(ions, boundary)``.
    """
    img = image.astype(np.float64)
    peel_residual = None
    h, w = img.shape
    s_lo, s_hi = sigma_range

    if isinstance(fit_hw, (list, tuple)):
        hw_y, hw_x = fit_hw
    else:
        hw_y = hw_x = fit_hw

    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg
    if use_matched_filter:
        kernel = build_matched_kernel()
        detect_map = fftconvolve(signal, kernel, mode="same")
    else:
        detect_map = signal

    boundary = estimate_crystal_boundary(signal)

    peak_yx = peaks_from_detect_map(
        detect_map,
        peak_size,
        rel_threshold,
        use_y_threshold_comp,
        amp_y_coef,
        amp_y_coef_path,
        amp_y_coef_mode,
        comp_floor,
        boundary,
    )

    jpx = joint_pair_x_gap
    if joint_pair_y_gap is not None and jpx is None:
        jpx = max(4.0, float(hw_x))

    ions = fit_all_peaks(
        img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, refine=refine, fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=joint_pair_y_gap, joint_pair_x_gap=jpx,
    )
    ions.sort(key=lambda d: (d["y0"], d["x0"]))

    if peak_peel and ions:
        peel = _accumulate_peel_model(h, w, ions, margin_sigma=peak_peel_margin_sigma)
        img_peeled = img - peel
        if return_peel_residual:
            peel_residual = img_peeled
        bg2 = gaussian_filter(img_peeled, sigma=bg_sigma)
        signal2 = img_peeled - bg2
        if use_matched_filter:
            detect_map2 = fftconvolve(signal2, kernel, mode="same")
        else:
            detect_map2 = signal2
        rel2 = peak_peel_rel_threshold
        if rel2 is None:
            rel2 = rel_threshold
        peak_yx2 = peaks_from_detect_map(
            detect_map2,
            peak_size,
            rel2,
            use_y_threshold_comp,
            amp_y_coef,
            amp_y_coef_path,
            amp_y_coef_mode,
            comp_floor,
            boundary,
        )
        if peak_peel_y_edges_only and boundary is not None:
            peak_yx2 = filter_peak_yx_y_edge_bands(
                peak_yx2, boundary, peak_peel_y_edge_frac,
            )
        ions2 = fit_all_peaks(
            img_peeled, signal2, peak_yx2, hw_y, hw_x, s_lo, s_hi,
            h, w, refine=refine, fix_theta_zero=fix_theta_zero,
            joint_pair_y_gap=joint_pair_y_gap, joint_pair_x_gap=jpx,
        )
        ions2.sort(key=lambda d: (d["y0"], d["x0"]))
        if peak_peel_y_edges_only and boundary is not None:
            ions2 = filter_ions_y_edge_bands(
                ions2, boundary, peak_peel_y_edge_frac,
            )
        if peak_peel_min_amp_frac is not None and ions:
            ref_amp = float(np.median([float(d["amplitude"]) for d in ions]))
            lo = float(peak_peel_min_amp_frac) * ref_amp
            ions2 = [d for d in ions2 if float(d["amplitude"]) >= lo]
        ions = merge_ions_by_distance(ions, ions2, peak_peel_min_sep)

    if return_peel_residual:
        return ions, boundary, peel_residual
    return ions, boundary
