"""
离子晶格图像中单离子的检测与椭圆拟合。

算法流程:
  1. 各向异性高斯平滑估计背景并减去
  2. 匹配滤波器增强离子信号, 抑制噪声
  3. 各向异性局部极大值检测定位离子候选中心
  4. 对每个候选用 2D 高斯最小二乘拟合 (可选 y 近邻 N=2 联合拟合, θ=0)
  5. 质量过滤, 输出中心坐标与椭圆参数
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, binary_erosion
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import time

# ──────────────────────────── 2D Gaussian Model ────────────────────────────

def _gauss2d(coords, amp, x0, y0, sx, sy, theta, offset):
    """旋转二维高斯函数。
    sx: 沿旋转角 theta 方向的 sigma
    sy: 沿垂直于 theta 方向的 sigma
    """
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return offset + amp * np.exp(-0.5 * (xr**2 / sx**2 + yr**2 / sy**2))


def _gauss2d_aligned(coords, amp, x0, y0, sx, sy, offset):
    """轴对齐二维高斯 (椭圆转角 θ=0, 半轴分别沿 x、y)。"""
    x, y = coords
    return offset + amp * np.exp(
        -0.5 * ((x - x0) ** 2 / sx**2 + (y - y0) ** 2 / sy**2)
    )


def _two_gauss2d_aligned_flat(xy, a1, x1, y1, sx1, sy1, a2, x2, y2, sx2, sy2, b):
    """两峰轴对齐高斯之和 (展平坐标), 供 curve_fit 使用。"""
    x, y = xy
    g1 = a1 * np.exp(
        -0.5 * ((x - x1) ** 2 / sx1**2 + (y - y1) ** 2 / sy1**2)
    )
    g2 = a2 * np.exp(
        -0.5 * ((x - x2) ** 2 / sx2**2 + (y - y2) ** 2 / sy2**2)
    )
    return b + g1 + g2


# ──────────────────────────── Core Detection ───────────────────────────────

def _build_matched_kernel(sigma_x=1.2, sigma_y=1.8, half_size=3):
    """构造与离子 PSF 匹配的高斯核，用于增强信噪比。"""
    ky, kx = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    kernel = np.exp(-0.5 * (kx**2 / sigma_x**2 + ky**2 / sigma_y**2))
    kernel /= kernel.sum()
    return kernel


def _eval_amp_y_model(y_rel: np.ndarray, coef: np.ndarray, mode: str = "even") -> np.ndarray:
    """Evaluate amplitude-vs-y model.

    mode='even': coef=[a0,a2,a4], amp(y)=a0+a2*y^2+a4*y^4
    mode='poly2': coef=[p2,p1,p0], amp(y)=p2*y^2+p1*y+p0
    """
    c = np.asarray(coef, dtype=np.float64).ravel()
    if c.size != 3:
        raise ValueError(f"Amplitude coefficient must contain 3 values, got shape {c.shape}.")

    y = np.asarray(y_rel, dtype=np.float64)
    if mode == "even":
        return c[0] + c[1] * y**2 + c[2] * y**4
    if mode == "poly2":
        return np.polyval(c, y)
    raise ValueError(f"Unsupported amp_y_coef_mode: {mode}")


def _build_row_threshold_scale(
    h: int,
    cy_ref: float,
    coef: np.ndarray,
    mode: str = "even",
    floor: float = 0.2,
) -> np.ndarray:
    """Build row-wise threshold scale in [floor, +inf)."""
    y_rel = np.arange(h, dtype=np.float64) - float(cy_ref)
    amp_env = _eval_amp_y_model(y_rel, coef, mode=mode)
    amp_env = np.clip(amp_env, 1e-6, None)
    scale = amp_env / float(np.max(amp_env))
    return np.clip(scale, max(float(floor), 1e-6), None)


def detect_ions(image, bg_sigma=(10, 30), peak_size=(5, 9),
                rel_threshold=0.025, fit_hw=(3, 4),
                sigma_range=(0.3, 3.5), use_matched_filter=True,
                refine=True, fix_theta_zero=False,
                use_y_threshold_comp=False,
                 amp_y_coef=None, amp_y_coef_path=None,
                 amp_y_coef_mode="even", comp_floor=0.2,
                 joint_pair_y_gap=None, joint_pair_x_gap=None):
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

    Returns
    -------
    ions : list[dict]  每个离子的参数:
        x0, y0       – 中心坐标 (像素, 亚像素精度)
        sigma_minor  – 短轴 sigma (像素)
        sigma_major  – 长轴 sigma (像素)
        theta_deg    – 长轴相对 x 轴的旋转角 (度)
        amplitude    – 高斯峰值强度
    """
    img = image.astype(np.float64)
    h, w = img.shape
    s_lo, s_hi = sigma_range

    if isinstance(fit_hw, (list, tuple)):
        hw_y, hw_x = fit_hw
    else:
        hw_y = hw_x = fit_hw

    # 各向异性背景减除
    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg

    # 匹配滤波器：与离子 PSF 卷积提高信噪比
    if use_matched_filter:
        kernel = _build_matched_kernel()
        detect_map = fftconvolve(signal, kernel, mode="same")
    else:
        detect_map = signal

    # 先估计晶格边界(后续也用于可选的 y 向阈值补偿)
    boundary = _estimate_crystal_boundary(signal)

    # 各向异性局部极大值检测
    local_max = maximum_filter(detect_map, size=peak_size)
    thresh = rel_threshold * detect_map.max()

    if use_y_threshold_comp:
        coef = amp_y_coef
        if coef is None:
            if amp_y_coef_path is None:
                raise ValueError(
                    "use_y_threshold_comp=True requires amp_y_coef or amp_y_coef_path."
                )
            coef = np.load(amp_y_coef_path)

        cy_ref = boundary[1] if boundary is not None else (h - 1) / 2.0
        row_scale = _build_row_threshold_scale(
            h,
            cy_ref=cy_ref,
            coef=coef,
            mode=amp_y_coef_mode,
            floor=comp_floor,
        )
        thresh_map = thresh * row_scale[:, np.newaxis]
        peak_mask = (detect_map == local_max) & (detect_map > thresh_map)
    else:
        peak_mask = (detect_map == local_max) & (detect_map > thresh)

    peak_yx = np.argwhere(peak_mask)

    # 在拟合前用晶格边界椭圆过滤噪声候选, 节省拟合时间
    if boundary is not None:
        peak_yx = _apply_boundary_filter(peak_yx, *boundary)

    jpx = joint_pair_x_gap
    if joint_pair_y_gap is not None and jpx is None:
        jpx = max(4.0, float(hw_x))

    ions = _fit_all_peaks(
        img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, refine=refine, fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=joint_pair_y_gap, joint_pair_x_gap=jpx,
    )
    ions.sort(key=lambda d: (d["y0"], d["x0"]))
    return ions, boundary


def _estimate_crystal_boundary(signal, smooth_sigma=8,
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
    boundary = mask & ~binary_erosion(mask, iterations=2)
    by, bx = np.where(boundary)

    if len(bx) < 10:
        return None

    cx = (bx.max() + bx.min()) / 2.0
    cy = (by.max() + by.min()) / 2.0
    a = (bx.max() - bx.min()) / 2.0 * margin
    b = (by.max() - by.min()) / 2.0 * margin
    return cx, cy, a, b


def _apply_boundary_filter(peak_yx, cx, cy, a, b):
    """保留椭圆边界内的候选峰, 移除外部噪声。"""
    px = peak_yx[:, 1].astype(np.float64)
    py = peak_yx[:, 0].astype(np.float64)
    inside = ((px - cx) / a)**2 + ((py - cy) / b)**2 <= 1.0
    return peak_yx[inside]


def _soft_weight_map(ph, pw_, ly, lx, hw_y, hw_x):
    """与单峰 weight_template 同形状的软权重, 可贴任意大小 ROI。"""
    yy, xx = np.mgrid[0:ph, 0:pw_].astype(np.float64)
    return np.exp(
        -0.5 * (
            (xx - lx) ** 2 / (hw_x * 0.7) ** 2
            + (yy - ly) ** 2 / (hw_y * 0.45) ** 2
        )
    )


def _partition_peaks_joint(peak_yx, y_gap, x_gap):
    """按 y 优先排序, 将峰分为「单峰」或与 y/x 近邻的另一峰组成「双峰对」。

    每个峰至多参与一对; 在 |Δy|≤y_gap 且 |Δx|≤x_gap 的候选中取 |Δy| 最小者配对。
    返回 [('single', (py, px)), ('pair', (py1, px1), (py2, px2)), ...]
    """
    n = peak_yx.shape[0]
    if n == 0:
        return []
    py = peak_yx[:, 0].astype(np.int32)
    px = peak_yx[:, 1].astype(np.int32)
    order = np.argsort(py, kind="mergesort")
    used = np.zeros(n, dtype=bool)
    groups = []
    for t in range(n):
        i = int(order[t])
        if used[i]:
            continue
        used[i] = True
        best_j = -1
        best_dy = float(y_gap) + 1.0
        for j in range(n):
            if used[j]:
                continue
            dy = abs(float(py[i]) - float(py[j]))
            if dy > y_gap:
                continue
            dx = abs(float(px[i]) - float(px[j]))
            if dx > x_gap:
                continue
            if dy < best_dy:
                best_dy = dy
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            p1 = (int(py[i]), int(px[i]))
            p2 = (int(py[best_j]), int(px[best_j]))
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
            groups.append(("pair", p1, p2))
        else:
            groups.append(("single", (int(py[i]), int(px[i]))))
    return groups


def _fit_all_peaks(img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
                   h, w, refine=True, fix_theta_zero=False,
                   joint_pair_y_gap=None, joint_pair_x_gap=None):
    """对所有候选峰做高斯拟合, 可选两阶段精修。"""
    # 权重模板: y 方向更锐利 (0.45), 抑制紧邻离子
    wy, wx = 2 * hw_y + 1, 2 * hw_x + 1
    gy_t, gx_t = np.mgrid[0:wy, 0:wx]
    cy_t, cx_t = hw_y, hw_x
    weight_template = np.exp(-0.5 * ((gx_t - cx_t)**2 / (hw_x * 0.7)**2
                                   + (gy_t - cy_t)**2 / (hw_y * 0.45)**2))

    j_y = joint_pair_y_gap
    j_x = joint_pair_x_gap
    ions = _do_fit_pass(
        img, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, weight_template, sigma_init=(1.2, 1.8),
        fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=j_y, joint_pair_x_gap=j_x,
    )

    if not refine or len(ions) < 20:
        return ions

    # ── 两阶段精修 ──
    # 从稳定拟合中计算参考 sigma
    majors = np.array([d["sigma_major"] for d in ions])
    minors = np.array([d["sigma_minor"] for d in ions])
    joint_skip = np.array(
        [bool(d.get("_joint_skip_refit", False)) for d in ions], dtype=bool
    )
    stable = (majors < s_hi * 0.9) & (minors < s_hi * 0.9) & ~joint_skip
    if stable.sum() < 10:
        return ions

    ref_minor = float(np.median(minors[stable]))
    ref_major = float(np.median(majors[stable]))

    # 标记需要重新拟合的离子 (sigma 偏离中位值太远)
    needs_refit = (majors > ref_major * 1.8) | (minors > ref_minor * 2.0)
    needs_refit &= ~joint_skip
    if needs_refit.sum() == 0:
        return ions

    # 对需重新拟合的离子, 用参考 sigma 约束上界
    refit_indices = np.where(needs_refit)[0]
    refit_yx = np.array([[ions[i]["_py"], ions[i]["_px"]] for i in refit_indices])

    tight_s_hi_minor = ref_minor * 1.6
    tight_s_hi_major = ref_major * 1.6

    refitted = _do_fit_pass(
        img, refit_yx, hw_y, hw_x, s_lo,
        max(tight_s_hi_minor, tight_s_hi_major),
        h, w, weight_template,
        sigma_init=(ref_minor, ref_major),
        fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=None,
        joint_pair_x_gap=None,
    )

    # 用重拟合结果替换
    refit_map = {}
    for ion in refitted:
        key = (ion["_py"], ion["_px"])
        refit_map[key] = ion

    result = []
    for i, ion in enumerate(ions):
        if needs_refit[i]:
            key = (ion["_py"], ion["_px"])
            if key in refit_map:
                result.append(refit_map[key])
            else:
                result.append(ion)
        else:
            result.append(ion)

    return result


def _fit_single_peak_at(
    img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
    weight_template, sigma_init, fix_theta_zero,
):
    """对单个候选峰做高斯拟合, 成功则返回参数字典, 否则 None。"""
    py, px = int(py), int(px)
    si_x, si_y = sigma_init
    y1, y2 = max(0, py - hw_y), min(h, py + hw_y + 1)
    x1, x2 = max(0, px - hw_x), min(w, px + hw_x + 1)
    patch = img[y1:y2, x1:x2]
    ph, pw_ = patch.shape

    yy, xx = np.mgrid[0:ph, 0:pw_]
    ly, lx = py - y1, px - x1
    amp0 = float(patch[ly, lx] - patch.min())
    if amp0 < 1:
        return None

    wt_y1 = hw_y - ly
    wt_x1 = hw_x - lx
    weights = weight_template[wt_y1:wt_y1 + ph, wt_x1:wt_x1 + pw_]
    fit_sigma = 1.0 / np.sqrt(np.clip(weights, 0.01, None))

    try:
        off0 = float(patch.min())
        if fix_theta_zero:
            p0 = [amp0, lx, ly, si_x, si_y, off0]
            lo = [0, 0, 0, s_lo, s_lo, 0]
            hi = [amp0 * 4, pw_ - 1, ph - 1, s_hi, s_hi, float(patch.max())]
            popt, _ = curve_fit(
                _gauss2d_aligned, (xx.ravel(), yy.ravel()), patch.ravel(),
                p0=p0, bounds=(lo, hi), maxfev=2000,
                sigma=fit_sigma.ravel(),
            )
            amp, fx, fy, sx, sy, offset = popt
            if sx > sy:
                sigma_minor, sigma_major = sy, sx
            else:
                sigma_minor, sigma_major = sx, sy
            theta_deg = 0.0
        else:
            p0 = [amp0, lx, ly, si_x, si_y, 0.0, off0]
            lo = [0, 0, 0, s_lo, s_lo, -np.pi, 0]
            hi = [amp0 * 4, pw_ - 1, ph - 1, s_hi, s_hi, np.pi, float(patch.max())]
            popt, _ = curve_fit(
                _gauss2d, (xx.ravel(), yy.ravel()), patch.ravel(),
                p0=p0, bounds=(lo, hi), maxfev=2000,
                sigma=fit_sigma.ravel(),
            )
            amp, fx, fy, sx, sy, theta, offset = popt

            if sx > sy:
                sx, sy = sy, sx
                theta += np.pi / 2

            theta_deg = np.degrees(theta)
            theta_deg = ((theta_deg + 90) % 180) - 90

            sigma_minor, sigma_major = sx, sy

    except (RuntimeError, ValueError):
        return None

    gx, gy = x1 + fx, y1 + fy

    if abs(gx - px) > hw_x or abs(gy - py) > hw_y:
        return None
    if fix_theta_zero:
        if not (s_lo <= sx <= s_hi and s_lo <= sy <= s_hi):
            return None
    else:
        if not (s_lo <= sigma_minor <= s_hi and s_lo <= sigma_major <= s_hi):
            return None

    return {
        "x0": gx, "y0": gy,
        "sigma_minor": sigma_minor, "sigma_major": sigma_major,
        "theta_deg": theta_deg,
        "amplitude": amp,
        "_py": py, "_px": px,
    }


def _fit_joint_two_peaks_at(
    img, py1, px1, py2, px2, hw_y, hw_x, s_lo, s_hi, h, w, sigma_init,
):
    """在合并 ROI 内对两峰做 N=2、θ=0 联合拟合; 成功返回两离子列表, 否则 None。"""
    si_x, si_y = sigma_init
    py1, px1, py2, px2 = int(py1), int(px1), int(py2), int(px2)
    y1 = max(0, min(py1, py2) - hw_y)
    y2 = min(h, max(py1, py2) + hw_y + 1)
    x1 = max(0, min(px1, px2) - hw_x)
    x2 = min(w, max(px1, px2) + hw_x + 1)
    patch = img[y1:y2, x1:x2]
    ph, pw_ = patch.shape
    if ph < 3 or pw_ < 3:
        return None

    yy, xx = np.mgrid[0:ph, 0:pw_]
    ly1, lx1 = py1 - y1, px1 - x1
    ly2, lx2 = py2 - y1, px2 - x1
    off0 = float(patch.min())
    amp01 = float(patch[ly1, lx1] - off0)
    amp02 = float(patch[ly2, lx2] - off0)
    if amp01 < 1 or amp02 < 1:
        return None

    w1 = _soft_weight_map(ph, pw_, ly1, lx1, hw_y, hw_x)
    w2 = _soft_weight_map(ph, pw_, ly2, lx2, hw_y, hw_x)
    fit_sigma = 1.0 / np.sqrt(np.clip(np.maximum(w1, w2), 0.01, None))

    a_hi = max(amp01, amp02) * 4.0
    p0 = [amp01, float(lx1), float(ly1), si_x, si_y,
          amp02, float(lx2), float(ly2), si_x, si_y, off0]
    lo = [0, 0, 0, s_lo, s_lo, 0, 0, 0, s_lo, s_lo, 0.0]
    hi = [a_hi, pw_ - 1e-3, ph - 1e-3, s_hi, s_hi,
          a_hi, pw_ - 1e-3, ph - 1e-3, s_hi, s_hi, float(patch.max())]

    try:
        popt, _ = curve_fit(
            _two_gauss2d_aligned_flat,
            (xx.ravel(), yy.ravel()),
            patch.ravel(),
            p0=p0, bounds=(lo, hi), maxfev=8000,
            sigma=fit_sigma.ravel(),
        )
    except (RuntimeError, ValueError):
        return None

    a1, fx1, fy1, sx1, sy1, a2, fx2, fy2, sx2, sy2, offset = popt
    gx1, gy1 = x1 + fx1, y1 + fy1
    gx2, gy2 = x1 + fx2, y1 + fy2

    peaks = [(px1, py1), (px2, py2)]
    fit_blocks = [(gx1, gy1, a1, sx1, sy1), (gx2, gy2, a2, sx2, sy2)]

    def _dist2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    cost_same = _dist2((gx1, gy1), peaks[0]) + _dist2((gx2, gy2), peaks[1])
    cost_swap = _dist2((gx1, gy1), peaks[1]) + _dist2((gx2, gy2), peaks[0])
    if cost_swap < cost_same:
        assignments = [(fit_blocks[0], peaks[1]), (fit_blocks[1], peaks[0])]
    else:
        assignments = [(fit_blocks[0], peaks[0]), (fit_blocks[1], peaks[1])]

    out = []
    for (gx, gy, amp, sx, sy), (px, py) in assignments:
        if abs(gx - px) > hw_x or abs(gy - py) > hw_y:
            return None
        if not (s_lo <= sx <= s_hi and s_lo <= sy <= s_hi):
            return None
        if sx > sy:
            smin, smaj = sy, sx
        else:
            smin, smaj = sx, sy
        out.append({
            "x0": gx, "y0": gy,
            "sigma_minor": smin, "sigma_major": smaj,
            "theta_deg": 0.0,
            "amplitude": amp,
            "_py": py, "_px": px,
            "_joint_skip_refit": True,
        })
    return out


def _do_fit_pass(img, peak_yx, hw_y, hw_x, s_lo, s_hi,
                 h, w, weight_template, sigma_init=(1.2, 1.8),
                 fix_theta_zero=False,
                 joint_pair_y_gap=None, joint_pair_x_gap=None):
    """单次高斯拟合: 可选 y 近邻双峰联合拟合 (N=2, θ=0), 其余为单峰。"""
    if joint_pair_y_gap is not None:
        jx = joint_pair_x_gap if joint_pair_x_gap is not None else max(4.0, float(hw_x))
        groups = _partition_peaks_joint(peak_yx, float(joint_pair_y_gap), float(jx))
    else:
        groups = [
            ("single", (int(r[0]), int(r[1])))
            for r in np.atleast_2d(peak_yx)
        ]

    ions = []
    for g in groups:
        if g[0] == "single":
            py, px = g[1]
            ion = _fit_single_peak_at(
                img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
                weight_template, sigma_init, fix_theta_zero,
            )
            if ion is not None:
                ions.append(ion)
        elif g[0] == "pair":
            assert len(g) == 3
            p1, p2 = g[1], g[2]
            pair = _fit_joint_two_peaks_at(
                img, p1[0], p1[1], p2[0], p2[1],
                hw_y, hw_x, s_lo, s_hi, h, w, sigma_init,
            )
            if pair is not None:
                ions.extend(pair)
            else:
                for py, px in (p1, p2):
                    ion = _fit_single_peak_at(
                        img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
                        weight_template, sigma_init, fix_theta_zero,
                    )
                    if ion is not None:
                        ions.append(ion)

    return ions

# ──────────────────────────── Visualization ────────────────────────────────

def visualize(image, ions, n_sigma=2.0, title="", output_path=None,
              show_zoom=True, zoom_center=None, zoom_size=80,
              boundary=None):
    """
    在图像上标注拟合椭圆。

    n_sigma 控制椭圆的半轴长度 (n_sigma * sigma)。
    boundary: (cx, cy, a, b) 晶格边界椭圆参数, 若提供则绘制在图上。
    可选显示一个局部放大区域。
    """
    nrows = 6 if show_zoom else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(20, 5 + (3 * 5 if show_zoom else 0)),
                             gridspec_kw={"height_ratios": [5, 2, 3, 3, 3, 2]
                                          if show_zoom else [1]})
    if not show_zoom:
        axes = [axes]

    # ── 全局图 ──
    ax = axes[0]
    ax.imshow(image, cmap="gray", aspect="auto", vmin=np.percentile(image, 1),
              vmax=np.percentile(image, 99.5))
    for ion in ions:
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor="red", facecolor="none", linewidth=0.4, alpha=0.8,
        )
        ax.add_patch(ell)
    if boundary is not None:
        bcx, bcy, ba, bb = boundary
        bnd_ell = Ellipse(
            xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
            edgecolor="cyan", facecolor="none",
            linewidth=1.2, linestyle="--", alpha=0.9,
        )
        ax.add_patch(bnd_ell)
    ax.set_title(
        f"{title}   [{len(ions)} ions, ellipse = {n_sigma} sigma]",
        fontsize=13,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # ── 放大图 (5 个区域: 上边缘, 左, 中心, 右, 下边缘) ──
    if show_zoom:
        regions = [
            ("Top edge",    (500, 35),  100, 20),
            ("Left",        (200, 75),   60, 30),
            ("Center",      (500, 85),   60, 30),
            ("Right",       (800, 75),   60, 30),
            ("Bottom edge", (500, 130), 100, 20),
        ]
        for i, (label, (rcx, rcy), rzs_x, rzs_y) in enumerate(regions):
            ax2 = axes[1 + i]
            x1z = max(0, rcx - rzs_x)
            x2z = min(image.shape[1], rcx + rzs_x)
            y1z = max(0, rcy - rzs_y)
            y2z = min(image.shape[0], rcy + rzs_y)

            ax2.imshow(image, cmap="gray", aspect="equal",
                       vmin=np.percentile(image, 1),
                       vmax=np.percentile(image, 99.5))
            for ion in ions:
                if x1z <= ion["x0"] <= x2z and y1z <= ion["y0"] <= y2z:
                    ell = Ellipse(
                        xy=(ion["x0"], ion["y0"]),
                        width=2 * n_sigma * ion["sigma_minor"],
                        height=2 * n_sigma * ion["sigma_major"],
                        angle=ion["theta_deg"],
                        edgecolor="lime", facecolor="none", linewidth=1.2,
                    )
                    ax2.add_patch(ell)
                    ax2.plot(ion["x0"], ion["y0"], "r.", markersize=2)
            if boundary is not None:
                bcx, bcy, ba, bb = boundary
                bnd_ell = Ellipse(
                    xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
                    edgecolor="cyan", facecolor="none",
                    linewidth=1.5, linestyle="--", alpha=0.9,
                )
                ax2.add_patch(bnd_ell)
            ax2.set_xlim(x1z, x2z)
            ax2.set_ylim(y2z, y1z)
            ax2.set_title(
                f"Zoom: {label}  x=[{x1z},{x2z}]  y=[{y1z},{y2z}]",
                fontsize=11,
            )
            ax2.set_xlabel("x (pixel)")
            ax2.set_ylabel("y (pixel)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[已保存] {output_path}")
    plt.close(fig)

# ──────────────────────────── Statistics ────────────────────────────────────

def print_summary(ions):
    if not ions:
        print("未检测到离子。")
        return
    minors = np.array([d["sigma_minor"] for d in ions])
    majors = np.array([d["sigma_major"] for d in ions])
    amps   = np.array([d["amplitude"]   for d in ions])
    ratios = majors / minors

    print(f"\n检测结果: {len(ions)} 个离子")
    print(f"  σ_minor  (短轴): mean={minors.mean():.2f} ± {minors.std():.2f}  "
          f"range=[{minors.min():.2f}, {minors.max():.2f}]")
    print(f"  σ_major  (长轴): mean={majors.mean():.2f} ± {majors.std():.2f}  "
          f"range=[{majors.min():.2f}, {majors.max():.2f}]")
    print(f"  长短轴比 (major/minor): mean={ratios.mean():.2f} ± {ratios.std():.2f}")
    print(f"  振幅: mean={amps.mean():.1f} ± {amps.std():.1f}  "
          f"range=[{amps.min():.1f}, {amps.max():.1f}]")

    xs = np.array([d["x0"] for d in ions])
    ys = np.array([d["y0"] for d in ions])
    print(f"  中心范围:  x ∈ [{xs.min():.1f}, {xs.max():.1f}],  "
          f"y ∈ [{ys.min():.1f}, {ys.max():.1f}]")


def _parse_slice_token(token):
    token = token.strip()
    if not token:
        raise ValueError("Empty index token.")

    if ":" not in token:
        return int(token)

    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice token: {token}")

    def _to_int_or_none(s):
        s = s.strip()
        return None if s == "" else int(s)

    start = _to_int_or_none(parts[0]) if len(parts) >= 1 else None
    stop = _to_int_or_none(parts[1]) if len(parts) >= 2 else None
    step = _to_int_or_none(parts[2]) if len(parts) >= 3 else None

    if step == 0:
        raise ValueError(f"Slice step cannot be 0: {token}")
    return slice(start, stop, step)


def _resolve_indices(spec_args, total):
    """解析索引规范，支持并集，如 ['::3,0:10', '25', '-1']。"""
    if not spec_args:
        spec_args = ["0"]

    selected = set()
    all_indices = np.arange(total)

    for raw in spec_args:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue

            obj = _parse_slice_token(token)
            if isinstance(obj, slice):
                selected.update(all_indices[obj].tolist())
            else:
                idx = obj
                if idx < 0:
                    idx += total
                if 0 <= idx < total:
                    selected.add(idx)
                else:
                    print(f"[skip] index {obj} out of range [0, {total - 1}]")

    return sorted(selected)

# ──────────────────────────── Main ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="离子检测与椭圆拟合可视化")
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help=(
            "要处理的索引规范，支持整数和 numpy 风格切片，并支持并集。"
            "例如: 0 5 -1 ::3,0:10"
        ),
    )
    parser.add_argument(
        "--save-pos",
        action="store_true",
        help="保存离子中心坐标到 .npy 文件。",
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=None,
        help="离子中心保存目录，默认是项目根目录下的 IonPos。",
    )
    parser.add_argument(
        "--use-y-thresh-comp",
        action="store_true",
        help="启用 y 方向亮度模型补偿阈值(降低边缘较暗区域的判定门槛)。",
    )
    parser.add_argument(
        "--amp-coef-path",
        type=Path,
        default=None,
        help=(
            "y 方向亮度拟合系数文件(.npy)。"
            "默认使用 visualization_output/amp_vs_y_coef_10.npy。"
        ),
    )
    parser.add_argument(
        "--amp-coef-mode",
        type=str,
        choices=("even", "poly2"),
        default="even",
        help="系数解释方式: even=[a0,a2,a4], poly2=[p2,p1,p0]。",
    )
    parser.add_argument(
        "--comp-floor",
        type=float,
        default=0.2,
        help="阈值缩放下限，越小边缘越容易被检出(也更易引入噪声)。",
    )
    parser.add_argument(
        "--fix-theta-zero",
        action="store_true",
        help="高斯拟合固定椭圆转角 θ=0 (轴对齐 x/y，不拟合旋转)。",
    )
    parser.add_argument(
        "--no-matched-filter",
        action="store_true",
        help="禁用匹配滤波; 峰检测直接使用减背景后的信号 (默认启用匹配滤波)。",
    )
    parser.add_argument(
        "--joint-pair-y-gap",
        type=float,
        default=None,
        metavar="DY",
        help=(
            "启用 y 向近邻双峰联合拟合 (N=2, θ=0): 两候选峰 |Δy|≤DY 且 |Δx| 在 "
            "--joint-pair-x-gap 内则合并 ROI 同时拟合; 失败则回退单峰。"
        ),
    )
    parser.add_argument(
        "--joint-pair-x-gap",
        type=float,
        default=None,
        metavar="DX",
        help="联合配对允许的 |Δx| (像素); 省略时默认 max(4, 拟合半宽 hw_x)。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "20260305_1727"
    out_dir  = project_root / "visualization_output"
    out_dir.mkdir(exist_ok=True)
    pos_dir = args.pos_dir or (project_root / "IonPos")
    default_amp_coef_path = project_root / "visualization_output" / "amp_vs_y_coef_10.npy"
    amp_coef_path = args.amp_coef_path or default_amp_coef_path
    if args.save_pos:
        pos_dir.mkdir(exist_ok=True)

    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    print(f"共找到 {len(files)} 个 npy 文件")
    selected_indices = _resolve_indices(args.indices, len(files))
    if not selected_indices:
        raise ValueError("没有可处理的索引。请检查输入的 indices。")

    for idx in selected_indices:
        target = files[idx]
        print(f"\n[{idx:04d}] 加载: {target.name}")
        image = np.load(target)

        print("正在检测离子...")
        t0 = time.perf_counter()
        ions, boundary = detect_ions(
            image,
            use_y_threshold_comp=args.use_y_thresh_comp,
            amp_y_coef_path=amp_coef_path,
            amp_y_coef_mode=args.amp_coef_mode,
            comp_floor=args.comp_floor,
            fix_theta_zero=args.fix_theta_zero,
            use_matched_filter=not args.no_matched_filter,
            joint_pair_y_gap=args.joint_pair_y_gap,
            joint_pair_x_gap=args.joint_pair_x_gap,
        )
        elapsed = time.perf_counter() - t0
        print(f"检测耗时: {elapsed:.2f} 秒")
        if boundary:
            cx, cy, a, b = boundary
            print(f"晶格边界椭圆: 中心=({cx:.1f}, {cy:.1f}), "
                  f"半轴=({a:.1f}, {b:.1f})")

        print_summary(ions)

        if args.save_pos:
            positions = np.array([[ion["x0"], ion["y0"]] for ion in ions], dtype=np.float64)
            if positions.size == 0:
                positions = np.empty((0, 2), dtype=np.float64)
            pos_path = pos_dir / target.name
            np.save(pos_path, positions)
            print(f"[已保存离子中心] {pos_path}")

        out_path = out_dir / f"ion_ellipses_{idx:04d}.png"
        visualize(image, ions, n_sigma=2.0,
                  title=f"[{idx:04d}] {target.name}",
                  output_path=out_path,
                  show_zoom=True, boundary=boundary)
