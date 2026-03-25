"""候选分组与高斯拟合 (单峰 / 联合双峰 / 精修)。"""
import numpy as np
from scipy.optimize import curve_fit

from .gaussian import _gauss2d, _gauss2d_aligned, _two_gauss2d_aligned_flat


def soft_weight_map(ph, pw_, ly, lx, hw_y, hw_x):
    """与单峰 weight_template 同形状的软权重, 可贴任意大小 ROI。"""
    yy, xx = np.mgrid[0:ph, 0:pw_].astype(np.float64)
    return np.exp(
        -0.5 * (
            (xx - lx) ** 2 / (hw_x * 0.7) ** 2
            + (yy - ly) ** 2 / (hw_y * 0.45) ** 2
        )
    )


def partition_peaks_joint(peak_yx, y_gap, x_gap):
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


def fit_all_peaks(img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
                  h, w, refine=True, fix_theta_zero=False,
                  joint_pair_y_gap=None, joint_pair_x_gap=None):
    """对所有候选峰做高斯拟合, 可选两阶段精修。"""
    wy, wx = 2 * hw_y + 1, 2 * hw_x + 1
    gy_t, gx_t = np.mgrid[0:wy, 0:wx]
    cy_t, cx_t = hw_y, hw_x
    weight_template = np.exp(-0.5 * ((gx_t - cx_t)**2 / (hw_x * 0.7)**2
                                   + (gy_t - cy_t)**2 / (hw_y * 0.45)**2))

    j_y = joint_pair_y_gap
    j_x = joint_pair_x_gap
    ions = do_fit_pass(
        img, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, weight_template, sigma_init=(1.2, 1.8),
        fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=j_y, joint_pair_x_gap=j_x,
    )

    if not refine or len(ions) < 20:
        return ions

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

    needs_refit = (majors > ref_major * 1.8) | (minors > ref_minor * 2.0)
    needs_refit &= ~joint_skip
    if needs_refit.sum() == 0:
        return ions

    refit_indices = np.where(needs_refit)[0]
    refit_yx = np.array([[ions[i]["_py"], ions[i]["_px"]] for i in refit_indices])

    tight_s_hi_minor = ref_minor * 1.6
    tight_s_hi_major = ref_major * 1.6

    refitted = do_fit_pass(
        img, refit_yx, hw_y, hw_x, s_lo,
        max(tight_s_hi_minor, tight_s_hi_major),
        h, w, weight_template,
        sigma_init=(ref_minor, ref_major),
        fix_theta_zero=fix_theta_zero,
        joint_pair_y_gap=None,
        joint_pair_x_gap=None,
    )

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


def fit_single_peak_at(
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

    out = {
        "x0": gx, "y0": gy,
        "sigma_minor": sigma_minor, "sigma_major": sigma_major,
        "theta_deg": theta_deg,
        "amplitude": amp,
        "_py": py, "_px": px,
    }
    if fix_theta_zero:
        out["_sigma_x"] = float(sx)
        out["_sigma_y"] = float(sy)
    return out


def fit_joint_two_peaks_at(
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

    w1 = soft_weight_map(ph, pw_, ly1, lx1, hw_y, hw_x)
    w2 = soft_weight_map(ph, pw_, ly2, lx2, hw_y, hw_x)
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
            "_sigma_x": float(sx),
            "_sigma_y": float(sy),
            "_py": py, "_px": px,
            "_joint_skip_refit": True,
        })
    return out


def do_fit_pass(img, peak_yx, hw_y, hw_x, s_lo, s_hi,
                h, w, weight_template, sigma_init=(1.2, 1.8),
                fix_theta_zero=False,
                joint_pair_y_gap=None, joint_pair_x_gap=None):
    """单次高斯拟合: 可选 y 近邻双峰联合拟合 (N=2, θ=0), 其余为单峰。"""
    if joint_pair_y_gap is not None:
        jx = joint_pair_x_gap if joint_pair_x_gap is not None else max(4.0, float(hw_x))
        groups = partition_peaks_joint(peak_yx, float(joint_pair_y_gap), float(jx))
    else:
        groups = [
            ("single", (int(r[0]), int(r[1])))
            for r in np.atleast_2d(peak_yx)
        ]

    ions = []
    for g in groups:
        if g[0] == "single":
            py, px = g[1]
            ion = fit_single_peak_at(
                img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
                weight_template, sigma_init, fix_theta_zero,
            )
            if ion is not None:
                ions.append(ion)
        elif g[0] == "pair":
            assert len(g) == 3
            p1, p2 = g[1], g[2]
            pair = fit_joint_two_peaks_at(
                img, p1[0], p1[1], p2[0], p2[1],
                hw_y, hw_x, s_lo, s_hi, h, w, sigma_init,
            )
            if pair is not None:
                ions.extend(pair)
            else:
                for py, px in (p1, p2):
                    ion = fit_single_peak_at(
                        img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
                        weight_template, sigma_init, fix_theta_zero,
                    )
                    if ion is not None:
                        ions.append(ion)

    return ions
