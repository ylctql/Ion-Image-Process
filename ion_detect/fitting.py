"""候选峰高斯拟合与两阶段精修。"""
import numpy as np
from scipy.optimize import curve_fit

from .gaussian import _gauss2d, _gauss2d_aligned


def weighted_r2(y, yhat, sigma):
    """与 ``curve_fit(..., sigma=...)`` 一致的加权 R²。

    权重 w_i = 1/σ_i²，ȳ = Σ w y / Σ w，
    R²_w = 1 - Σ w (y-ŷ)² / Σ w (y-ȳ)²。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if y.shape != yhat.shape or y.shape != sigma.shape:
        return float("nan")
    w = 1.0 / np.clip(sigma ** 2, 1e-30, None)
    sw = float(np.sum(w))
    if sw <= 0.0 or not np.isfinite(sw):
        return float("nan")
    y_bar = float(np.sum(w * y) / sw)
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - y_bar) ** 2))
    if ss_tot <= 1e-30 or not np.isfinite(ss_tot):
        return float("nan")
    r2 = 1.0 - ss_res / ss_tot
    return float(r2) if np.isfinite(r2) else float("nan")


def fit_all_peaks(img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
                  h, w, refine=True, fix_theta_zero=True):
    """对所有候选峰做高斯拟合, 可选两阶段精修。"""
    wy, wx = 2 * hw_y + 1, 2 * hw_x + 1
    gy_t, gx_t = np.mgrid[0:wy, 0:wx]
    cy_t, cx_t = hw_y, hw_x
    weight_template = np.exp(-0.5 * ((gx_t - cx_t)**2 / (hw_x * 0.7)**2
                                   + (gy_t - cy_t)**2 / (hw_y * 0.45)**2))

    ions = do_fit_pass(
        img, peak_yx, hw_y, hw_x, s_lo, s_hi,
        h, w, weight_template, sigma_init=(1.2, 1.8),
        fix_theta_zero=fix_theta_zero,
    )

    if not refine or len(ions) < 20:
        return ions

    majors = np.array([d["sigma_major"] for d in ions])
    minors = np.array([d["sigma_minor"] for d in ions])
    stable = (majors < s_hi * 0.9) & (minors < s_hi * 0.9)
    if stable.sum() < 10:
        return ions

    ref_minor = float(np.median(minors[stable]))
    ref_major = float(np.median(majors[stable]))

    needs_refit = (majors > ref_major * 1.8) | (minors > ref_minor * 2.0)
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
            yhat = _gauss2d_aligned((xx.ravel(), yy.ravel()), *popt)
            r2_w = weighted_r2(patch.ravel(), yhat, fit_sigma.ravel())
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
            yhat = _gauss2d((xx.ravel(), yy.ravel()), *popt)
            r2_w = weighted_r2(patch.ravel(), yhat, fit_sigma.ravel())
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
        "r2_weighted": r2_w,
        "_py": py, "_px": px,
    }
    if fix_theta_zero:
        out["_sigma_x"] = float(sx)
        out["_sigma_y"] = float(sy)
    return out


def do_fit_pass(img, peak_yx, hw_y, hw_x, s_lo, s_hi,
                h, w, weight_template, sigma_init=(1.2, 1.8),
                fix_theta_zero=True):
    """单次高斯拟合：各候选峰独立单峰。"""
    ions = []
    for r in np.atleast_2d(peak_yx):
        py, px = int(r[0]), int(r[1])
        ion = fit_single_peak_at(
            img, py, px, hw_y, hw_x, s_lo, s_hi, h, w,
            weight_template, sigma_init, fix_theta_zero,
        )
        if ion is not None:
            ions.append(ion)
    return ions
