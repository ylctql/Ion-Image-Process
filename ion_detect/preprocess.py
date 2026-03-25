"""背景/匹配滤波、y 向阈值补偿与峰候选提取。"""
import numpy as np
from scipy.ndimage import maximum_filter

from .boundary import apply_boundary_filter


def build_matched_kernel(sigma_x=1.2, sigma_y=1.8, half_size=3):
    """构造与离子 PSF 匹配的高斯核，用于增强信噪比。"""
    ky, kx = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    kernel = np.exp(-0.5 * (kx**2 / sigma_x**2 + ky**2 / sigma_y**2))
    kernel /= kernel.sum()
    return kernel


def eval_amp_y_model(y_rel: np.ndarray, coef: np.ndarray, mode: str = "even") -> np.ndarray:
    """mode='even': coef=[a0,a2,a4], amp(y)=a0+a2*y^2+a4*y^4
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


def build_row_threshold_scale(
    h: int,
    cy_ref: float,
    coef: np.ndarray,
    mode: str = "even",
    floor: float = 0.2,
) -> np.ndarray:
    """Build row-wise threshold scale in [floor, +inf)."""
    y_rel = np.arange(h, dtype=np.float64) - float(cy_ref)
    amp_env = eval_amp_y_model(y_rel, coef, mode=mode)
    amp_env = np.clip(amp_env, 1e-6, None)
    scale = amp_env / float(np.max(amp_env))
    return np.clip(scale, max(float(floor), 1e-6), None)


def peaks_from_detect_map(
    detect_map: np.ndarray,
    peak_size,
    rel_threshold: float,
    use_y_threshold_comp: bool,
    amp_y_coef,
    amp_y_coef_path,
    amp_y_coef_mode: str,
    comp_floor: float,
    boundary,
):
    """在检测图 detect_map 上做局部极大 + 阈值, 返回 peak_yx (N,2) [row, col]。"""
    h, w = detect_map.shape
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
        row_scale = build_row_threshold_scale(
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
    if boundary is not None:
        peak_yx = apply_boundary_filter(peak_yx, *boundary)
    return peak_yx
