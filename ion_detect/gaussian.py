"""2D 高斯模型与剥离用核叠加。"""
import numpy as np


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


def _ion_gaussian_core(ion: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """无局部 offset 的 2D 高斯正值核, 与单峰拟合使用的模型一致。

    轴对齐拟合 (θ=0) 在结果中带 _sigma_x/_sigma_y; 一般椭圆用 sigma_minor/sigma_major 与 theta_deg。
    """
    amp = float(ion["amplitude"])
    x0 = float(ion["x0"])
    y0 = float(ion["y0"])
    if "_sigma_x" in ion:
        sx = float(ion["_sigma_x"])
        sy = float(ion["_sigma_y"])
        return amp * np.exp(
            -0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)
        )
    smin = float(ion["sigma_minor"])
    smaj = float(ion["sigma_major"])
    th = float(ion["theta_deg"])
    return _gauss2d(
        (x, y), amp, x0, y0, smin, smaj, np.radians(th), 0.0,
    )


def _ion_support_bbox(ion: dict, h: int, w: int, margin_sigma: float):
    """用于累加剥离模型的 ROI (各向外扩 margin_sigma * σ_major)。"""
    x0 = float(ion["x0"])
    y0 = float(ion["y0"])
    r = int(np.ceil(margin_sigma * float(ion["sigma_major"]))) + 2
    y1 = max(0, int(np.floor(y0 - r)))
    y2 = min(h, int(np.ceil(y0 + r)) + 1)
    x1 = max(0, int(np.floor(x0 - r)))
    x2 = min(w, int(np.ceil(x0 + r)) + 1)
    return y1, y2, x1, x2


def _accumulate_peel_model(
    h: int, w: int, ions: list, margin_sigma: float = 4.5,
) -> np.ndarray:
    """将已检出离子的高斯核叠加成一张与图像同形状的剥离模型。"""
    model = np.zeros((h, w), dtype=np.float64)
    for ion in ions:
        y1, y2, x1, x2 = _ion_support_bbox(ion, h, w, margin_sigma)
        if y1 >= y2 or x1 >= x2:
            continue
        grid = np.mgrid[y1:y2, x1:x2].astype(np.float64)
        yy, xx = grid[0], grid[1]
        model[y1:y2, x1:x2] += _ion_gaussian_core(ion, xx, yy)
    return model
