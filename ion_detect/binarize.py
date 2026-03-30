"""减高斯背景后 signal (bgsub) 的二值化。"""
from __future__ import annotations

import numpy as np


def bgsub_binarize(bgsub: np.ndarray, threshold: float, *, ge: bool = True) -> np.ndarray:
    """
    对噪声减除图 ``bgsub``（与 ``detect_ions`` 中首轮 ``signal = image - Gaussian_bg`` 同量纲）做二值化。

    Parameters
    ----------
    bgsub : ndarray (H, W), 浮点
        减背景后的有符号信号。
    threshold : float
        阈值；默认前景为 ``bgsub >= threshold``。
    ge : bool
        若为 True，前景为 ``>= threshold``；若为 False，为 ``> threshold``。

    Returns
    -------
    binary : ndarray (H, W), bool
        前景为 True，背景为 False。
    """
    z = np.asarray(bgsub, dtype=np.float64)
    if ge:
        return z >= float(threshold)
    return z > float(threshold)


def bgsub_binarize_u8(bgsub: np.ndarray, threshold: float, *, ge: bool = True) -> np.ndarray:
    """与 :func:`bgsub_binarize` 相同规则，返回 ``uint8`` 的 0 / 255 图（便于保存 PNG）。"""
    m = bgsub_binarize(bgsub, threshold, ge=ge)
    return (m.astype(np.uint8) * np.uint8(255))
