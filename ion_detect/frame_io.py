"""从数据目录枚举帧文件并加载为 ``detect_ions`` 所需的二维 float 数组。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

__all__ = [
    "RASTER_SUFFIXES",
    "is_supported_frame_path",
    "list_frame_files",
    "load_frame",
    "normalize_to_2d_float",
]

# 与 npy 并列扫描的常见栅格格式（小写后缀比较）
RASTER_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
)


def normalize_to_2d_float(arr: np.ndarray) -> np.ndarray:
    """将 ``(H,W)`` 或 ``(H,W,C)`` 数组转为 ``(H,W)`` float64。"""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 3:
        return arr.mean(axis=-1).astype(np.float64, copy=False)
    raise ValueError(f"不支持的数组维度: {arr.shape}")


def is_supported_frame_path(path: Path) -> bool:
    """是否为支持的帧文件：``.npy`` 或栅格图像（如 ``.jpg`` / ``.png``）。"""
    suf = path.suffix.lower()
    return suf == ".npy" or suf in RASTER_SUFFIXES


def list_frame_files(data_dir: Path) -> list[Path]:
    """
    列出目录下所有支持的帧文件，按文件名排序。

    同时包含 ``.npy`` 与栅格图像；若需仅使用其中一类，请自行过滤路径列表。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"不是目录: {data_dir}")
    out = [p for p in data_dir.iterdir() if p.is_file() and is_supported_frame_path(p)]
    return sorted(out)


def load_frame(path: Path) -> np.ndarray:
    """
    加载单帧为 ``(H, W)`` float64，供 ``detect_ions`` 使用。

    - ``.npy``：与现有脚本一致，若为三维则对通道取平均。
    - 栅格图像：经 ``matplotlib.pyplot.imread`` 读取（依赖 Pillow 以支持 JPEG 等）。
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        return normalize_to_2d_float(np.load(path))
    if suf in RASTER_SUFFIXES:
        raw = plt.imread(path)
        return normalize_to_2d_float(raw)
    raise ValueError(f"不支持的帧文件类型: {path}")
