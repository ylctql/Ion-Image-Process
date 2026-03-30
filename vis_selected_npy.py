"""
将 20260305_1727（或任意目录）中指定的 .npy 矩阵可视化并保存为 PNG。

用法示例:
  python vis_selected_npy.py 20260305_005542 20260305_010000
  python vis_selected_npy.py --dir 20260305_1727 20260305_005542
  python vis_selected_npy.py --one-figure --cmap inferno 20260305_005542 20260305_010000
  # (171,1024) 数据：仅对行方向插值上采样 4 倍
  python vis_selected_npy.py --zoom-axes 4,1 20260305_005542
  # 仅弹出窗口显示，不写 PNG
  python vis_selected_npy.py --show 20260305_005542
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import zoom

from output_paths import DEFAULT_DATA_DIR, OUT_NPY_SELECTED, default_vis_selected_montage_png


def _resolve_npy_path(data_dir: Path, name: str) -> Path:
    """接受 stem（无扩展名）或完整文件名。"""
    p = Path(name)
    if p.suffix.lower() == ".npy":
        candidate = data_dir / p.name if not p.is_absolute() else p
    else:
        candidate = data_dir / f"{p.name}.npy"
    if not candidate.is_file():
        raise FileNotFoundError(f"找不到文件: {candidate}")
    return candidate


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _prepare_array(
    arr: np.ndarray,
    zoom_axes: tuple[float, float],
    zoom_order: int,
) -> np.ndarray:
    """对矩阵逐轴放大采样（插值），不改变数值范围，只增加栅格点数。"""
    z0, z1 = zoom_axes
    if z0 == 1.0 and z1 == 1.0:
        return arr
    return zoom(arr, (z0, z1), order=zoom_order)


def plot_separate_files(
    paths: list[Path],
    out_dir: Path | None,
    cmap: str,
    dpi: int,
    share_scale: bool,
    zoom_axes: tuple[float, float],
    zoom_order: int,
    imshow_interpolation: str,
    show_only: bool,
) -> None:
    if not show_only:
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
    arrays = [_prepare_array(np.load(p), zoom_axes, zoom_order) for p in paths]
    if share_scale:
        vmin = min(a.min() for a in arrays)
        vmax = max(a.max() for a in arrays)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for p, arr in zip(paths, arrays):
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(
            arr,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation=imshow_interpolation,
        )
        ax.set_title(p.stem, fontsize=12)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Intensity")
        fig.tight_layout()
        if show_only:
            continue
        assert out_dir is not None
        out = out_dir / f"selected_{p.stem}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"[已保存] {out}")

    if show_only:
        plt.show()


def plot_one_figure(
    paths: list[Path],
    out_path: Path | None,
    cmap: str,
    dpi: int,
    share_scale: bool,
    zoom_axes: tuple[float, float],
    zoom_order: int,
    imshow_interpolation: str,
    show_only: bool,
) -> None:
    arrays = [_prepare_array(np.load(p), zoom_axes, zoom_order) for p in paths]
    n = len(arrays)
    rows, cols = _grid_shape(n)
    if share_scale:
        vmin = min(a.min() for a in arrays)
        vmax = max(a.max() for a in arrays)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes_flat = np.atleast_1d(axes).ravel()

    last_im = None
    for i, (p, arr) in enumerate(zip(paths, arrays)):
        ax = axes_flat[i]
        last_im = ax.imshow(
            arr,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation=imshow_interpolation,
        )
        ax.set_title(p.stem, fontsize=10)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if last_im is None:
        raise RuntimeError("内部错误：没有可绘制的数组")
    fig.suptitle("Selected frames", fontsize=14, y=1.02)
    fig.colorbar(last_im, ax=axes_flat.tolist(), shrink=0.6, label="Intensity")
    fig.tight_layout()
    if show_only:
        plt.show()
        return
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="可视化指定目录中的 .npy 文件（默认 20260305_1727）"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="文件名或 stem，例如 20260305_005542 或 20260305_005542.npy",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"数据目录（默认: {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--one-figure",
        action="store_true",
        help="将所有帧画在一张子图网格中",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="matplotlib colormap（默认 viridis）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "保存 PNG 时的栅格化分辨率（每英寸点数）；与 figsize 一起决定输出像素多少。"
            "不会增加 .npy 里的采样点数，只是把整张图画得更细。"
        ),
    )
    parser.add_argument(
        "--zoom-axes",
        metavar="R_ROW,R_COL",
        default=None,
        help=(
            "对数组按轴上采样倍数（scipy.ndimage.zoom），例如 4,1 表示第 0 轴(行)×4、"
            "第 1 轴(列)不变；常见形状 (171,1024) 时若觉得“短边”太稀，可对行方向加大 R_ROW。"
        ),
    )
    parser.add_argument(
        "--zoom-order",
        type=int,
        choices=[0, 1, 3],
        default=1,
        help="zoom 插值：0 最近邻，1 线性（默认），3 三次",
    )
    parser.add_argument(
        "--interpolation",
        default="antialiased",
        help=(
            "imshow 在像素边界上的绘制方式（默认 antialiased）；"
            "可改为 nearest 保持清晰方块像素。"
        ),
    )
    parser.add_argument(
        "--per-file-scale",
        action="store_true",
        help="每帧独立色标；默认多帧时共用 min/max 便于对比",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="仅弹出 matplotlib 窗口显示，不保存 PNG（关闭所有窗口后程序结束）",
    )
    args = parser.parse_args()

    data_dir = args.dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")

    paths = [_resolve_npy_path(data_dir, name) for name in args.files]
    share_scale = not args.per_file_scale and len(paths) > 1
    OUT_NPY_SELECTED.mkdir(parents=True, exist_ok=True)

    if args.zoom_axes is None:
        zoom_axes = (1.0, 1.0)
    else:
        raw = args.zoom_axes.replace("，", ",").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise SystemExit("--zoom-axes 需要两个数，用英文逗号分隔，例如 4,1")
        zoom_axes = (float(parts[0]), float(parts[1]))

    if args.one_figure:
        if args.show:
            plot_one_figure(
                paths,
                None,
                args.cmap,
                args.dpi,
                share_scale,
                zoom_axes,
                args.zoom_order,
                args.interpolation,
                show_only=True,
            )
        else:
            plot_one_figure(
                paths,
                default_vis_selected_montage_png(),
                args.cmap,
                args.dpi,
                share_scale,
                zoom_axes,
                args.zoom_order,
                args.interpolation,
                show_only=False,
            )
    else:
        if args.show:
            plot_separate_files(
                paths,
                None,
                args.cmap,
                args.dpi,
                share_scale,
                zoom_axes,
                args.zoom_order,
                args.interpolation,
                show_only=True,
            )
        else:
            plot_separate_files(
                paths,
                OUT_NPY_SELECTED,
                args.cmap,
                args.dpi,
                share_scale,
                zoom_axes,
                args.zoom_order,
                args.interpolation,
                show_only=False,
            )


if __name__ == "__main__":
    main()
