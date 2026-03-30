"""连通域轴对齐最小外接矩形可视化。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

from .blob_components import MinAreaRect


def _add_rect_patches(
    ax,
    rects: list[MinAreaRect],
    *,
    edgecolor: str = "lime",
    linewidth: float = 1.2,
    zorder: float = 6,
) -> None:
    for r in rects:
        ax.add_patch(
            Polygon(
                r.corners_xy,
                closed=True,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            ),
        )


def visualize_blob_rects(
    image: np.ndarray,
    *,
    boundary: tuple[float, float, float, float] | None,
    rects: list[MinAreaRect],
    title: str = "",
    output_path: Path | None = None,
    show: bool = False,
    draw_boundary: bool = True,
    rect_edgecolor: str = "lime",
    rect_linewidth: float = 1.2,
    rect_facecolor: str = "none",
) -> None:
    """
    灰度底图 + 可选晶格椭圆 + 各连通域轴对齐外接矩形（边平行于 x/y）。
    """
    _ = rect_facecolor  # API 兼容；Polygon 仅描边
    im = np.asarray(image, dtype=np.float64)
    h, w = im.shape
    lo = float(np.percentile(im, 1))
    hi = float(np.percentile(im, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(im)), float(np.nanmax(im))

    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    ax.imshow(im, cmap="gray", aspect="equal", vmin=lo, vmax=hi)

    if draw_boundary and boundary is not None:
        bcx, bcy, ba, bb = boundary
        ell = Ellipse(
            xy=(bcx, bcy),
            width=2 * ba,
            height=2 * bb,
            angle=0,
            edgecolor="cyan",
            facecolor="none",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(ell)

    _add_rect_patches(ax, rects, edgecolor=rect_edgecolor, linewidth=rect_linewidth)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(f"{title}  |  blobs={len(rects)}".strip())
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_blob_workflow(
    image: np.ndarray,
    binary: np.ndarray,
    *,
    boundary: tuple[float, float, float, float] | None,
    rects: list[MinAreaRect],
    title: str = "",
    output_path: Path | None = None,
    show: bool = False,
    draw_boundary: bool = True,
    rect_edgecolor: str = "lime",
    rect_linewidth: float = 1.2,
) -> None:
    """
    上：原始灰度 + 可选 boundary + 轴对齐矩形；
    下：二值化图 + 同组轴对齐矩形。
    """
    im = np.asarray(image, dtype=np.float64)
    bin_im = np.asarray(binary, dtype=bool)
    if im.shape != bin_im.shape:
        raise ValueError(f"image shape {im.shape} != binary shape {bin_im.shape}")
    h, w = im.shape

    lo = float(np.percentile(im, 1))
    hi = float(np.percentile(im, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(im)), float(np.nanmax(im))

    fig, axes = plt.subplots(2, 1, figsize=(14, 16), constrained_layout=True)
    ax0, ax1 = axes[0], axes[1]

    ax0.imshow(im, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    if draw_boundary and boundary is not None:
        bcx, bcy, ba, bb = boundary
        ax0.add_patch(
            Ellipse(
                xy=(bcx, bcy),
                width=2 * ba,
                height=2 * bb,
                angle=0,
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                zorder=5,
            ),
        )
    _add_rect_patches(ax0, rects, edgecolor=rect_edgecolor, linewidth=rect_linewidth)
    ax0.set_xlim(-0.5, w - 0.5)
    ax0.set_ylim(h - 0.5, -0.5)
    ax0.set_xlabel("x (pixel)")
    ax0.set_ylabel("y (pixel)")
    ax0.set_title(f"grayscale + boundary + rects  |  blobs={len(rects)}")

    ax1.imshow(bin_im.astype(np.float64), cmap="gray", aspect="equal", vmin=0.0, vmax=1.0)
    _add_rect_patches(ax1, rects, edgecolor=rect_edgecolor, linewidth=rect_linewidth)
    ax1.set_xlim(-0.5, w - 0.5)
    ax1.set_ylim(h - 0.5, -0.5)
    ax1.set_xlabel("x (pixel)")
    ax1.set_ylabel("y (pixel)")
    ax1.set_title("binary + rects")

    fig.suptitle(title.strip(), fontsize=12)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)
