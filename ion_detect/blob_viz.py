"""连通域轴对齐最小外接矩形可视化。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

from .blob_components import MinAreaRect
from .blob_ion_positions import ion_equilibrium_positions_xy

# 与 viz.visualize_* 一致：像素为正方形
_VIS_ASPECT = "equal"


def _brightness_colorbar_label(*, use_bgsub: bool, use_matched_filter: bool) -> str:
    if use_bgsub and use_matched_filter:
        return "matched filter of (raw − Gaussian bg)"
    if use_bgsub:
        return "raw − Gaussian bg"
    if use_matched_filter:
        return "matched filter of raw"
    return "raw intensity"


def _symmetric_abs_scale(z: np.ndarray, pct: float = 99.5) -> float:
    ap = float(np.percentile(np.abs(z), pct))
    if not np.isfinite(ap) or ap < 1e-18:
        ap = max(float(np.nanmax(np.abs(z))) if z.size else 0.0, 1e-12)
    return ap


def _add_rect_patches(
    ax,
    rects: list[MinAreaRect],
    *,
    edgecolor: str = "lime",
    edgecolor_merged: str | None = "darkorange",
    linewidth: float = 1.2,
    zorder: float = 6,
) -> None:
    """绘制轴对齐矩形；``from_edge_merge`` 为 True 时使用 ``edgecolor_merged``（不为 None 时）。"""
    for r in rects:
        ec = (
            edgecolor_merged
            if (edgecolor_merged is not None and getattr(r, "from_edge_merge", False))
            else edgecolor
        )
        ax.add_patch(
            Polygon(
                r.corners_xy,
                closed=True,
                fill=False,
                edgecolor=ec,
                linewidth=linewidth,
                zorder=zorder,
            ),
        )


def _add_ion_equilibrium_markers(
    ax,
    positions: list[tuple[float, float]],
    *,
    color: str = "magenta",
    marker: str = "+",
    markersize: float = 10.0,
    markeredgewidth: float = 1.5,
    zorder: float = 8,
) -> None:
    """离子平衡坐标（矩形中心或 y 向条带内前景质心）。"""
    if not positions:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(
        xs,
        ys,
        linestyle="none",
        marker=marker,
        color=color,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        markeredgecolor=color,
        zorder=zorder,
    )


def _maybe_add_rect_merge_legend(
    ax,
    rects: list[MinAreaRect],
    *,
    edgecolor: str,
    edgecolor_merged: str,
) -> None:
    if not any(getattr(r, "from_edge_merge", False) for r in rects):
        return
    ax.legend(
        handles=[
            Line2D([0], [0], color=edgecolor, lw=2, label="Not merged"),
            Line2D([0], [0], color=edgecolor_merged, lw=2, label="Edge-band merged"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=9,
        framealpha=0.9,
    )


def _make_blob_brightness_distribution_figure(
    float_map_pre_binarize: np.ndarray,
    *,
    boundary: tuple[float, float, float, float] | None,
    rects: list[MinAreaRect],
    binary: np.ndarray,
    title: str = "",
    threshold: float | None = None,
    use_bgsub: bool = True,
    use_matched_filter: bool = False,
    draw_boundary: bool = True,
    rect_edgecolor: str = "lime",
    rect_edgecolor_merged: str = "darkorange",
    rect_linewidth: float = 1.2,
    n_edge_sliver_merges: int = 0,
    rect_y_split: bool = False,
    max_ysize: float = 9.0,
    ion_xy: list[tuple[float, float]] | None = None,
) -> Figure:
    """
    二值化所用浮点图的空域 brightness 分布（与 ``viz.visualize_bgsub_binarized`` 的 bgsub 面板一致：
    bgsub 和/或匹配滤波时用 ``RdBu_r`` + 对称 |z| 分位色标；仅原图时用灰度百分位拉伸）。
    叠加 crystal boundary 与 blob 轴对齐矩形。由调用方 ``plt.show()`` 后关闭。
    """
    z = np.asarray(float_map_pre_binarize, dtype=np.float64)
    h, w = z.shape
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)

    if use_bgsub or use_matched_filter:
        ap = _symmetric_abs_scale(z)
        im = ax.imshow(z, cmap="RdBu_r", aspect=_VIS_ASPECT, vmin=-ap, vmax=ap)
    else:
        lo = float(np.percentile(z, 1))
        hi = float(np.percentile(z, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
        im = ax.imshow(z, cmap="gray", aspect=_VIS_ASPECT, vmin=lo, vmax=hi)

    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.12,
        label=_brightness_colorbar_label(
            use_bgsub=use_bgsub,
            use_matched_filter=use_matched_filter,
        ),
    )

    if draw_boundary and boundary is not None:
        bcx, bcy, ba, bb = boundary
        ax.add_patch(
            Ellipse(
                xy=(bcx, bcy),
                width=2 * ba,
                height=2 * bb,
                angle=0,
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                zorder=5,
            ),
        )
    _add_rect_patches(
        ax,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
        linewidth=rect_linewidth,
    )
    eq_xy = (
        ion_xy
        if ion_xy is not None
        else ion_equilibrium_positions_xy(
            rects,
            binary,
            split=rect_y_split,
            max_ysize=float(max_ysize),
        )
    )
    _add_ion_equilibrium_markers(ax, eq_xy)
    _maybe_add_rect_merge_legend(
        ax,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
    )

    prep_parts: list[str] = []
    if use_bgsub:
        prep_parts.append("bgsub")
    if use_matched_filter:
        prep_parts.append("matched filter")
    prep_s = "; ".join(prep_parts) if prep_parts else "raw"
    thr_s = f"{threshold:g}" if threshold is not None else "?"
    n_ion = len(rects)
    ion_s = f"ions (after merge)={n_ion}"
    if n_edge_sliver_merges > 0:
        ion_s += (
            f"  |  edge merges={n_edge_sliver_merges}  "
            f"rects before merge={n_ion + n_edge_sliver_merges}"
        )
    n_eq = len(eq_xy)
    eq_s = f"  |  eq. positions (+)={n_eq}"
    if rect_y_split:
        eq_s += f"  |  y-split max_ysize={max_ysize:g}"
    ax.set_title(
        f"{title.strip()}   [{prep_s}; binarize ≥ {thr_s}]  |  {ion_s}{eq_s}",
        fontsize=12,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    return fig


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
    rect_edgecolor_merged: str = "darkorange",
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

    _add_rect_patches(
        ax,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
        linewidth=rect_linewidth,
    )
    _maybe_add_rect_merge_legend(
        ax,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
    )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(
        f"{title}  |  ions={len(rects)}".strip(),
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_blob_workflow(
    float_map_pre_binarize: np.ndarray,
    binary: np.ndarray,
    *,
    boundary: tuple[float, float, float, float] | None,
    rects: list[MinAreaRect],
    title: str = "",
    threshold: float | None = None,
    use_bgsub: bool = True,
    use_matched_filter: bool = False,
    output_path: Path | None = None,
    show: bool = False,
    draw_boundary: bool = True,
    rect_edgecolor: str = "lime",
    rect_edgecolor_merged: str = "darkorange",
    rect_linewidth: float = 1.2,
    n_edge_sliver_merges: int = 0,
    rect_y_split: bool = False,
    max_ysize: float = 9.0,
) -> None:
    """
    上：二值化前的浮点图（与 ``run_blob_workflow`` 阈值所用阵列一致）+ 可选 boundary + 矩形；
       带 colorbar（brightness）及 bgsub / matched-filter 是否启用的标注。
    下：二值图 + 同组矩形；上下两栏在 ``boundary`` 非空时均绘制晶格椭圆。

    ``show=True`` 时额外建 **空域 brightness 分布图**（``RdBu_r`` / 横向 colorbar，与 ``ion_detect.viz``
    中 bgsub 二值化附录图一致），随后与该双栏图一并 ``plt.show()``。

    ``rect_y_split=True`` 时，对 y 向跨度大于 ``max_ysize`` 的合并后矩形在 y 方向等分为
    ``ceil(height / max_ysize)`` 条，条带内对二值前景求质心；若条带与**不需 split** 的其它矩形
    AABB 相交，则先去掉落在这些盒子内的像素再求质心。短矩形仍用其几何中心。
    洋红 ``+`` 标出所有平衡位置（上下栏、brightness 图一致）。
    """
    im = np.asarray(float_map_pre_binarize, dtype=np.float64)
    bin_im = np.asarray(binary, dtype=bool)
    if im.shape != bin_im.shape:
        raise ValueError(
            f"float_map shape {im.shape} != binary shape {bin_im.shape}",
        )
    h, w = im.shape
    fig_brightness: Figure | None = None
    ion_xy: list[tuple[float, float]] = ion_equilibrium_positions_xy(
        rects,
        bin_im,
        split=rect_y_split,
        max_ysize=float(max_ysize),
    )

    if show:
        fig_brightness = _make_blob_brightness_distribution_figure(
            im,
            boundary=boundary,
            rects=rects,
            binary=bin_im,
            title=title,
            threshold=threshold,
            use_bgsub=use_bgsub,
            use_matched_filter=use_matched_filter,
            draw_boundary=draw_boundary,
            rect_edgecolor=rect_edgecolor,
            rect_edgecolor_merged=rect_edgecolor_merged,
            rect_linewidth=rect_linewidth,
            n_edge_sliver_merges=n_edge_sliver_merges,
            rect_y_split=rect_y_split,
            max_ysize=max_ysize,
            ion_xy=ion_xy,
        )

    lo = float(np.percentile(im, 1))
    hi = float(np.percentile(im, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(im)), float(np.nanmax(im))
    dmin = float(np.nanmin(im))
    dmax = float(np.nanmax(im))

    fig, axes = plt.subplots(2, 1, figsize=(14, 16), constrained_layout=True)
    ax0, ax1 = axes[0], axes[1]

    im0 = ax0.imshow(im, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.02, label="brightness (linear)")
    prep_lines = [
        f"bgsub: {'on' if use_bgsub else 'off'}",
        f"matched filter: {'on' if use_matched_filter else 'off'}",
        f"display vmin/vmax: {lo:.5g} / {hi:.5g}  (1–99.5 pct)",
        f"data min / max: {dmin:.5g} / {dmax:.5g}",
    ]
    if threshold is not None:
        prep_lines.append(f"threshold T: {threshold:g}")
    n_ion = len(rects)
    n_merged_boxes = sum(1 for r in rects if getattr(r, "from_edge_merge", False))
    prep_lines.append(f"Ion count (after merge): {n_ion}")
    if n_edge_sliver_merges > 0:
        prep_lines.append(f"Edge-band merge operations: {n_edge_sliver_merges}")
        prep_lines.append(f"Bounding boxes before merge: {n_ion + n_edge_sliver_merges}")
    if n_merged_boxes > 0:
        prep_lines.append(f"Edge-merged boxes (orange): {n_merged_boxes}")
    prep_lines.append(f"Ion equilibrium: magenta +  (n={len(ion_xy)})")
    if rect_y_split:
        prep_lines.append(
            f"y-split tall rects: on  (max_ysize={max_ysize:g})",
        )
    else:
        prep_lines.append("y-split tall rects: off  (use rect centers)")

    ax0.text(
        0.02,
        0.98,
        "\n".join(prep_lines),
        transform=ax0.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        family="monospace",
        zorder=10,
    )
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
    _add_rect_patches(
        ax0,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
        linewidth=rect_linewidth,
    )
    _add_ion_equilibrium_markers(ax0, ion_xy)
    _maybe_add_rect_merge_legend(
        ax0,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
    )
    ax0.set_xlim(-0.5, w - 0.5)
    ax0.set_ylim(h - 0.5, -0.5)
    ax0.set_xlabel("x (pixel)")
    ax0.set_ylabel("y (pixel)")
    ax0.set_title(
        f"float map (intensity) + boundary + rects  |  ions={n_ion}  |  eq. (+)={len(ion_xy)}",
    )

    ax1.imshow(bin_im.astype(np.float64), cmap="gray", aspect="equal", vmin=0.0, vmax=1.0)
    if draw_boundary and boundary is not None:
        bcx, bcy, ba, bb = boundary
        ax1.add_patch(
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
    _add_rect_patches(
        ax1,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
        linewidth=rect_linewidth,
    )
    _add_ion_equilibrium_markers(ax1, ion_xy)
    _maybe_add_rect_merge_legend(
        ax1,
        rects,
        edgecolor=rect_edgecolor,
        edgecolor_merged=rect_edgecolor_merged,
    )
    ax1.set_xlim(-0.5, w - 0.5)
    ax1.set_ylim(h - 0.5, -0.5)
    ax1.set_xlabel("x (pixel)")
    ax1.set_ylabel("y (pixel)")
    thr_s = f"{threshold:g}" if threshold is not None else "?"
    ax1.set_title(
        f"binary (T={thr_s}) + boundary + rects  |  ions={n_ion}  |  eq. (+)={len(ion_xy)}",
    )

    fig.suptitle(title.strip(), fontsize=12)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)
    if fig_brightness is not None:
        plt.close(fig_brightness)
