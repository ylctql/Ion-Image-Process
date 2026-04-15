"""连通域轴对齐最小外接矩形可视化。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

from .blob_components import MinAreaRect
from .blob_ion_positions import ion_equilibrium_positions_xy
from .edge_strip import outer_y_edge_strip_masks

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


def _add_y_edge_band_split_lines(
    ax,
    boundary: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    *,
    y_edge_frac: float,
    merge_band_clip_ellipse: bool = True,
    color: str = "gold",
    linewidth: float = 1.2,
    linestyle: str = "--",
    alpha: float = 0.95,
    zorder: float = 6.5,
) -> None:
    """在 y外缘条带与内部区域交界处画水平虚线（``y_below`` / ``y_above``，弦向 ``|x-cx|<=x_half``）。"""
    h, w = int(image_shape[0]), int(image_shape[1])
    _top, _bot, meta = outer_y_edge_strip_masks(
        boundary,
        float(y_edge_frac),
        (h, w),
        clip_ellipse=merge_band_clip_ellipse,
    )
    cx = float(meta["cx"])
    y_below = float(meta["y_below"])
    y_above = float(meta["y_above"])
    x_half = float(meta["x_half"])
    if x_half <= 1e-9:
        x_lo, x_hi = 0.0, float(max(0, w - 1))
    else:
        x_lo = cx - x_half
        x_hi = cx + x_half
    kw = dict(
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.plot([x_lo, x_hi], [y_below, y_below], **kw)
    ax.plot([x_lo, x_hi], [y_above, y_above], **kw)


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
    refine_x: bool = False,
    x_profile_threshold: float = 0.5,
    x_profile_rel_to_max: bool = False,
    ion_xy: list[tuple[float, float]] | None = None,
    labeled: np.ndarray | None = None,
    y_edge_frac: float | None = None,
    merge_band_clip_ellipse: bool = True,
) -> Figure:
    """
    二值化所用浮点图的空域 brightness 分布（与 ``viz.visualize_bgsub_binarized`` 的 bgsub 面板一致：
    bgsub 和/或匹配滤波时用 ``RdBu_r`` + 对称 |z| 分位色标；仅原图时用灰度百分位拉伸）。
    叠加 crystal boundary 与 blob 轴对齐矩形。由调用方 ``plt.show()`` 后关闭。

    ``y_edge_frac`` 非空且 ``boundary`` 非空时，在上图叠加外缘条带分界水平虚线（与 ``outer_y_edge_strip_masks`` 一致）。
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
    if boundary is not None and y_edge_frac is not None:
        _add_y_edge_band_split_lines(
            ax,
            boundary,
            (h, w),
            y_edge_frac=float(y_edge_frac),
            merge_band_clip_ellipse=merge_band_clip_ellipse,
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
            labeled=labeled,
            split=rect_y_split,
            max_ysize=float(max_ysize),
            refine_x=refine_x,
            x_profile_threshold=float(x_profile_threshold),
            x_profile_rel_to_max=x_profile_rel_to_max,
            intensity=z,
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
    if refine_x:
        eq_s += f"  |  x-refine thr={x_profile_threshold:g}"
        if x_profile_rel_to_max:
            eq_s += " (rel max)"
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
    refine_x: bool = False,
    x_profile_threshold: float = 0.5,
    x_profile_rel_to_max: bool = False,
    labeled: np.ndarray | None = None,
    y_edge_frac: float | None = 0.3,
    merge_band_clip_ellipse: bool = True,
    ion_xy: list[tuple[float, float]] | None = None,
    thr_norm: Literal["none", "p95", "p95_all"] = "none",
    thr_norm_percentile: float | None = None,
    thr_norm_scale: float | None = None,
) -> None:
    """
    上：二值化前的浮点图（与 ``run_blob_workflow`` 阈值所用阵列一致）+ 可选 boundary + 矩形；
       带 colorbar（brightness）及 bgsub / matched-filter 是否启用的标注。
       若 ``boundary`` 与 ``y_edge_frac`` 非空，在上图用金色水平虚线标出外缘条带与内部的 y 分界
       （与 ``outer_y_edge_strip_masks`` / edge merge 几何一致）。
    下：二值图 + 同组矩形；上下两栏在 ``boundary`` 非空时均绘制晶格椭圆。

    ``show=True`` 时额外建 **空域 brightness 分布图**（``RdBu_r`` / 横向 colorbar，与 ``ion_detect.viz``
    中 bgsub 二值化附录图一致），随后与该双栏图一并 ``plt.show()``。

    ``rect_y_split=True`` 时，对 y 向跨度大于 ``max_ysize`` 的矩形在 y 方向等分为
    ``ceil(height / max_ysize)`` 条，条带内对前景求质心。若传入 ``labeled``（连通域标号图），
    则每条条带仅使用该矩形 ``component_labels`` 对应标签的像素，避免 AABB 重叠处混入其它域。
    未传 ``labeled`` 时行为同旧版（矩形内全部二值前景；并可剔除落在短矩形 AABB 内的像素）。
    短矩形仍用其几何中心。洋红 ``+`` 标出所有平衡位置（上下栏、brightness 图一致）。

    ``ion_xy`` 非空时直接使用该列表绘制 ``+``（例如 CLI 在 ``ion_equilibrium_positions_xy`` 之后做过近距合并），
    不再调用 ``ion_equilibrium_positions_xy``。
    """
    im = np.asarray(float_map_pre_binarize, dtype=np.float64)
    bin_im = np.asarray(binary, dtype=bool)
    if im.shape != bin_im.shape:
        raise ValueError(
            f"float_map shape {im.shape} != binary shape {bin_im.shape}",
        )
    h, w = im.shape
    fig_brightness: Figure | None = None
    if ion_xy is None:
        ion_xy = ion_equilibrium_positions_xy(
            rects,
            bin_im,
            labeled=labeled,
            split=rect_y_split,
            max_ysize=float(max_ysize),
            refine_x=refine_x,
            x_profile_threshold=float(x_profile_threshold),
            x_profile_rel_to_max=x_profile_rel_to_max,
            intensity=im,
        )
    else:
        ion_xy = list(ion_xy)

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
            refine_x=refine_x,
            x_profile_threshold=x_profile_threshold,
            x_profile_rel_to_max=x_profile_rel_to_max,
            ion_xy=ion_xy,
            labeled=labeled,
            y_edge_frac=y_edge_frac,
            merge_band_clip_ellipse=merge_band_clip_ellipse,
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
    cbar_label = (
        "z / P(ROI signed) scale (threshold map)"
        if thr_norm == "p95_all"
        else (
            "z / P(+) scale (threshold map)"
            if thr_norm == "p95"
            else "brightness (linear)"
        )
    )
    fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.02, label=cbar_label)
    prep_lines = [
        f"bgsub: {'on' if use_bgsub else 'off'}",
        f"matched filter: {'on' if use_matched_filter else 'off'}",
        f"display vmin/vmax: {lo:.5g} / {hi:.5g}  (1–99.5 pct)",
        f"data min / max: {dmin:.5g} / {dmax:.5g}",
    ]
    if thr_norm == "p95":
        ptxt = (
            f"{thr_norm_percentile:g}"
            if thr_norm_percentile is not None
            else "?"
        )
        stxt = (
            f"{thr_norm_scale:g}"
            if thr_norm_scale is not None
            else "?"
        )
        prep_lines.append(
            f"thr norm: P({ptxt})(+) in ellipse ROI; scale = {stxt}",
        )
    elif thr_norm == "p95_all":
        ptxt = (
            f"{thr_norm_percentile:g}"
            if thr_norm_percentile is not None
            else "?"
        )
        stxt = (
            f"{thr_norm_scale:g}"
            if thr_norm_scale is not None
            else "?"
        )
        prep_lines.append(
            f"thr norm: P({ptxt})(signed all finite in ellipse ROI); scale = {stxt}",
        )
    if threshold is not None:
        tnote = (
            " (on normalized z)"
            if thr_norm in ("p95", "p95_all")
            else ""
        )
        prep_lines.append(f"threshold T: {threshold:g}{tnote}")
    if boundary is not None and y_edge_frac is not None:
        clip_s = "ellipse-clipped" if merge_band_clip_ellipse else "rect strip"
        prep_lines.append(
            f"y_edge_frac F={y_edge_frac:g}: gold -- = outer-edge band boundary ({clip_s})",
        )
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
    if refine_x:
        if x_profile_rel_to_max:
            prep_lines.append(
                f"x-refine: col_mean > {x_profile_threshold:g} * max(col) (on)",
            )
        else:
            prep_lines.append(
                f"x-refine (column mean > {x_profile_threshold:g}): on",
            )

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
    if boundary is not None and y_edge_frac is not None:
        _add_y_edge_band_split_lines(
            ax0,
            boundary,
            (h, w),
            y_edge_frac=float(y_edge_frac),
            merge_band_clip_ellipse=merge_band_clip_ellipse,
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
    top_title = (
        "float map (P(signed ROI)-normalized for T) + boundary + rects"
        if thr_norm == "p95_all"
        else (
            "float map (P(+) ROI-normalized for T) + boundary + rects"
            if thr_norm == "p95"
            else "float map (intensity) + boundary + rects"
        )
    )
    ax0.set_title(
        f"{top_title}  |  ions={n_ion}  |  eq. (+)={len(ion_xy)}",
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
    bin_note = "norm " if thr_norm in ("p95", "p95_all") else ""
    ax1.set_title(
        f"binary ({bin_note}T={thr_s}) + boundary + rects  |  ions={n_ion}  |  eq. (+)={len(ion_xy)}",
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
