"""可视化与结果摘要。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .peel import y_edge_band_thresholds


def visualize(image, ions, n_sigma=2.0, title="", output_path=None,
              show_zoom=True, zoom_center=None, zoom_size=80,
              boundary=None):
    """
    在图像上标注拟合椭圆。

    n_sigma 控制椭圆的半轴长度 (n_sigma * sigma)。
    boundary: (cx, cy, a, b) 晶格边界椭圆参数, 若提供则绘制在图上。
    可选显示一个局部放大区域。
    zoom_center / zoom_size 保留为兼容参数, 当前未使用 (放大区域为内置固定列表)。
    """
    _ = (zoom_center, zoom_size)
    nrows = 6 if show_zoom else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(20, 5 + (3 * 5 if show_zoom else 0)),
                             gridspec_kw={"height_ratios": [5, 2, 3, 3, 3, 2]
                                          if show_zoom else [1]})
    if not show_zoom:
        axes = [axes]

    ax = axes[0]
    ax.imshow(image, cmap="gray", aspect="auto", vmin=np.percentile(image, 1),
              vmax=np.percentile(image, 99.5))
    for ion in ions:
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor="red", facecolor="none", linewidth=0.4, alpha=0.8,
        )
        ax.add_patch(ell)
    if boundary is not None:
        bcx, bcy, ba, bb = boundary
        bnd_ell = Ellipse(
            xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
            edgecolor="cyan", facecolor="none",
            linewidth=1.2, linestyle="--", alpha=0.9,
        )
        ax.add_patch(bnd_ell)
    ax.set_title(
        f"{title}   [{len(ions)} ions, ellipse = {n_sigma} sigma]",
        fontsize=13,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    if show_zoom:
        regions = [
            ("Top edge",    (500, 35),  100, 20),
            ("Left",        (200, 75),   60, 30),
            ("Center",      (500, 85),   60, 30),
            ("Right",       (800, 75),   60, 30),
            ("Bottom edge", (500, 130), 100, 20),
        ]
        for i, (label, (rcx, rcy), rzs_x, rzs_y) in enumerate(regions):
            ax2 = axes[1 + i]
            x1z = max(0, rcx - rzs_x)
            x2z = min(image.shape[1], rcx + rzs_x)
            y1z = max(0, rcy - rzs_y)
            y2z = min(image.shape[0], rcy + rzs_y)

            ax2.imshow(image, cmap="gray", aspect="equal",
                       vmin=np.percentile(image, 1),
                       vmax=np.percentile(image, 99.5))
            for ion in ions:
                if x1z <= ion["x0"] <= x2z and y1z <= ion["y0"] <= y2z:
                    ell = Ellipse(
                        xy=(ion["x0"], ion["y0"]),
                        width=2 * n_sigma * ion["sigma_minor"],
                        height=2 * n_sigma * ion["sigma_major"],
                        angle=ion["theta_deg"],
                        edgecolor="lime", facecolor="none", linewidth=1.2,
                    )
                    ax2.add_patch(ell)
                    ax2.plot(ion["x0"], ion["y0"], "r.", markersize=2)
            if boundary is not None:
                bcx, bcy, ba, bb = boundary
                bnd_ell = Ellipse(
                    xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
                    edgecolor="cyan", facecolor="none",
                    linewidth=1.5, linestyle="--", alpha=0.9,
                )
                ax2.add_patch(bnd_ell)
            ax2.set_xlim(x1z, x2z)
            ax2.set_ylim(y2z, y1z)
            ax2.set_title(
                f"Zoom: {label}  x=[{x1z},{x2z}]  y=[{y1z},{y2z}]",
                fontsize=11,
            )
            ax2.set_xlabel("x (pixel)")
            ax2.set_ylabel("y (pixel)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[已保存] {output_path}")
    plt.close(fig)


def visualize_peel_residual(
    residual,
    title="",
    output_path=None,
    boundary=None,
    reference_image=None,
    peak_peel_y_edges_only=False,
    peak_peel_y_edge_frac=0.25,
):
    """Peak-peel residual (raw image minus first-round Gaussian sum).

    If ``reference_image`` is set, ``vmin``/``vmax`` match ``visualize``:
    1st and 99.5th percentile of the raw frame so gray levels are comparable
    with the ion ellipse PNG (values outside range clip to black/white).
    If ``None``, uses percentiles of ``residual`` only (legacy behavior).

    When ``boundary`` is set, draws semi-transparent y-edge bands for
    ``|y-cy|/b >= 1 - peak_peel_y_edge_frac`` (same as
    ``filter_peak_yx_y_edge_bands``). Round-2 actually restricts to these bands
    only if ``peak_peel_y_edges_only`` is enabled in the detection run.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if reference_image is not None:
        ref = np.asarray(reference_image, dtype=np.float64)
        lo = float(np.percentile(ref, 1))
        hi = float(np.percentile(ref, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.nanmin(ref))
            hi = float(np.nanmax(ref))
            if lo >= hi:
                lo, hi = lo - 1.0, hi + 1.0
    else:
        lo = float(np.percentile(residual, 1))
        hi = float(np.percentile(residual, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(residual)), float(np.nanmax(residual))
            if lo >= hi:
                lo, hi = lo - 1.0, hi + 1.0
    h, w = int(residual.shape[0]), int(residual.shape[1])
    ax.imshow(residual, cmap="gray", aspect="equal", vmin=lo, vmax=hi)

    y_img_max = float(max(h - 1, 0))
    if boundary is not None:
        edges = y_edge_band_thresholds(boundary, peak_peel_y_edge_frac)
        if edges is not None:
            y_below, y_above = edges
            if y_below > 0.0:
                y_top_band_end = min(y_below, y_img_max)
                if y_top_band_end > 0.0:
                    ax.axhspan(
                        0.0,
                        y_top_band_end,
                        facecolor="darkmagenta",
                        alpha=0.14,
                        edgecolor="none",
                        zorder=2,
                    )
                    ax.axhline(
                        y_top_band_end,
                        color="darkmagenta",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.85,
                        zorder=4,
                    )
            if y_above < y_img_max:
                y_bot_band_start = max(y_above, 0.0)
                if y_bot_band_start < y_img_max:
                    ax.axhspan(
                        y_bot_band_start,
                        y_img_max,
                        facecolor="darkmagenta",
                        alpha=0.14,
                        edgecolor="none",
                        zorder=2,
                    )
                    ax.axhline(
                        y_bot_band_start,
                        color="darkmagenta",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.85,
                        zorder=4,
                    )

    if boundary is not None:
        bcx, bcy, ba, bb = boundary
        bnd_ell = Ellipse(
            xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
            edgecolor="cyan", facecolor="none",
            linewidth=1.2, linestyle="--", alpha=0.9, zorder=6,
        )
        ax.add_patch(bnd_ell)

    subtitle_parts = ["peak-peel residual"]
    if reference_image is not None:
        subtitle_parts.append("intensity scale = raw 1–99.5 pctl")
    if boundary is not None:
        fe = f"{float(peak_peel_y_edge_frac):g}"
        if peak_peel_y_edges_only:
            subtitle_parts.append(
                f"magenta = y-edge bands (|y-cy|/b>=1-F, F={fe}); round-2 filter ON"
            )
        else:
            subtitle_parts.append(
                f"magenta = y-edge bands (F={fe}); round-2 uses full frame"
            )
    ax.set_title(f"{title}  [{'; '.join(subtitle_parts)}]", fontsize=13)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[已保存残差图] {output_path}")
    plt.close(fig)


def print_summary(ions):
    if not ions:
        print("未检测到离子。")
        return
    minors = np.array([d["sigma_minor"] for d in ions])
    majors = np.array([d["sigma_major"] for d in ions])
    amps   = np.array([d["amplitude"]   for d in ions])
    ratios = majors / minors

    print(f"\n检测结果: {len(ions)} 个离子")
    print(f"  σ_minor  (短轴): mean={minors.mean():.2f} ± {minors.std():.2f}  "
          f"range=[{minors.min():.2f}, {minors.max():.2f}]")
    print(f"  σ_major  (长轴): mean={majors.mean():.2f} ± {majors.std():.2f}  "
          f"range=[{majors.min():.2f}, {majors.max():.2f}]")
    print(f"  长短轴比 (major/minor): mean={ratios.mean():.2f} ± {ratios.std():.2f}")
    print(f"  振幅: mean={amps.mean():.1f} ± {amps.std():.1f}  "
          f"range=[{amps.min():.1f}, {amps.max():.1f}]")

    xs = np.array([d["x0"] for d in ions])
    ys = np.array([d["y0"] for d in ions])
    print(f"  中心范围:  x ∈ [{xs.min():.1f}, {xs.max():.1f}],  "
          f"y ∈ [{ys.min():.1f}, {ys.max():.1f}]")
