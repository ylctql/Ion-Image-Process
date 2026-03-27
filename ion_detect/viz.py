"""Visualization and console summaries for detection output."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .peel import y_edge_band_thresholds


_VIS_ASPECT = "auto"


def _plot_weighted_r2_panel(ax, image, ions, boundary, n_sigma):
    """Panel below main image: weighted R² at each ion location."""
    im = np.asarray(image, dtype=np.float64)
    vmin_i = float(np.percentile(im, 1))
    vmax_i = float(np.percentile(im, 99.5))
    ax.imshow(im, cmap="gray", aspect=_VIS_ASPECT, vmin=vmin_i, vmax=vmax_i, alpha=0.42)
    for ion in ions:
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor="red", facecolor="none", linewidth=0.35, alpha=0.65,
        )
        ax.add_patch(ell)
    if boundary is not None:
        bcx, bcy, ba, bb = boundary
        bnd_ell = Ellipse(
            xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
            edgecolor="cyan", facecolor="none",
            linewidth=1.0, linestyle="--", alpha=0.85,
        )
        ax.add_patch(bnd_ell)
    if not ions:
        ax.set_title("Fit quality (weighted R²): no ions detected", fontsize=12)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        return
    xs = np.array([float(ion["x0"]) for ion in ions], dtype=np.float64)
    ys = np.array([float(ion["y0"]) for ion in ions], dtype=np.float64)
    vals = np.array(
        [
            float("nan") if ion.get("r2_weighted") is None else float(ion["r2_weighted"])
            for ion in ions
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(vals)
    if not valid.any():
        ax.scatter(xs, ys, s=28, c="0.7", edgecolors="k", linewidths=0.25, zorder=5)
        ax.set_title("Fit quality (weighted R²): no valid values", fontsize=12)
    else:
        v_ok = vals[valid]
        c_lo = float(np.min(v_ok))
        c_hi = float(np.max(v_ok))
        vmin_c = min(0.0, c_lo)
        vmax_c = max(1.0, c_hi)
        if vmax_c - vmin_c < 1e-9:
            vmax_c = vmin_c + 1e-6
        sc = ax.scatter(
            xs[valid], ys[valid], c=v_ok, cmap="RdYlGn", s=42,
            vmin=vmin_c, vmax=vmax_c, edgecolors="0.15", linewidths=0.25, zorder=6,
        )
        plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label="Weighted R²")
        n_bad = int(np.sum(~valid))
        tail = f"; {n_bad} without R²" if n_bad else ""
        ax.set_title(
            f"Fit quality (weighted R², same weights as curve_fit)  "
            f"[min={float(np.nanmin(v_ok)):.3f}, max={float(np.nanmax(v_ok)):.3f}, "
            f"median={float(np.nanmedian(v_ok)):.3f}]{tail}",
            fontsize=11,
        )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")


def visualize(image, ions, n_sigma=2.0, title="", output_path=None,
              boundary=None, show_fit_quality=True, show=False):
    """
    Draw the frame with fitted ellipses.

    n_sigma scales ellipse axes (n_sigma * sigma).
    boundary: (cx, cy, a, b) lattice-boundary ellipse, drawn if given.
    If show_fit_quality is True, add a second panel with weighted R² per ion.
    If show is True, open an interactive window (``plt.show()``); still saves when
    output_path is set.
    """
    height_ratios = [5, 2] if show_fit_quality else [5]
    nrows = len(height_ratios)
    fig_h = 5.0 + (2.6 if show_fit_quality else 0.0)
    fig, axes = plt.subplots(
        nrows, 1, figsize=(20, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes).ravel()

    r2_row = 1 if show_fit_quality else None

    ax = axes[0]
    ax.imshow(image, cmap="gray", aspect=_VIS_ASPECT,
              vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.5))
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

    if show_fit_quality and r2_row is not None:
        _plot_weighted_r2_panel(axes[r2_row], image, ions, boundary, n_sigma)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_peel_residual(
    residual,
    title="",
    output_path=None,
    boundary=None,
    reference_image=None,
    peak_peel_y_edges_only=False,
    peak_peel_y_edge_frac=0.25,
    show=False,
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

    If show is True, display the figure with ``plt.show()`` (after saving if
    output_path is set).
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
        print(f"[Saved residual] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(ions):
    if not ions:
        print("No ions detected.")
        return
    minors = np.array([d["sigma_minor"] for d in ions])
    majors = np.array([d["sigma_major"] for d in ions])
    amps   = np.array([d["amplitude"]   for d in ions])
    ratios = majors / minors

    print(f"\nDetection: {len(ions)} ions")
    r2_list = [d.get("r2_weighted") for d in ions]
    if any(v is not None for v in r2_list):
        r2_arr = np.array(
            [np.nan if v is None else float(v) for v in r2_list],
            dtype=np.float64,
        )
        ok = np.isfinite(r2_arr)
        if ok.any():
            ra = r2_arr[ok]
            print(
                f"  Weighted R² (patch fit): mean={ra.mean():.3f} ± {ra.std():.3f}  "
                f"range=[{ra.min():.3f}, {ra.max():.3f}]  "
                f"(n={int(ok.sum())}/{len(ions)})"
            )
    print(f"  σ_minor: mean={minors.mean():.2f} ± {minors.std():.2f}  "
          f"range=[{minors.min():.2f}, {minors.max():.2f}]")
    print(f"  σ_major: mean={majors.mean():.2f} ± {majors.std():.2f}  "
          f"range=[{majors.min():.2f}, {majors.max():.2f}]")
    print(f"  Axis ratio (major/minor): mean={ratios.mean():.2f} ± {ratios.std():.2f}")
    print(f"  Amplitude: mean={amps.mean():.1f} ± {amps.std():.1f}  "
          f"range=[{amps.min():.1f}, {amps.max():.1f}]")

    xs = np.array([d["x0"] for d in ions])
    ys = np.array([d["y0"] for d in ions])
    print(f"  Center range: x ∈ [{xs.min():.1f}, {xs.max():.1f}],  "
          f"y ∈ [{ys.min():.1f}, {ys.max():.1f}]")
