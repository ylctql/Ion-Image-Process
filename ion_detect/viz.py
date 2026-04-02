"""Visualization and console summaries for detection output."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from .binarize import bgsub_binarize


# 与像素坐标一致：x/y 单位同为像素，作图时保持正方形像素（勿用 auto 拉扁椭圆）
_VIS_ASPECT = "equal"


def _major_axis_near_flags_along_x(ions, boundary, tol):
    """长轴沿 x (y = cy)：|y0 - cy| <= tol 为 True。tol/boundary 无效时返回 None。"""
    if tol is None or boundary is None:
        return None
    cy = float(boundary[1])
    t = float(tol)
    return np.array(
        [abs(float(ion["y0"]) - cy) <= t for ion in ions],
        dtype=bool,
    )


def _plot_weighted_r2_panel(
    ax,
    image,
    ions,
    boundary,
    n_sigma,
    near_flags=None,
    *,
    show_major_axis_hline=False,
):
    """Panel below main image: weighted R² at each ion location."""
    im = np.asarray(image, dtype=np.float64)
    vmin_i = float(np.percentile(im, 1))
    vmax_i = float(np.percentile(im, 99.5))
    ax.imshow(im, cmap="gray", aspect=_VIS_ASPECT, vmin=vmin_i, vmax=vmax_i, alpha=0.42)
    if not ions:
        ax.set_title("Fit quality (weighted R²): no ions detected", fontsize=12)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        return
    xs = np.array([float(ion["x0"]) for ion in ions], dtype=np.float64)
    ys = np.array([float(ion["y0"]) for ion in ions], dtype=np.float64)
    if near_flags is None:
        near_flags = np.zeros(len(ions), dtype=bool)
    if show_major_axis_hline and boundary is not None:
        ax.axhline(
            float(boundary[1]), color="gold", linestyle="--", linewidth=0.9, alpha=0.85, zorder=4,
        )
    for i, ion in enumerate(ions):
        near = bool(near_flags[i])
        ec, lw = ("lime", 0.9) if near else ("red", 0.35)
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor=ec, facecolor="none", linewidth=lw, alpha=0.72,
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
    vals = np.array(
        [
            float("nan") if ion.get("r2_weighted") is None else float(ion["r2_weighted"])
            for ion in ions
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(vals)
    if not valid.any():
        ec = np.where(near_flags, "#FFD700", "k")
        lw = np.where(near_flags, 2.0, 0.25)
        ax.scatter(xs, ys, s=28, c="0.7", edgecolors=ec, linewidths=lw, zorder=5)
        ax.set_title("Fit quality (weighted R²): no valid values", fontsize=12)
    else:
        v_ok = vals[valid]
        c_lo = float(np.min(v_ok))
        c_hi = float(np.max(v_ok))
        vmin_c = min(0.0, c_lo)
        vmax_c = max(1.0, c_hi)
        if vmax_c - vmin_c < 1e-9:
            vmax_c = vmin_c + 1e-6
        nf_v = near_flags[valid]
        ec = np.where(nf_v, "#FFD700", "0.15")
        lw = np.where(nf_v, 2.0, 0.25)
        sc = ax.scatter(
            xs[valid], ys[valid], c=v_ok, cmap="RdYlGn", s=42,
            vmin=vmin_c, vmax=vmax_c, edgecolors=ec, linewidths=lw, zorder=6,
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
              boundary=None, show_fit_quality=True, show=False, *,
              near_major_axis_tol=None):
    """
    Draw the frame with fitted ellipses.

    n_sigma scales ellipse axes (n_sigma * sigma).
    boundary: (cx, cy, a, b) lattice-boundary ellipse, drawn if given.
    If show_fit_quality is True, add a second panel with weighted R² per ion.
    If show is True, open an interactive window (``plt.show()``); still saves when
    output_path is set.

    near_major_axis_tol
        若与 ``boundary`` 同时给出，假定长轴沿 x，在图上画 ``y = cy`` 金线，并将
        ``|y0 - cy| <= tol`` 的离子椭圆改为亮绿加粗，其余保持红色（便于标出长轴邻域离子）。
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

    near_flags = _major_axis_near_flags_along_x(ions, boundary, near_major_axis_tol)
    ax = axes[0]
    ax.imshow(image, cmap="gray", aspect=_VIS_ASPECT,
              vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.5))
    if near_flags is not None:
        bcy_line = float(boundary[1])
        ax.axhline(
            bcy_line, color="gold", linestyle="--", linewidth=1.1, alpha=0.9, zorder=4,
        )
    for i, ion in enumerate(ions):
        near = near_flags[i] if near_flags is not None else False
        ec, lw = ("lime", 1.2) if near else ("red", 0.4)
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor=ec, facecolor="none", linewidth=lw, alpha=0.88,
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
    title_extra = ""
    if near_flags is not None:
        n_near = int(near_flags.sum())
        title_extra = (
            f"; {n_near} within |y−cy|≤{float(near_major_axis_tol):g} px"
        )
    ax.set_title(
        f"{title}   [{len(ions)} ions{title_extra}, ellipse = {n_sigma} sigma]",
        fontsize=13,
    )
    if near_flags is not None:
        leg = [
            Line2D([0], [0], color="red", lw=2.5, label="other ions"),
            Line2D(
                [0], [0], color="lime", lw=2.5,
                label=f"near axis (|y-cy|<={float(near_major_axis_tol):g} px)",
            ),
            Line2D([0], [0], color="gold", lw=2, ls="--", label="major axis y=cy"),
        ]
        ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.92)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    if show_fit_quality and r2_row is not None:
        r2_near = near_flags if near_flags is not None else None
        _plot_weighted_r2_panel(
            axes[r2_row], image, ions, boundary, n_sigma,
            near_flags=r2_near,
            show_major_axis_hline=near_flags is not None,
        )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[Saved] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_bgsub(
    bgsub,
    ions,
    n_sigma=2.0,
    title="",
    output_path=None,
    boundary=None,
    show=False,
):
    """``image - Gaussian background`` (与 ``detect_ions`` 中首轮 ``signal`` 一致).

    有符号量, 使用关于 0 对称的显示范围 (基于 |signal| 的高分位), diverging colormap。
    椭圆与晶格边界与 ``visualize`` 顶栏一致。
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    z = np.asarray(bgsub, dtype=np.float64)
    ap = float(np.percentile(np.abs(z), 99.5))
    if not np.isfinite(ap) or ap < 1e-18:
        ap = max(float(np.nanmax(np.abs(z))) if z.size else 0.0, 1e-12)
    im = ax.imshow(z, cmap="RdBu_r", aspect=_VIS_ASPECT, vmin=-ap, vmax=ap)
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.12,
        label="raw − Gaussian bg",
    )
    for ion in ions:
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor="lime", facecolor="none", linewidth=0.45, alpha=0.92,
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
        f"{title}   [bgsub / matched-filter input map; {len(ions)} ions; "
        f"ellipse = {n_sigma} sigma; |·| 99.5% scale ≈ {ap:.3g}]",
        fontsize=12,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[Saved bgsub] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_bgsub_binarized(
    bgsub,
    threshold: float,
    ions,
    n_sigma: float = 2.0,
    title: str = "",
    output_path=None,
    boundary=None,
    show: bool = False,
    *,
    ge: bool = True,
):
    """减背景 ``bgsub`` 按阈值二值化：输出 **两张独立 figure**（先 bgsub，再二值掩膜）。

    若给定 ``output_path``（如 ``…/ion_bgsub_binary_0000_thr50.png``），则保存为
    ``…_bgsub.png`` 与 ``…_mask.png``（在同目录下、同名加后缀）。

    ``threshold`` 与 :func:`~ion_detect.binarize.bgsub_binarize` 一致（默认前景 ``>= threshold``）。
    """
    z = np.asarray(bgsub, dtype=np.float64)
    mask = bgsub_binarize(z, threshold, ge=ge)
    n_fg = int(np.count_nonzero(mask))
    n_pix = int(mask.size)
    frac = (100.0 * n_fg / n_pix) if n_pix else 0.0
    rule = "≥" if ge else ">"
    ap = float(np.percentile(np.abs(z), 99.5))
    if not np.isfinite(ap) or ap < 1e-18:
        ap = max(float(np.nanmax(np.abs(z))) if z.size else 0.0, 1e-12)

    def _overlays(ax):
        for ion in ions:
            ax.add_patch(
                Ellipse(
                    xy=(ion["x0"], ion["y0"]),
                    width=2 * n_sigma * ion["sigma_minor"],
                    height=2 * n_sigma * ion["sigma_major"],
                    angle=ion["theta_deg"],
                    edgecolor="lime", facecolor="none", linewidth=0.45, alpha=0.92,
                ),
            )
        if boundary is not None:
            bcx, bcy, ba, bb = boundary
            ax.add_patch(
                Ellipse(
                    xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
                    edgecolor="cyan", facecolor="none",
                    linewidth=1.2, linestyle="--", alpha=0.9,
                ),
            )

    path_bgsub = path_mask = None
    if output_path is not None:
        base = Path(output_path)
        path_bgsub = base.with_name(f"{base.stem}_bgsub{base.suffix}")
        path_mask = base.with_name(f"{base.stem}_mask{base.suffix}")

    # Figure 1: bgsub
    fig0, ax0 = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    im0 = ax0.imshow(z, cmap="RdBu_r", aspect=_VIS_ASPECT, vmin=-ap, vmax=ap)
    plt.colorbar(
        im0,
        ax=ax0,
        orientation="horizontal",
        fraction=0.046,
        pad=0.12,
        label="raw − Gaussian bg",
    )
    _overlays(ax0)
    ax0.set_title(
        f"{title}   [bgsub; binarize {rule} {threshold:g}]",
        fontsize=12,
    )
    ax0.set_xlabel("x (pixel)")
    ax0.set_ylabel("y (pixel)")
    if path_bgsub is not None:
        fig0.savefig(path_bgsub, dpi=200, bbox_inches="tight")
        print(f"[Saved bgsub (binarize context)] {path_bgsub}")
    if show:
        plt.show()
    plt.close(fig0)

    # Figure 2: binary mask
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    im1 = ax1.imshow(mask.astype(np.float64), cmap="gray", aspect=_VIS_ASPECT, vmin=0.0, vmax=1.0)
    plt.colorbar(
        im1,
        ax=ax1,
        orientation="horizontal",
        fraction=0.046,
        pad=0.12,
        label="foreground=1",
    )
    _overlays(ax1)
    ax1.set_title(
        f"{title}   [binary bgsub {rule} {threshold:g}  |  fg {n_fg} px ({frac:.2f}%)]",
        fontsize=12,
    )
    ax1.set_xlabel("x (pixel)")
    ax1.set_ylabel("y (pixel)")
    if path_mask is not None:
        fig1.savefig(path_mask, dpi=200, bbox_inches="tight")
        print(f"[Saved bgsub binary mask] {path_mask}")
    if show:
        plt.show()
    plt.close(fig1)


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
