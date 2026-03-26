"""
对 y 向椭圆外缘条带（与 |y-cy|/b >= 1-F 同一截线几何）按列聚合，可视化 1D 曲线并报告峰值。

边界估计与 ``detect_ions`` 一致：在 ``image - 高斯背景`` 上调用 ``estimate_crystal_boundary``。

用法示例:
  python edge_strip_profile.py 0
  python edge_strip_profile.py 0 5 --y-edge-frac 0.25 --preprocess raw
  python edge_strip_profile.py 0 --preprocess peel
  python edge_strip_profile.py 0 --preprocess peel --plot-peel
  python edge_strip_profile.py 0 --preprocess bgsub --col-metric mean
  python edge_strip_profile.py 0 --col-metric sum
  python edge_strip_profile.py 0 --col-metric max
  python edge_strip_profile.py ::10 --out outputs/edge_strip_profiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from scipy.signal import peak_prominences

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_EDGE_STRIP, PROJECT_ROOT

from ion_detect.boundary import estimate_crystal_boundary
from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.pipeline import detect_ions


def _ascii_figure_title(idx: int, npy_filename: str) -> str:
    """Matplotlib titles: ASCII only (avoid missing CJK glyphs in default fonts)."""
    stem = Path(npy_filename).stem
    safe = stem.encode("ascii", "replace").decode("ascii")
    return f"frame {idx:04d} ({safe})"


def _boundary_from_image_like_pipeline(image: np.ndarray, bg_sigma=(10, 30)):
    img = image.astype(np.float64)
    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg
    return estimate_crystal_boundary(signal), signal


def _bgsub_array(arr: np.ndarray, bg_sigma=(10, 30)) -> np.ndarray:
    """Same Gaussian background subtraction as detect_ions (round-2 uses this on peeled image)."""
    u = np.asarray(arr, dtype=np.float64)
    bg = gaussian_filter(u, sigma=bg_sigma)
    return u - bg


def _raw_or_bgsub_map(image: np.ndarray, signal: np.ndarray, mode: str) -> np.ndarray:
    """``preprocess`` 为 raw / bgsub 时，得到送入条带列聚合的二维图。"""
    if mode == "raw":
        return image.astype(np.float64)
    if mode == "bgsub":
        return np.asarray(signal, dtype=np.float64)
    raise ValueError(f"unknown preprocess (expected raw or bgsub): {mode}")


def _y_for_peak_prominence(y_prof: np.ndarray) -> np.ndarray:
    """Replace NaN so ``peak_prominences`` can use the full length of the profile."""
    y = np.asarray(y_prof, dtype=np.float64)
    if np.all(np.isfinite(y)):
        return y
    y2 = y.copy()
    lo = np.nanmin(y2)
    fill = float(lo) if np.isfinite(lo) else 0.0
    y2[~np.isfinite(y2)] = fill
    return y2


def _strip_local_maxima_ixy(
    x_grid: np.ndarray,
    y_prof: np.ndarray,
    col_counts: np.ndarray | None,
) -> list[tuple[int, float, float]]:
    """Strict 1D local maxima (i..n-2): y[i] > y[i-1] and y[i] > y[i+1].

    Skips columns with non-finite values or col_counts[i] <= 0 when counts are given.
    Returns ``(sample_index, x, y)`` sorted by x.
    """
    xg = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_prof, dtype=np.float64)
    n = int(y.size)
    cnt = np.asarray(col_counts, dtype=np.float64) if col_counts is not None else None
    out: list[tuple[int, float, float]] = []
    for i in range(1, n - 1):
        if cnt is not None and cnt[i] <= 0.0:
            continue
        a, b, c = y[i - 1], y[i], y[i + 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        if b > a and b > c:
            out.append((i, float(xg[i]), float(b)))
    return sorted(out, key=lambda t: t[1])


def _strip_peaks_min_distance(
    peaks_ixy: list[tuple[int, float, float]],
    y_fin: np.ndarray,
    peak_dist: float,
) -> list[float]:
    """Require spacing > peak_dist between adjacent kept peaks (in x).

    If two peaks violate that, drop the one with lower `scipy.signal.peak_prominences`
    (recomputed on the current survivor set each time). Tie: higher y.
    ``peak_dist <= 0`` skips filtering.
    """
    if peak_dist <= 0.0 or len(peaks_ixy) <= 1:
        return [p[1] for p in sorted(peaks_ixy, key=lambda t: t[1])]

    pts = sorted(peaks_ixy, key=lambda t: t[1])
    y_fin = np.asarray(y_fin, dtype=np.float64)
    while True:
        j = -1
        for i in range(len(pts) - 1):
            if pts[i + 1][1] - pts[i][1] <= peak_dist:
                j = i
                break
        if j < 0:
            break
        k0, k1 = j, j + 1
        ix_arr = np.array([p[0] for p in pts], dtype=np.intp)
        prom, _, _ = peak_prominences(y_fin, ix_arr, wlen=None)
        p0 = float(prom[k0])
        p1 = float(prom[k1])
        _ix0, _x0, y0 = pts[k0]
        _ix1, _x1, y1 = pts[k1]
        if p0 > p1:
            del pts[k1]
        elif p1 > p0:
            del pts[k0]
        elif y0 >= y1:
            del pts[k1]
        else:
            del pts[k0]
    return [p[1] for p in pts]


def _strip_profile_peak_xs(
    x_grid: np.ndarray,
    y_prof: np.ndarray,
    col_counts: np.ndarray | None,
    peak_dist: float,
) -> list[float]:
    peaks = _strip_local_maxima_ixy(x_grid, y_prof, col_counts)
    y_fin = _y_for_peak_prominence(y_prof)
    return _strip_peaks_min_distance(peaks, y_fin, peak_dist)


def _draw_strip_outlines(ax, meta: dict, image_shape: tuple[int, int]) -> None:
    h, w = image_shape
    cx, cy, a, b = meta["cx"], meta["cy"], meta["a"], meta["b"]
    x_half = meta["x_half"]
    y_below, y_above = meta["y_below"], meta["y_above"]
    y_top_v, y_bot_v = meta["y_top_vertex"], meta["y_bot_vertex"]
    lc = "orange"
    lw = 1.2
    xt = [
        cx - x_half, cx + x_half, cx + x_half, cx - x_half, cx - x_half,
    ]
    yt = [y_top_v, y_top_v, y_below, y_below, y_top_v]
    ax.plot(xt, yt, color=lc, linewidth=lw, linestyle="-", alpha=0.9)
    xb = [
        cx - x_half, cx + x_half, cx + x_half, cx - x_half, cx - x_half,
    ]
    yb = [y_above, y_above, y_bot_v, y_bot_v, y_above]
    ax.plot(xb, yb, color=lc, linewidth=lw, linestyle="-", alpha=0.9)
    ax.axhline(y_below, color="darkmagenta", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.axhline(y_above, color="darkmagenta", linestyle=":", linewidth=1.0, alpha=0.75)
    ell = Ellipse(
        xy=(cx, cy), width=2 * a, height=2 * b, angle=0,
        edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--", alpha=0.85,
    )
    ax.add_patch(ell)
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)


def _plot_and_save(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    result: dict,
    out_path: Path | None,
    title: str,
    preprocess: str,
    *,
    show: bool,
    peak_dist: float,
    left_panel: np.ndarray | None = None,
) -> None:
    """left_panel: imshow 顶栏；None 时用原始 ``image``；peel 模式仅在指定 ``--plot-peel`` 时传入残差图。"""
    panel0 = image if left_panel is None else left_panel
    meta = result["meta"]
    h, w = panel0.shape
    x = result["x"]
    top_p = result["top_profile"]
    bot_p = result["bot_profile"]

    # Top: full-width lattice + strips; bottom: top/bottom strip profiles (more image area).
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.33, wspace=0.2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    lo = float(np.percentile(panel0, 1))
    hi = float(np.percentile(panel0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(panel0)), float(np.nanmax(panel0))
    ax0.imshow(panel0, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    _draw_strip_outlines(ax0, meta, (h, w))
    fe = f"{float(meta['y_edge_frac']):g}"
    clip = "clip_ellipse" if meta["clip_ellipse"] else "full_rect"
    cm = meta.get("col_metric", "mean")
    agg = {"sum": "col_sum", "mean": "col_mean/mask", "max": "col_max/mask"}.get(cm, cm)
    pd = float(peak_dist)
    dash_hint = (
        f"strip peaks: spacing>{pd:g} px (prominence tie-break)"
        if pd > 0.0
        else "strip peaks: all local maxima"
    )
    ax0.set_title(
        f"{title}\nF={fe}, preprocess={preprocess}, {clip}, {agg}\n"
        f"(dashes: {dash_hint}; tomato=top, lime=bottom)"
    )
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")

    _yl = {
        "sum": ("col sum (strip)", "col sum", "sum ~"),
        "mean": ("col mean (strip)", "col mean", "mean ~"),
        "max": ("col max (strip)", "col max", "max ~"),
    }
    y_metric, ylbl_short, peak_lbl = _yl.get(cm, (cm, cm, "peak ~"))

    tc = result.get("top_col_counts")
    bc = result.get("bot_col_counts")
    if cm in ("mean", "max"):
        top_plot = np.asarray(top_p, dtype=np.float64).copy()
        bot_plot = np.asarray(bot_p, dtype=np.float64).copy()
        if tc is not None:
            top_plot[tc <= 0] = np.nan
        if bc is not None:
            bot_plot[bc <= 0] = np.nan
    else:
        top_plot = top_p
        bot_plot = bot_p

    y_below_m = float(meta["y_below"])
    y_above_m = float(meta["y_above"])
    y_top_v = float(meta["y_top_vertex"])
    y_bot_v = float(meta["y_bot_vertex"])
    for xp in _strip_profile_peak_xs(x, top_plot, tc, peak_dist):
        ax0.plot(
            [xp, xp], [y_top_v, y_below_m],
            color="tomato", linestyle="--", linewidth=1.0, alpha=0.92, zorder=5,
        )
    for xp in _strip_profile_peak_xs(x, bot_plot, bc, peak_dist):
        ax0.plot(
            [xp, xp], [y_above_m, y_bot_v],
            color="lime", linestyle="--", linewidth=1.0, alpha=0.92, zorder=5,
        )

    ax1.plot(x, top_plot, color="tab:blue", linewidth=1.0)
    ax1.axvline(result["top_peak_x"], color="tab:red", linestyle="--", alpha=0.85)
    ax1.scatter([result["top_peak_x"]], [result["top_peak_value"]], color="tab:red", s=36, zorder=3)
    ax1.set_title(
        f"Top strip: {y_metric} vs x\npeak x ~ {result['top_peak_x']:.2f}, "
        f"{peak_lbl} {result['top_peak_value']:.4g}"
    )
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel(ylbl_short)

    ax2.plot(x, bot_plot, color="tab:green", linewidth=1.0)
    ax2.axvline(result["bot_peak_x"], color="tab:red", linestyle="--", alpha=0.85)
    ax2.scatter([result["bot_peak_x"]], [result["bot_peak_value"]], color="tab:red", s=36, zorder=3)
    ax2.set_title(
        f"Bottom strip: {y_metric} vs x\npeak x ~ {result['bot_peak_x']:.2f}, "
        f"{peak_lbl} {result['bot_peak_value']:.4g}"
    )
    ax2.set_xlabel("x (px)")
    ax2.set_ylabel(ylbl_short)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"[已保存] {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="y 向外缘条带按列聚合与峰值分析")
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引（与 ion_detect 相同 numpy 风格切片）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "20260305_1727",
        help="npy 数据目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_EDGE_STRIP,
        help="输出 PNG 目录",
    )
    parser.add_argument(
        "--y-edge-frac",
        type=float,
        default=0.25,
        metavar="F",
        help="边缘带参数 F，与 |y-cy|/b≥1-F 一致（默认 0.25）",
    )
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub", "peel", "peel_bgsub"),
        default="raw",
        help=(
            "Which 2D map feeds strip column aggregation: raw; bgsub = image - Gaussian bg; "
            "peel = first-round peak-peel residual (needs ions); "
            "peel_bgsub = that residual minus Gaussian bg (same as detect_ions round-2 map input), needs ions."
        ),
    )
    parser.add_argument(
        "--plot-peel",
        action="store_true",
        help=(
            "Only when --preprocess is peel or peel_bgsub: show the peeled (or peel+bgsub) map in the top imshow. "
            "Default: top panel uses the raw loaded image; profiles still use the peel-mode map."
        ),
    )
    parser.add_argument(
        "--no-clip-ellipse",
        action="store_true",
        help="不按椭圆裁剪，对整个轴对齐矩形内像素求和（可能含晶格外角点）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="交互显示图形窗口",
    )
    parser.add_argument(
        "--col-metric",
        choices=("sum", "mean", "max"),
        default="mean",
        help=(
            "Per-column aggregate over strip mask pixels: sum, mean (default), or max. "
            "Mean divides by mask pixel count per column. Max uses NaN where a column has no mask pixels."
        ),
    )
    parser.add_argument(
        "--peak-dist",
        type=float,
        default=5.0,
        metavar="D",
        help=(
            "Strip-profile auxiliary peaks (vertical dashes): keep peaks spaced by > D in x; "
            "if closer, drop the lower-prominence peak (scipy.signal.peak_prominences; tie: higher y). "
            "Use <=0 for every strict local maximum."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 下没有 npy 文件")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有可处理的索引")

    clip_ellipse = not args.no_clip_ellipse

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        left_panel = None

        if args.preprocess in ("peel", "peel_bgsub"):
            peel_out = detect_ions(
                image,
                peak_peel=True,
                return_peel_residual=True,
            )
            assert len(peel_out) == 3
            _ions, boundary, peel_residual = peel_out
            if boundary is None:
                print(f"[{idx:04d}] {target.name}: no boundary, skip.")
                continue
            if peel_residual is None:
                print(
                    f"[{idx:04d}] {target.name}: "
                    "no peak-peel residual (first round had no ions), skip peel preprocess."
                )
                continue
            if args.preprocess == "peel_bgsub":
                strip_map = _bgsub_array(peel_residual)
                if args.plot_peel:
                    left_panel = strip_map
            else:
                strip_map = peel_residual
                if args.plot_peel:
                    left_panel = peel_residual
        else:
            boundary, signal = _boundary_from_image_like_pipeline(image)
            if boundary is None:
                print(f"[{idx:04d}] {target.name}: 无法估计 boundary，跳过。")
                continue
            strip_map = _raw_or_bgsub_map(image, signal, args.preprocess)

        result = outer_y_edge_column_profiles(
            strip_map,
            boundary,
            args.y_edge_frac,
            clip_ellipse=clip_ellipse,
            col_metric=args.col_metric,
        )
        vlabel = {"sum": "ΣI", "mean": "均值", "max": "最大值"}[args.col_metric]
        print(
            f"\n[{idx:04d}] {target.name}  boundary=({boundary[0]:.1f}, {boundary[1]:.1f}, "
            f"a={boundary[2]:.1f}, b={boundary[3]:.1f})"
        )
        print(
            f"  上缘条带 col={args.col_metric} 峰值: x≈{result['top_peak_x']:.3f} px, "
            f"{vlabel}≈{result['top_peak_value']:.5g} (preprocess={args.preprocess})"
        )
        print(
            f"  下缘条带 col={args.col_metric} 峰值: x≈{result['bot_peak_x']:.3f} px, "
            f"{vlabel}≈{result['bot_peak_value']:.5g}"
        )

        out_png = args.out / f"edge_strip_profile_{idx:04d}.png"
        _plot_and_save(
            image,
            boundary,
            result,
            out_png,
            title=_ascii_figure_title(idx, target.name),
            preprocess=args.preprocess,
            show=args.show,
            peak_dist=args.peak_dist,
            left_panel=left_panel,
        )


if __name__ == "__main__":
    main()
