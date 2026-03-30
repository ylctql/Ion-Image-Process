"""
基于 merge_ion_centers 的合并中心：对 y 频数直方图取选定峰（默认第 2 个，``--line-id``）所在行 y0，
将 I(y0-1)+I(y0)+I(y0+1) 沿 x 求和找峰得到 x，再在 x±N 列、y0±halfwin 行上做 COM 得精确 y（默认 N=1 含左右邻列）；
在离子图上画出三行剖面线、质心窗口边界及识别坐标（y 为列内亮度质心，非峰拟合）。

``--line-id`` 为按峰位 y 从小到大排序后的峰序号（从 1 计数）。输出文件名含 ``line{N}`` 后缀。
用法与 merge_ion_centers.py 的检测/合并参数对齐，便于复现实验。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_AMP_Y_FIT, OUT_SECOND_LAYER_PEAKS, PROJECT_ROOT

from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.pipeline import detect_ions
from merge_ion_centers import (
    _strip_input_map,
    fuse_detect_strip_by_distance,
    merge_centers_hybrid,
)
from second_layer_core import (
    _peak_indices_with_padded_ends,
    com_y_column,
    second_histogram_peak_y_row,
    three_row_sum_profile,
)


def _collect_merged_centers(
    files: list[Path],
    selected: list[int],
    *,
    edge_x_lo: float,
    edge_x_hi: float,
    y_edge_frac: float,
    y_fit_frac: float,
    peak_dist: float,
    col_metric: str,
    strip_center_mode: str,
    add_neighbor_x: bool,
    preprocess: str,
    clip_ellipse: bool,
    ion_dist: float,
    use_matched_filter: bool,
    amp_coef_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    fids: list[int] = []
    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions, boundary = detect_ions(
            image,
            fix_theta_zero=True,
            use_matched_filter=use_matched_filter,
            amp_y_coef_path=amp_coef_path,
            amp_y_coef_mode="even",
        )
        if boundary is None:
            continue
        try:
            strip_map, _ = _strip_input_map(
                image, boundary, preprocess, ions_for_peel=ions,
            )
        except ValueError:
            continue
        strip_result = outer_y_edge_column_profiles(
            strip_map,
            boundary,
            y_edge_frac,
            clip_ellipse=clip_ellipse,
            col_metric=col_metric,
        )
        merged, _stats = merge_centers_hybrid(
            ions,
            boundary,
            strip_map,
            strip_result=strip_result,
            edge_x_lo=edge_x_lo,
            edge_x_hi=edge_x_hi,
            peak_dist=peak_dist,
            clip_ellipse=clip_ellipse,
            y_fit_frac=float(y_fit_frac),
            add_neighbor_x=add_neighbor_x,
            strip_center_mode=strip_center_mode,
        )
        merged, _nf = fuse_detect_strip_by_distance(merged, float(ion_dist))
        for p in merged:
            xs.append(float(p["x0"]))
            ys.append(float(p["y0"]))
            fids.append(int(idx))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(fids, dtype=np.int64),
    )


def _plot_y_histogram_selected_peak(
    y_arr: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray,
    peak_ix: np.ndarray,
    y_a: float,
    y_b: float,
    out_path: Path,
    *,
    line_id: int,
    hist_prominence: float,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y_arr, bins=bin_edges, edgecolor="black", linewidth=0.35, color="steelblue", alpha=0.85)
    peak_y = 0.5 * (bin_edges[peak_ix] + bin_edges[peak_ix + 1])
    for yi in peak_y:
        ax.axvline(yi, color="crimson", ls="--", lw=1.0, alpha=0.9)
    ax.axvline(
        0.5 * (y_a + y_b),
        color="darkgreen",
        ls="-",
        lw=2.0,
        label=f"selected peak (line-id={line_id}, sorted by y)",
    )
    ax.set_xlabel("merged center y (pixel)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Merged centers: y histogram (1 px bins) | line-id={line_id} bin [{y_a:.0f}, {y_b:.0f}) | "
        f"find_peaks prominence > {hist_prominence:g}",
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_peak_detection_figure(
    *,
    fid: int,
    stem: str,
    yr0: int,
    line_id: int,
    xs: np.ndarray,
    prof: np.ndarray,
    peaks_ix: np.ndarray,
    ion_xy: list[tuple[int, float, float, int, int, int, int]],
    profile_x_range: tuple[float, float],
    prof_prominence_frac: float,
    halfwin: int,
    com_neighbor_cols: int,
    out_path: Path,
    show: bool,
) -> None:
    """Full figure: three-row-sum profile along x and detected peak positions (COM y in legend/table)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x_pix = xs.astype(np.float64)
    ax.plot(x_pix, prof, "-", color="tab:purple", lw=1.2, label="sum I(y0-1 : y0+1, x)")
    for ix in peaks_ix:
        ax.scatter([float(x_pix[ix])], [prof[ix]], c="red", s=50, zorder=6, edgecolors="k", lw=0.35)
    ax.scatter([], [], c="red", s=50, edgecolors="k", lw=0.35, label="detected x peaks")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("brightness (ADU)")
    ax.set_title(
        f"Peak detection | frame [{fid:04d}] {stem} | line-id={line_id} → y0={yr0} | "
        f"x in [{profile_x_range[0]:.0f}, {profile_x_range[1]:.0f}] | "
        f"prominence ≥ {100 * prof_prominence_frac:.0f}% of profile max | "
        f"y COM: ±{halfwin} rows × x±{com_neighbor_cols} cols",
    )
    lines_txt = []
    n = 0
    for x_px, y_com, _xp, _r0, _r1, _xl, _xh in ion_xy:
        n += 1
        if np.isfinite(y_com):
            lines_txt.append(f"  #{n}: x={x_px} px, y_COM={y_com:.3f} px")
        else:
            lines_txt.append(f"  #{n}: x={x_px} px, y_COM=nan")
    table = "\n".join(lines_txt) if lines_txt else "  (no peaks)"
    ax.text(
        0.02,
        0.98,
        f"Ion coordinates (y COM, x±{com_neighbor_cols} cols × ±{halfwin} rows):\n" + table,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.9, edgecolor="0.4"),
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_ion_image_figure(
    image: np.ndarray,
    *,
    fid: int,
    stem: str,
    yr0: int,
    line_id: int,
    ion_xy: list[tuple[int, float, float, int, int, int, int]],
    halfwin: int,
    com_neighbor_cols: int,
    out_path: Path,
    show: bool,
) -> None:
    """Full figure: ion image with construction lines, COM windows, and refined positions."""
    h, w = image.shape
    y_lines = (yr0 - 1, yr0, yr0 + 1)
    fig, ax = plt.subplots(figsize=(12, 10))

    lo = float(np.percentile(image, 1.5))
    hi = float(np.percentile(image, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, aspect="equal", origin="upper", interpolation="nearest")

    line_styles = ("--", "-", "--")
    colors_3 = ("cyan", "yellow", "cyan")
    for yl, ls, c in zip(y_lines, line_styles, colors_3):
        if 0 <= yl < h:
            ax.axhline(yl, color=c, ls=ls, lw=1.3, alpha=0.95)
    ax.plot([], [], color="yellow", ls="-", lw=1.3, label="y0 (center row of sum)")
    ax.plot([], [], color="cyan", ls="--", lw=1.3, label="y0 ± 1 (rows in sum)")

    for x_px, y_com, _x_plot, r0, r1, x_pl, x_ph in ion_xy:
        if not np.isfinite(y_com):
            continue
        rw = float(x_ph - x_pl + 1)
        rect = Rectangle(
            (x_pl - 0.5, r0 - 0.5),
            rw,
            float(r1 - r0),
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.scatter([x_px], [y_com], c="red", s=55, zorder=7, edgecolors="k", linewidths=0.4, marker="o")

    y_win_lo = yr0 - halfwin - 0.5
    y_win_hi = yr0 + halfwin + 0.5
    ax.axhline(y_win_lo, color="orange", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(y_win_hi, color="orange", ls=":", lw=1.0, alpha=0.85)
    ax.plot([], [], color="orange", ls=":", lw=1.0, label=f"y0 ± {halfwin} (COM window edges)")

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(
        f"Ion image | frame [{fid:04d}] {stem} | line-id={line_id}, y0={yr0} | "
        f"yellow/cyan: sum rows; orange: ±{halfwin} window; lime: COM patch (x±{com_neighbor_cols}); red: (x_peak, y)",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并中心 y 直方图选峰 → 三行和剖面找 x 峰 → 在 x±N 与 y0±halfwin 上 COM 得精确 y 并出图",
    )
    parser.add_argument("indices", nargs="*", default=["0"], help="帧索引（与 merge_ion_centers 相同）")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "20260305_1727")
    parser.add_argument("--out", type=Path, default=OUT_SECOND_LAYER_PEAKS)
    parser.add_argument(
        "--edge-x-range",
        type=float,
        nargs=2,
        default=[300.0, 600.0],
        metavar=("X0", "X1"),
        help="与 merge_ion_centers 相同：条带优先域 x（合并中心用）",
    )
    parser.add_argument(
        "--profile-x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("X0", "X1"),
        help="三行和沿 x 积分的列范围（像素闭区间）；默认与 --edge-x-range 相同",
    )
    parser.add_argument("--y-edge-frac", type=float, default=0.25)
    parser.add_argument("--y-fit-frac", type=float, default=0.35)
    parser.add_argument("--peak-dist", type=float, default=5.0)
    parser.add_argument("--col-metric", choices=("sum", "mean", "max"), default="mean")
    parser.add_argument(
        "--strip-center-mode",
        choices=("com", "com_fit", "fit"),
        default="com",
    )
    parser.add_argument("--add-neighbor-x", action="store_true")
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub", "peel", "peel_bgsub"),
        default="raw",
    )
    parser.add_argument("--no-clip-ellipse", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-matched-filter", action="store_true")
    parser.add_argument("--ion-dist", type=float, default=4.0, metavar="PX")
    parser.add_argument(
        "--hist-prominence",
        type=float,
        default=10.0,
        help="y 频数直方图上 find_peaks 的 prominence",
    )
    parser.add_argument(
        "--prof-prominence-frac",
        type=float,
        default=0.08,
        help="剖面最大值的比例，作为 x 向 find_peaks prominence",
    )
    parser.add_argument("--prof-peak-distance", type=int, default=4, help="x 向峰最小间距（像素）")
    parser.add_argument("--y-halfwin", type=int, default=3, help="y0 上下各取若干行做 COM（默认 3）")
    parser.add_argument(
        "--y-com-neighbor-cols",
        type=int,
        default=1,
        metavar="N",
        help="y 质心时在峰值列两侧各并入 N 列（0=仅峰值列；默认 1 即 x−1,x,x+1）",
    )
    parser.add_argument(
        "--line-id",
        type=int,
        default=2,
        metavar="N",
        help="合并中心 y 直方图上，按峰位 y 从小到大排序后的峰序号（从 1 计数；默认 2 即原“第二峰”）",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 下没有 npy 文件")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有可处理的索引")

    ex_lo, ex_hi = float(args.edge_x_range[0]), float(args.edge_x_range[1])
    if args.profile_x_range is None:
        px_lo, px_hi = ex_lo, ex_hi
    else:
        px_lo, px_hi = float(args.profile_x_range[0]), float(args.profile_x_range[1])
    if px_lo > px_hi:
        px_lo, px_hi = px_hi, px_lo

    amp_path = OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy"
    clip_ellipse = not args.no_clip_ellipse

    x_arr, y_arr, fid_arr = _collect_merged_centers(
        files,
        selected,
        edge_x_lo=ex_lo,
        edge_x_hi=ex_hi,
        y_edge_frac=float(args.y_edge_frac),
        y_fit_frac=float(args.y_fit_frac),
        peak_dist=float(args.peak_dist),
        col_metric=args.col_metric,
        strip_center_mode=args.strip_center_mode,
        add_neighbor_x=args.add_neighbor_x,
        preprocess=args.preprocess,
        clip_ellipse=clip_ellipse,
        ion_dist=float(args.ion_dist),
        use_matched_filter=not args.no_matched_filter,
        amp_coef_path=amp_path,
    )
    if y_arr.size == 0:
        raise SystemExit("未得到任何合并中心，请检查数据与参数")

    line_id = int(args.line_id)
    if line_id < 1:
        raise SystemExit("--line-id 须为 >= 1 的整数（按 y 排序后的第 N 个峰）")

    y_lo = int(np.floor(y_arr.min()))
    y_hi = int(np.ceil(y_arr.max()))
    bin_edges = np.arange(y_lo, y_hi + 2, dtype=float)
    y0, _k, y_a, y_b, counts, peak_ix = second_histogram_peak_y_row(
        y_arr, bin_edges, args.hist_prominence, line_id,
    )

    fn_line = f"line{line_id}"
    hist_png = args.out / f"y_histogram_second_layer_{fn_line}.png"
    _plot_y_histogram_selected_peak(
        y_arr,
        bin_edges,
        counts,
        peak_ix,
        y_a,
        y_b,
        hist_png,
        line_id=line_id,
        hist_prominence=float(args.hist_prominence),
        show=args.show,
    )
    print(
        f"合并中心数 N={y_arr.size}；y 直方图 line-id={line_id} → y0={y0}（bin [{y_a:.0f},{y_b:.0f})）",
    )
    print(f"已保存 y 直方图: {hist_png}")

    halfwin = int(args.y_halfwin)
    com_n = max(0, int(args.y_com_neighbor_cols))
    prom_frac = float(args.prof_prominence_frac)
    dist_px = max(1, int(args.prof_peak_distance))

    for idx in selected:
        target = files[idx]
        image = np.asarray(np.load(target), dtype=np.float64)
        h, _w = image.shape
        yr = int(np.clip(y0, 0, h - 1))
        xs, prof = three_row_sum_profile(image, yr, int(np.floor(px_lo)), int(np.ceil(px_hi)))
        if prof.size == 0:
            print(f"[{idx:04d}] {target.name}: 剖面 x 范围无效，跳过")
            continue
        pmax = float(np.max(prof))
        prom = max(pmax * prom_frac, 1e-9)
        peaks_ix = _peak_indices_with_padded_ends(
            prof, prominence=prom, distance=dist_px,
        )

        ion_xy: list[tuple[int, float, float, int, int, int, int]] = []
        for ix in peaks_ix:
            x_px = int(xs[ix])
            y_com, r0, r1, x_pl, x_ph = com_y_column(
                image, x_px, yr, halfwin, neighbor_cols=com_n,
            )
            ion_xy.append((x_px, y_com, float(xs[ix]), r0, r1, x_pl, x_ph))

        stem = Path(target.name).stem
        out_profile = args.out / f"second_layer_profile_{idx:04d}_{fn_line}.png"
        out_image = args.out / f"second_layer_image_{idx:04d}_{fn_line}.png"
        _plot_peak_detection_figure(
            fid=idx,
            stem=stem,
            yr0=yr,
            line_id=line_id,
            xs=xs,
            prof=prof,
            peaks_ix=peaks_ix,
            ion_xy=ion_xy,
            profile_x_range=(px_lo, px_hi),
            prof_prominence_frac=prom_frac,
            halfwin=halfwin,
            com_neighbor_cols=com_n,
            out_path=out_profile,
            show=args.show,
        )
        _plot_ion_image_figure(
            image,
            fid=idx,
            stem=stem,
            yr0=yr,
            line_id=line_id,
            ion_xy=ion_xy,
            halfwin=halfwin,
            com_neighbor_cols=com_n,
            out_path=out_image,
            show=args.show,
        )
        n_ok = sum(1 for t in ion_xy if np.isfinite(t[1]))
        print(
            f"[{idx:04d}] {target.name}: x peaks={len(peaks_ix)}, valid COM={n_ok} → "
            f"{out_profile.name}, {out_image.name}",
        )


if __name__ == "__main__":
    main()
