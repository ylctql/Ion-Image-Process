"""
轴对齐矩形条带内离子中心：按列对条带做 y 向聚合得到 x 向 1D 曲线 -> 找局部峰（间距 peak_dist）
-> 在峰所在列上沿 y 做 COM / 高斯等（与 edge_strip_profile 相同分析链）。

用法示例:
  python custom_strip_centers.py 0 --x-range 400 600 --y-range 20 80
  python custom_strip_centers.py 0 --x-range 400 600 --y-norm-range 0.25 0.3 --draw-boundary
  （上带 F 与 edge_strip --y-edge-frac 一致；两数须同号，正=上带负=下带）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_AMP_Y_FIT, OUT_CUSTOM_STRIP_CENTERS, PROJECT_ROOT

from ion_detect.boundary import estimate_crystal_boundary
from ion_detect.cli_helpers import resolve_indices
from ion_detect.custom_strip import (
    fitted_xy_for_rect_strip_peaks,
    pixel_y_range_from_y_edge_frac_interval,
    rect_strip_column_profiles,
    rect_strip_profile_for_peak_search,
)
from ion_detect.edge_strip_profile_analysis import strip_profile_peak_xs
from ion_detect.pipeline import detect_ions


def _parse_y_norm_range_f_signed(v0: float, v1: float) -> tuple[float, float, bool]:
    """解析 --y-norm-range：返回 (f_lo, f_hi, top)。须两数同号，否则报错。"""
    if v0 >= 0.0 and v1 >= 0.0:
        return min(v0, v1), max(v0, v1), True
    if v0 <= 0.0 and v1 <= 0.0:
        return min(abs(v0), abs(v1)), max(abs(v0), abs(v1)), False
    raise ValueError(
        "y-norm-range 两数须同号：均为正表示上带 F 区间，均为负表示下带 F 区间（取绝对值）",
    )


def _ascii_title(idx: int, npy_filename: str) -> str:
    stem = Path(npy_filename).stem
    safe = stem.encode("ascii", "replace").decode("ascii")
    return f"frame {idx:04d} ({safe})"


def _bgsub_array(arr: np.ndarray, bg_sigma=(10, 30)) -> np.ndarray:
    u = np.asarray(arr, dtype=np.float64)
    bg = gaussian_filter(u, sigma=bg_sigma)
    return u - bg


def _strip_map_from_preprocess(
    image: np.ndarray,
    preprocess: str,
    ions: list | None,
) -> tuple[np.ndarray, tuple | None]:
    """返回送入条带聚合的二维图；可选 boundary（raw/bgsub 时在减背景信号上估计）。"""
    if preprocess == "raw":
        sig = _bgsub_array(image)
        boundary = estimate_crystal_boundary(sig)
        return image.astype(np.float64), boundary
    if preprocess == "bgsub":
        sig = _bgsub_array(image)
        boundary = estimate_crystal_boundary(sig)
        return sig, boundary
    if preprocess == "peel":
        from ion_detect.gaussian import _accumulate_peel_model

        if not ions:
            raise ValueError("preprocess=peel 需要首轮 ions")
        h, w = image.shape
        peel = _accumulate_peel_model(h, w, ions, margin_sigma=4.5)
        resid = image.astype(np.float64) - peel
        boundary = estimate_crystal_boundary(_bgsub_array(image))
        return resid, boundary
    if preprocess == "peel_bgsub":
        from ion_detect.gaussian import _accumulate_peel_model

        if not ions:
            raise ValueError("preprocess=peel_bgsub 需要首轮 ions")
        h, w = image.shape
        peel = _accumulate_peel_model(h, w, ions, margin_sigma=4.5)
        resid = image.astype(np.float64) - peel
        boundary = estimate_crystal_boundary(_bgsub_array(image))
        return _bgsub_array(resid), boundary
    raise ValueError(f"unknown preprocess: {preprocess}")


def _plot_dashboard(
    image: np.ndarray,
    strip_map: np.ndarray,
    result: dict,
    centers: list[tuple[float, float]],
    boundary: tuple[float, float, float, float] | None,
    out_path: Path,
    title: str,
    peak_dist: float,
    *,
    show: bool,
) -> None:
    meta = result["meta"]
    x_lo, x_hi = int(meta["x_lo"]), int(meta["x_hi"])
    y_lo, y_hi = int(meta["y_lo"]), int(meta["y_hi"])
    h, w = image.shape

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    panel0 = image
    lo = float(np.percentile(panel0, 1))
    hi = float(np.percentile(panel0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(panel0)), float(np.nanmax(panel0))
    ax0.imshow(panel0, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    rect = Rectangle(
        (x_lo - 0.5, y_lo - 0.5),
        x_hi - x_lo + 1,
        y_hi - y_lo + 1,
        linewidth=1.4,
        edgecolor="orange",
        facecolor="none",
        alpha=0.95,
    )
    ax0.add_patch(rect)
    if boundary is not None:
        cx, cy, a, b = boundary
        ell = Ellipse(
            xy=(cx, cy), width=2 * a, height=2 * b, angle=0,
            edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--", alpha=0.85,
        )
        ax0.add_patch(ell)
    prof_plot, tc = rect_strip_profile_for_peak_search(result)
    x = result["x"]
    for xp in strip_profile_peak_xs(x, prof_plot, tc, peak_dist):
        ax0.axvline(xp, color="gold", linestyle="--", linewidth=0.9, alpha=0.85)
    if centers:
        xs, ys = zip(*centers)
        ax0.scatter(xs, ys, s=26, c="deepskyblue", edgecolors="k", linewidths=0.35, zorder=6)
    ax0.set_xlim(0, w - 1)
    ax0.set_ylim(h - 1, 0)
    cm = meta.get("col_metric", "mean")
    ynorm = meta.get("y_crystal_norm")
    ynorm_str = f" {ynorm}" if ynorm is not None else ""
    ax0.set_title(
        f"{title}\nrect strip x=[{x_lo},{x_hi}] y=[{y_lo},{y_hi}]{ynorm_str} col={cm} | {len(centers)} centers",
    )
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")

    ax1.imshow(strip_map, cmap="gray", aspect="equal",
               vmin=np.percentile(strip_map, 1),
               vmax=np.percentile(strip_map, 99.5))
    ax1.add_patch(
        Rectangle(
            (x_lo - 0.5, y_lo - 0.5),
            x_hi - x_lo + 1,
            y_hi - y_lo + 1,
            linewidth=1.2,
            edgecolor="orange",
            facecolor="none",
        ),
    )
    ax1.set_title("strip map (preprocess)")
    ax1.set_xlim(0, w - 1)
    ax1.set_ylim(h - 1, 0)

    ylab = {"sum": "col sum", "mean": "col mean / mask", "max": "col max / mask"}.get(cm, cm)
    ax2.plot(x, result["profile"], color="tab:blue", linewidth=1.0)
    ax2.axvline(result["peak_x"], color="tab:red", linestyle="--", alpha=0.85)
    ax2.scatter([result["peak_x"]], [result["peak_value"]], color="tab:red", s=36, zorder=3)
    ax2.set_title(f"x-profile ({ylab}); parabola global max x≈{result['peak_x']:.2f}")
    ax2.set_xlabel("x (px)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="轴对齐矩形条带：x 向峰 + 列内 y 中心（复用 edge_strip 分析链）",
    )
    parser.add_argument("indices", nargs="*", default=["0"], help="帧索引")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "20260305_1727")
    parser.add_argument("--out", type=Path, default=OUT_CUSTOM_STRIP_CENTERS)
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        required=True,
        metavar=("X0", "X1"),
        help="条带列范围（含端点，像素）",
    )
    y_grp = parser.add_mutually_exclusive_group(required=True)
    y_grp.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        metavar=("Y0", "Y1"),
        help="条带行范围（像素含端点；行 0 在图像顶部）",
    )
    y_grp.add_argument(
        "--y-norm-range",
        type=float,
        nargs=2,
        metavar=("F0", "F1"),
        help=(
            "与 edge_strip --y-edge-frac 同一参数 F：在上带 y=(cy-b)+F*b、在下带 y=(cy+b)-F*b；"
            "指定 F 闭区间（两值不必有序，会裁剪到 [0,1]）。两数须同号：正=上带，负=下带（用绝对值作 F）。"
            "须能 estimate_crystal_boundary"
        ),
    )
    parser.add_argument("--peak-dist", type=float, default=5.0)
    parser.add_argument(
        "--col-metric",
        choices=("sum", "mean", "max"),
        default="mean",
    )
    parser.add_argument(
        "--strip-center-mode",
        choices=("com", "com_fit", "fit"),
        default="com",
    )
    parser.add_argument("--add-neighbor-x", action="store_true")
    parser.add_argument("--double-peak-fit", action="store_true")
    parser.add_argument(
        "--prominence",
        nargs="?",
        const=0.0,
        default=None,
        type=float,
    )
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub", "peel", "peel_bgsub"),
        default="raw",
    )
    parser.add_argument(
        "--draw-boundary",
        action="store_true",
        help="在背景图上叠绘 estimate_crystal_boundary（基于原图减高斯背景）",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--no-matched-filter",
        action="store_true",
        help="为 peel 准备 ions 时 detect_ions 禁用匹配滤波",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 下没有 npy 文件")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有可处理的索引")

    amp_coef_path = OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy"
    x0, x1 = float(args.x_range[0]), float(args.x_range[1])
    y_norm_tuple: tuple[float, float] | None = None
    if args.y_norm_range is not None:
        y_norm_tuple = (float(args.y_norm_range[0]), float(args.y_norm_range[1]))
    pixel_y_tuple: tuple[float, float] | None = None
    if args.y_range is not None:
        pixel_y_tuple = (float(args.y_range[0]), float(args.y_range[1]))

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions = None
        if args.preprocess in ("peel", "peel_bgsub"):
            ions, _b = detect_ions(
                image,
                fix_theta_zero=True,
                use_matched_filter=not args.no_matched_filter,
                amp_y_coef_path=amp_coef_path,
                amp_y_coef_mode="even",
            )
            if _b is None:
                print(f"[{idx:04d}] {target.name}: 无 boundary，peel 仍继续但 boundary 仅用于绘图可为空")
        try:
            strip_map, boundary_for_strip = _strip_map_from_preprocess(
                image, args.preprocess, ions,
            )
        except ValueError as e:
            print(f"[{idx:04d}] {target.name}: {e}，跳过")
            continue

        if y_norm_tuple is not None:
            if boundary_for_strip is None:
                print(f"[{idx:04d}] {target.name}: 无 boundary，无法使用 --y-norm-range，跳过")
                continue
            try:
                f_lo, f_hi, top_band = _parse_y_norm_range_f_signed(
                    y_norm_tuple[0], y_norm_tuple[1],
                )
            except ValueError as e:
                print(f"[{idx:04d}] {target.name}: {e}，跳过")
                continue
            y0, y1 = pixel_y_range_from_y_edge_frac_interval(
                boundary_for_strip,
                f_lo,
                f_hi,
                image.shape[0],
                top=top_band,
            )
        else:
            assert pixel_y_tuple is not None
            y0, y1 = pixel_y_tuple[0], pixel_y_tuple[1]

        boundary_draw: tuple | None = None
        if args.draw_boundary:
            boundary_draw = estimate_crystal_boundary(_bgsub_array(image))

        result = rect_strip_column_profiles(
            strip_map,
            x0,
            x1,
            y0,
            y1,
            col_metric=args.col_metric,
        )
        if y_norm_tuple is not None:
            band = "top" if top_band else "bottom"
            result["meta"]["y_crystal_norm"] = (
                f"F=[{f_lo:g},{f_hi:g}] {band} (same as y-edge-frac)"
            )
        prom = args.prominence
        if prom is not None and args.strip_center_mode != "fit":
            print(
                f"[{idx:04d}] 提示: --prominence 仅在 --strip-center-mode fit 下生效，已忽略。",
            )
            prom = None
        centers = fitted_xy_for_rect_strip_peaks(
            strip_map,
            result,
            peak_dist=float(args.peak_dist),
            add_neighbor_x=args.add_neighbor_x,
            center_mode=args.strip_center_mode,
            double_peak_fit=args.double_peak_fit,
            prominence_min=prom,
        )

        out_png = args.out / f"custom_strip_centers_{idx:04d}.png"
        title = _ascii_title(idx, target.name)
        print(
            f"\n[{idx:04d}] {target.name} rect x=[{result['meta']['x_lo']},{result['meta']['x_hi']}] "
            f"y=[{result['meta']['y_lo']},{result['meta']['y_hi']}] -> {len(centers)} centers"
        )
        _plot_dashboard(
            image,
            strip_map,
            result,
            centers,
            boundary_draw,
            out_png,
            title,
            float(args.peak_dist),
            show=args.show,
        )
        print(f"  saved {out_png}")


if __name__ == "__main__":
    main()
