"""
对 y 向椭圆外缘条带（与 |y-cy|/b >= 1-F 同一截线几何）按列聚合，可视化 1D 曲线并报告峰值。

边界估计与 ``detect_ions`` 一致：在 ``image - 高斯背景`` 上调用 ``estimate_crystal_boundary``。

核心算法与作图逻辑位于 ``ion_detect.edge_strip_profile_analysis`` 与
``ion_detect.edge_strip_profile_viz``；本文件为命令行入口。

用法示例:
  python edge_strip_profile.py 0
  python edge_strip_profile.py 0 5 --y-edge-frac 0.25 --preprocess raw
  python edge_strip_profile.py 0 --preprocess peel
  python edge_strip_profile.py 0 --preprocess peel --plot-peel
  python edge_strip_profile.py 0 --preprocess bgsub --col-metric mean
  python edge_strip_profile.py 0 --col-metric sum
  python edge_strip_profile.py 0 --col-metric max
  python edge_strip_profile.py 0 --peak-col-gallery
  python edge_strip_profile.py 0 --peak-col-gallery --y-fit-frac 0.35
  python edge_strip_profile.py 0 --peak-col-gallery --add-neighbor-x
  python edge_strip_profile.py 0 --plot-center
  python edge_strip_profile.py 0 --plot-center fit
  python edge_strip_profile.py 0 --plot-center com
  python edge_strip_profile.py 0 --plot-center com_fit
  python edge_strip_profile.py 0 --plot-center --double-peak-fit
  python edge_strip_profile.py 0 --plot-center --prominence
  python edge_strip_profile.py 0 --plot-center --prominence 0.5
  python edge_strip_profile.py ::10 --out outputs/edge_strip_profiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_EDGE_STRIP, PROJECT_ROOT

from ion_detect.boundary import estimate_crystal_boundary
from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.edge_strip_profile_viz import plot_edge_strip_dashboard, show_peak_column_gallery
from ion_detect.pipeline import detect_ions


def _interactive_show_blocking() -> None:
    """在 IPython/Cursor 等环境中 ``plt.show()`` 默认非阻塞，GUI 事件循环不常驻，Slider 等会无响应；强制阻塞直到窗口关闭。"""
    plt.show(block=True)


def _warn_noninteractive_backend() -> None:
    be = matplotlib.get_backend().lower()
    if be == "agg" or "inline" in be:
        print(
            "[提示] 当前 Matplotlib 后端为 "
            f"{matplotlib.get_backend()}（无可交互窗口或内联静态图）。若需拖动滑动条/单选项，"
            "请在系统终端运行本脚本并设置 GUI 后端，例如 PowerShell: "
            "$env:MPLBACKEND='TkAgg'; python edge_strip_profile.py 0 --peak-col-gallery"
        )


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
    parser.add_argument(
        "--peak-col-gallery",
        action="store_true",
        help=(
            "额外打开交互窗口：在条带轮廓线算法得到的每个辅助峰所在列上，绘制沿 y 的像素 profile，"
            "并作一维高斯拟合；滑动条切换峰序号，单选项切换上/下条带（会同时开启交互显示）。"
        ),
    )
    parser.add_argument(
        "--plot-center",
        nargs="?",
        const="fit",
        default=None,
        choices=("fit", "com", "com_fit"),
        metavar="MODE",
        help=(
            "在顶图 imshow 上标注各辅助峰列的 y 中心：省略不写则不标注；单独写 --plot-center 等价 MODE=fit。"
            "fit：列 y-profile 上单/双高斯或 --prominence；"
            "com：亮度加权质心 sum(y·w)/sum(w)（w=max(I,0)，必要时减 min(I)），无 prominence/高斯；"
            "com_fit：先算 com，再沿 y 递增 profile 上找严格局部极大（强于左右邻采样点），"
            "取与 com 距离最近的峰 y 作为标点（无局部极大则退回 com）。"
            "可与 --y-fit-frac、--add-neighbor-x 组合；--peak-col-gallery 与此 MODE 一致（仅 gallery 时默认 fit）。"
        ),
    )
    parser.add_argument(
        "--double-peak-fit",
        action="store_true",
        help=(
            "列 y 向 profile 用双高斯（两峰 + 公共偏置），标点 y 为两 mu 平均；失败回退单高斯。"
            "若同时指定 --prominence，则 y 中心以 prominence 选材为准（本项仅影响回退后的高斯曲线）。"
            "用于 --peak-col-gallery 与 --plot-center。"
        ),
    )
    parser.add_argument(
        "--prominence",
        nargs="?",
        const=0.0,
        default=None,
        type=float,
        metavar="MIN",
        help=(
            "列 y 向 profile：严格局部极大为峰候选，计算 scipy.signal.peak_prominences，"
            "仅保留 prominence≥MIN 的候选（单独写 --prominence 时 MIN=0）；再取 prominence 最大的两个峰，"
            "以其 y 坐标平均作为中心（若仅余一个候选则中心为该峰）。用于 --peak-col-gallery 与 --plot-center，"
            "且优先于 --double-peak-fit 的 y 中心；失败时回退双/单高斯逻辑。"
        ),
    )
    parser.add_argument(
        "--y-fit-frac",
        type=float,
        default=None,
        metavar="Ff",
        help=(
            "沿列采样与一维高斯拟合所用外缘条带的 F（与 --y-edge-frac 语义相同），用于 --peak-col-gallery"
            "与 --plot-center；更大的 F 一般使 y 向条带更宽。省略时与 --y-edge-frac 一致。"
            "条带 1D 轮廓与辅助峰 x 仍始终由 --y-edge-frac 决定。"
        ),
    )
    parser.add_argument(
        "--add-neighbor-x",
        action="store_true",
        help=(
            "列 y 向 profile 每个采样点为 x-1,x,x+1 三列强度之和（边界缺列则不加），"
            "用于 --peak-col-gallery 与 --plot-center。"
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
    if args.peak_col_gallery:
        args.show = True
    _viz_fit_opts = args.peak_col_gallery or (args.plot_center is not None)
    if args.y_fit_frac is not None and not _viz_fit_opts:
        print("[提示] 已设置 --y-fit-frac 但未使用 --peak-col-gallery / --plot-center，该参数将忽略。")
    if args.add_neighbor_x and not _viz_fit_opts:
        print("[提示] 已设置 --add-neighbor-x 但未使用 --peak-col-gallery / --plot-center，该参数将忽略。")
    if args.double_peak_fit and not _viz_fit_opts:
        print("[提示] 已设置 --double-peak-fit 但未使用 --peak-col-gallery / --plot-center，该参数将忽略。")
    if args.prominence is not None and not _viz_fit_opts:
        print("[提示] 已设置 --prominence 但未使用 --peak-col-gallery / --plot-center，该参数将忽略。")
    if args.show:
        plt.ioff()
        _warn_noninteractive_backend()

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
        plot_edge_strip_dashboard(
            image,
            boundary,
            result,
            out_png,
            title=_ascii_figure_title(idx, target.name),
            preprocess=args.preprocess,
            show=args.show,
            peak_dist=args.peak_dist,
            left_panel=left_panel,
            strip_map=strip_map,
            plot_center=args.plot_center,
            clip_ellipse=clip_ellipse,
            y_fit_frac=args.y_fit_frac,
            add_neighbor_x=args.add_neighbor_x,
            double_peak_fit=args.double_peak_fit,
            prominence_min=args.prominence,
        )
        if args.peak_col_gallery:
            show_peak_column_gallery(
                strip_map,
                result,
                peak_dist=args.peak_dist,
                title=_ascii_figure_title(idx, target.name),
                preprocess=args.preprocess,
                boundary=boundary,
                clip_ellipse=clip_ellipse,
                y_fit_frac=args.y_fit_frac,
                add_neighbor_x=args.add_neighbor_x,
                double_peak_fit=args.double_peak_fit,
                prominence_min=args.prominence,
                center_mode=args.plot_center if args.plot_center is not None else "fit",
            )
        if args.show:
            _interactive_show_blocking()


if __name__ == "__main__":
    main()
