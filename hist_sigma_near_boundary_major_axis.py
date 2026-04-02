"""
对多帧离子图运行与 ``detect_ions`` 相同流水线（高斯减背景 + 匹配滤波 +
maximum_filter 峰值 + 高斯拟合），筛出拟合中心满足 ``|y0 - cy|`` 不超过给定像素数的离子
（**boundary 长轴约定为沿 x**，即过长轴为 ``y = cy``），汇总 ``_sigma_x`` / ``_sigma_y``
并绘制直方图；默认另存每帧识别图（亮绿标出 ``|y0-cy|≤tol`` 的离子，金虚线为长轴）。

细节见 :func:`ion_detect.boundary.offset_perpendicular_to_boundary_major_axis` 与
:func:`ion_detect.viz.visualize` 的 ``near_major_axis_tol``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.boundary import offset_perpendicular_to_boundary_major_axis
from ion_detect.cli_helpers import resolve_indices
from ion_detect.pipeline import detect_ions
from ion_detect.viz import visualize
from output_paths import DEFAULT_DATA_DIR, OUT_HISTOGRAM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="长轴附近离子 sigma_x/sigma_y 直方图（复用 detect_ions 全流程）",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引，与 python -m ion_detect 相同（支持切片并集）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f".npy 数据目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--major-axis-tol",
        type=float,
        default=3.0,
        help="|y0 - cy| 上限 (像素)；长轴视为沿 x (y=cy)，默认 3",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="直方图 bin 数量",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG 路径 (默认 outputs/histogram/sigma_xy_near_major_axis.png)",
    )
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="可选：保存 sigma_x, sigma_y, offsets 等到此 .npz",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help=(
            "保存每帧识别可视化 PNG 的目录；默认 outputs/histogram/near_major_axis_detect。"
            "与 --no-viz 同用时以 --no-viz 为准。"
        ),
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不保存识别结果图",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录中无 .npy: {data_dir}")

    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有有效索引")

    all_sx: list[float] = []
    all_sy: list[float] = []
    all_off: list[float] = []
    n_frames_ok = 0
    n_frames_no_boundary = 0

    viz_dir: Path | None = None
    if not args.no_viz:
        viz_dir = args.viz_dir
        if viz_dir is None:
            viz_dir = OUT_HISTOGRAM / "near_major_axis_detect"
        viz_dir = viz_dir.resolve()
        viz_dir.mkdir(parents=True, exist_ok=True)

    tol = float(args.major_axis_tol)

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions, boundary = detect_ions(image)
        if boundary is None:
            n_frames_no_boundary += 1
            continue
        n_frames_ok += 1
        if viz_dir is not None:
            n_near = sum(
                1
                for ion in ions
                if abs(float(ion["y0"]) - float(boundary[1])) <= tol
            )
            vis_path = viz_dir / f"detect_near_axis_{idx:04d}.png"
            visualize(
                image,
                ions,
                n_sigma=2.0,
                title=f"[{idx:04d}] {target.name}  |  {n_near}/{len(ions)} near axis",
                output_path=vis_path,
                boundary=boundary,
                show_fit_quality=True,
                show=False,
                near_major_axis_tol=tol,
            )
            print(
                f"[frame {idx:04d}] ions={len(ions)}, |y-cy|≤{tol:g}: {n_near}  -> {vis_path}"
            )
        for ion in ions:
            off = offset_perpendicular_to_boundary_major_axis(
                ion["x0"], ion["y0"], boundary,
            )
            if off is None or abs(off) > tol:
                continue
            if "_sigma_x" not in ion or "_sigma_y" not in ion:
                continue
            all_sx.append(float(ion["_sigma_x"]))
            all_sy.append(float(ion["_sigma_y"]))
            all_off.append(float(off))

    sx = np.asarray(all_sx, dtype=np.float64)
    sy = np.asarray(all_sy, dtype=np.float64)

    out_png = args.out
    if out_png is None:
        OUT_HISTOGRAM.mkdir(parents=True, exist_ok=True)
        out_png = OUT_HISTOGRAM / "sigma_xy_near_major_axis.png"
    else:
        out_png = out_png.resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)

    # sans-serif 列表会按顺序匹配；显式赋值以覆盖 matplotlib 默认的 DejaVu 优先
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    suptitle = (
        f"σx / σy (|y0-cy|≤{tol:g} px, major axis along x; "
        f"detect_ions) — N={sx.size} from {n_frames_ok} frames"
    )
    if n_frames_no_boundary:
        suptitle += f"; {n_frames_no_boundary} frames skipped (no boundary)"
    fig.suptitle(suptitle, fontsize=11)

    bins = max(5, int(args.bins))

    def _hist_stats_box_text(arr: np.ndarray) -> str:
        if arr.size == 0:
            return ""
        mu = float(np.mean(arr))
        med = float(np.median(arr))
        if arr.size > 1:
            sd_txt = f"{float(np.std(arr, ddof=1)):.4f}"
        else:
            sd_txt = "—"
        return f"均值 μ = {mu:.4f}\n标准差 s = {sd_txt}\n中位数 = {med:.4f}\nN = {arr.size}"

    if sx.size > 0:
        axes[0].hist(sx, bins=bins, color="steelblue", edgecolor="black", linewidth=0.35, alpha=0.85)
        mu_x = float(np.mean(sx))
        axes[0].axvline(mu_x, color="darkviolet", ls="-", lw=1.5, label=f"均值 μ={mu_x:.4f}")
        axes[0].axvline(float(np.median(sx)), color="crimson", ls="--", lw=1.2, label=f"中位数={np.median(sx):.4f}")
        stx = _hist_stats_box_text(sx)
        axes[0].text(
            0.98, 0.97, stx, transform=axes[0].transAxes, va="top", ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.62),
        )
        axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_xlabel("σx (pixel)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("σx")

    if sy.size > 0:
        axes[1].hist(sy, bins=bins, color="darkseagreen", edgecolor="black", linewidth=0.35, alpha=0.85)
        mu_y = float(np.mean(sy))
        axes[1].axvline(mu_y, color="darkviolet", ls="-", lw=1.5, label=f"均值 μ={mu_y:.4f}")
        axes[1].axvline(float(np.median(sy)), color="crimson", ls="--", lw=1.2, label=f"中位数={np.median(sy):.4f}")
        sty = _hist_stats_box_text(sy)
        axes[1].text(
            0.98, 0.97, sty, transform=axes[1].transAxes, va="top", ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.62),
        )
        axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_xlabel("σy (pixel)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("σy")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}  (N={sx.size})")

    if args.save_npz is not None:
        path = args.save_npz.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        off_arr = np.asarray(all_off, dtype=np.float64)
        np.savez(path, sigma_x=sx, sigma_y=sy, offset_perp_major=off_arr, major_axis_tol=tol)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
