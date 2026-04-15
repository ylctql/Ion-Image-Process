"""python -m ion_detect.blob_cli — 连通域 + 轴对齐最小外接矩形批处理。"""
from __future__ import annotations

import argparse
from typing import Literal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import (
    DEFAULT_DATA_DIR,
    OUT_BLOB_CONNECTED,
    OUT_BLOB_MERGE_SPLIT_HIST_PNG,
    OUT_BLOB_MERGE_SPLIT_LOG,
    OUT_PIXEL_HIST,
)

from ion_detect.blob_ion_positions import ion_equilibrium_positions_xy, merge_close_ion_positions_xy
from ion_detect.blob_viz import visualize_blob_workflow
from ion_detect.blob_preprocess import ellipse_interior_mask
from ion_detect.blob_workflow import run_blob_workflow
from ion_detect.cli_helpers import resolve_indices


def _write_pixel_brightness_hist(
    denoised_map: np.ndarray,
    output_path: Path,
    *,
    boundary: tuple[float, float, float, float] | None,
    frame_idx: int,
    threshold: float,
    use_bgsub: bool,
    use_matched_filter: bool,
    brightness_xlabel: str | None = None,
) -> None:
    """
    Histogram of pixels **inside the crystal boundary ellipse** on ``denoised_map``.
    Marks ``threshold`` with a vertical line and reports % of ellipse pixels with value > T.
    Figure text in English; title includes ``frame_idx``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if boundary is None:
        print("Warning: no crystal boundary for --plot-pixel-hist; skip file.")
        return
    zmap = np.asarray(denoised_map, dtype=np.float64)
    m = ellipse_interior_mask(zmap.shape, boundary)
    z = zmap[m]
    z = z[np.isfinite(z)]
    if z.size == 0:
        print("Warning: no finite pixels inside boundary for --plot-pixel-hist; skip file.")
        return
    t = float(threshold)
    n_above = int(np.count_nonzero(z > t))
    pct_above = 100.0 * float(n_above) / float(z.size)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(z, bins="auto", color="steelblue", edgecolor="white", linewidth=0.25)
    ax.axvline(t, color="crimson", linestyle="--", linewidth=2.0, zorder=4, label=f"Threshold T = {t:g}")
    ax.set_xlabel(
        brightness_xlabel
        if brightness_xlabel is not None
        else "Brightness (denoised map)",
    )
    ax.set_ylabel("Pixel count")
    prep_parts: list[str] = (
        ["Gaussian bgsub"] if use_bgsub else ["no bgsub"]
    ) + (
        ["matched filter"] if use_matched_filter else ["no matched filter"]
    )
    ax.set_title(
        f"Frame index {frame_idx:04d} — inside boundary ellipse — "
        f"{', '.join(prep_parts)} — n = {int(z.size)} pixels\n"
        f"P(brightness > T) = {pct_above:.2f}%  (T = {t:g})",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output_path}")


def _write_merge_split_hist(counts: list[int], output_path: Path) -> None:
    """Save histogram of final ion counts; all figure text in English."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(counts, bins="auto", color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Final ion count (after y-split / x-refine / ion-dist merge)")
    ax.set_ylabel("Number of frames")
    ax.set_title(
        f"Distribution of final ion counts (n = {len(counts)} frames)",
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_blob_workflow：可选减背景（默认开）与可选匹配滤波 → 在该浮点图上二值化 → boundary → 连通域 → 轴对齐矩形；"
        "输出上下两幅子图 PNG（上：二值化前浮点图+色标与预处理说明；下：二值+矩形）",
    )
    parser.add_argument("indices", nargs="*", default=["0"], help="帧索引（与 ion_detect 相同切片语法）")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="含 .npy 的数据目录",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        metavar="T",
        help="二值化阈值：默认作用在 denoised_map；若 --thr-norm p95 / p95_all 则作用在 z/scale（见 --thr-norm-pct）",
    )
    parser.add_argument(
        "--thr-norm",
        type=str,
        choices=("none", "p95", "p95_all"),
        default="p95",
        help="帧间统一阈值：none=在 denoised_map 上直接用 T；"
        "p95=椭圆内仅正值子集的 P 分位作 scale（旧行为）；"
        "p95_all=椭圆内全体有限像素的有符号 P 分位作 scale（非 |z|；无椭圆则全图有限像素）",
    )
    parser.add_argument(
        "--thr-norm-pct",
        type=float,
        default=1.0,
        metavar="P",
        help="与 --thr-norm p95 / p95_all 配合：分位数 P∈[1,100]；p95 时尺度为正值子集 P 分位，p95_all 时为 ROI 内有符号分布的 P 分位",
    )
    parser.add_argument(
        "--no-bgsub",
        action="store_true",
        help="不做高斯减背景：在原始图像浮点上继续做可选匹配滤波并二值化（默认先做 bgsub）",
    )
    parser.add_argument(
        "--matched-filter",
        action="store_true",
        help="对当前 signal 做与 detect_ions 相同的匹配滤波后再二值化",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="连通域邻接：4 或 8（默认 4）",
    )
    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=1,
        metavar="N",
        help="忽略像素数 < N 的连通域",
    )
    parser.add_argument(
        "--no-merge-small-rects",
        action="store_true",
        help="关闭椭圆 y 外缘带内过薄小矩形与最近邻 AABB 合并（默认会进行合并）",
    )
    parser.add_argument(
        "--y-edge-frac",
        type=float,
        default=0.35,
        help="外缘带参数 F（默认 0.35），与 outer_y_edge_strip_masks 一致",
    )
    parser.add_argument(
        "--min-edge-ysize",
        type=float,
        default=5.0,
        help="y 向边长小于该值的条带内矩形才参与合并（默认 5）",
    )
    parser.add_argument(
        "--no-merge-band-clip-ellipse",
        action="store_true",
        help="条带掩膜不按椭圆裁剪（与 --no-clip-ellipse 类脚本对应）",
    )
    parser.add_argument(
        "--no-pre-merge-drop",
        action="store_true",
        help="不在 merge 前剔除两轴跨度均 ≤1 的矩形（默认会先剔除再 merge / 可视化 split）",
    )
    parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="合并后若矩形 y 向跨度 > max_ysize，则按 ceil(height/max_ysize) 等分 y 条带，"
        "条带内对二值前景求质心作为离子平衡位置；否则仍为矩形中心（洋红 + 标在各图上）；默认开，可用 --no-split 关闭",
    )
    parser.add_argument(
        "--max-ysize",
        type=float,
        default=9.0,
        metavar="Y",
        help="与 --split 配合：y 向边长阈值（默认 9）；仅当跨度更大时才分割",
    )
    parser.add_argument(
        "--refine-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每个 y 子带（含未 y 细分的单带）内：按列对 y 求二值前景均值，"
        "大于 --x-profile-threshold 的 x 列连成段，每段内用去噪浮点图正部加权求质心；"
        "多段对应多个离子。建议与 --split 同用；默认开，可用 --no-refine-x 关闭",
    )
    parser.add_argument(
        "--x-profile-threshold",
        type=float,
        default=0.4,
        metavar="P",
        help="x 向列阈值 P：默认与条带内 y 向二值占有率 (0–1) 比较；若加 --x-profile-rel-to-max 则与 P*max(col) 比较",
    )
    parser.add_argument(
        "--x-profile-rel-to-max",
        action="store_true",
        help="列掩膜改为 col_mean > P * max(col_mean)，适合离子 y 向很薄、绝对占有率难超过 P 时",
    )
    parser.add_argument(
        "--ion-dist",
        type=float,
        default=5.0,
        metavar="D",
        help="识别位置（含 x 细化）全部求出后：若两位置欧氏距离 < D 像素则合并（最近邻对优先）；"
        "合并点为去噪图正部加权质心；D≤0 关闭（默认 5）",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Append ion-count summary (TSV) to outputs/blob/merge_split.log；"
        "列为 frame_idx…final_ions（after_equilibrium=split/x-refine 后；final_ions=ion_dist 合并后）、"
        "预处理与 thr_norm 等，便于与当前 CLI 参数一一对应",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Save histogram of final ion counts to outputs/blob/hist_merge_split.png (English labels)",
    )
    parser.add_argument(
        "--plot-pixel-hist",
        action="store_true",
        help="Per frame: histogram of denoised_map inside boundary ellipse; marks --threshold T "
        "and P(brightness > T) among ellipse pixels; outputs/pixel_hist/ (English labels)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗：除双栏结果外，另开空域 brightness 分布图（RdBu_r/色标与 ion_detect 附录 bgsub 图一致），再显示双栏 PNG 内容",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 无 .npy")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("索引为空")

    out_dir = OUT_BLOB_CONNECTED
    out_dir.mkdir(parents=True, exist_ok=True)

    log_f = None
    if args.log:
        log_path = OUT_BLOB_MERGE_SPLIT_LOG
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        log_f = open(log_path, "a", encoding="utf-8")
        if write_header:
            log_f.write(
                "frame_idx\tfile\tinitial_ions\tafter_drop_merge\tafter_equilibrium\tfinal_ions\t"
                "threshold\tbgsub\tmatched_filter\tconnectivity\tmin_area_pixels\t"
                "split_on\tmax_ysize\trefine_x\tx_profile_threshold\tx_profile_rel_to_max\t"
                "elapsed_s\tlabels\tpre_merge_dropped\tedge_merges\tion_dist\tn_ion_dist_merge\t"
                "thr_norm\tthr_norm_pct\tthr_norm_scale\n",
            )
        print(f"Logging to {log_path.resolve()}")

    final_ion_counts: list[int] = []

    try:
        for idx in selected:
            target = files[idx]
            print(f"\n[{idx:04d}] {target.name}")
            image = np.load(target)
            if image.ndim == 3:
                image = image.mean(axis=0)
            image = np.asarray(image, dtype=np.float64)

            conn: Literal[4, 8] = 4 if args.connectivity == 4 else 8
            t0 = time.perf_counter()
            thr_norm_kw: Literal["none", "p95", "p95_all"]
            if args.thr_norm == "p95_all":
                thr_norm_kw = "p95_all"
            elif args.thr_norm == "p95":
                thr_norm_kw = "p95"
            else:
                thr_norm_kw = "none"
            res = run_blob_workflow(
                image,
                args.threshold,
                use_bgsub=not args.no_bgsub,
                use_matched_filter=args.matched_filter,
                connectivity=conn,
                min_area_pixels=int(args.min_area_pixels),
                merge_small_rects=not args.no_merge_small_rects,
                y_edge_frac=float(args.y_edge_frac),
                min_edge_ysize=float(args.min_edge_ysize),
                merge_band_clip_ellipse=not args.no_merge_band_clip_ellipse,
                pre_merge_drop_max_span=None if args.no_pre_merge_drop else 1.0,
                thr_norm=thr_norm_kw,
                thr_norm_percentile=float(args.thr_norm_pct),
            )
            elapsed = time.perf_counter() - t0
            n_initial = res.n_rects_after_labeling
            n_after_drop_merge = len(res.rects)
            eq_xy = ion_equilibrium_positions_xy(
                res.rects,
                res.binary,
                labeled=res.labeled,
                split=args.split,
                max_ysize=float(args.max_ysize),
                refine_x=args.refine_x,
                x_profile_threshold=float(args.x_profile_threshold),
                x_profile_rel_to_max=args.x_profile_rel_to_max,
                intensity=res.preprocess.denoised_map,
            )
            if float(args.ion_dist) > 0.0:
                final_xy, n_ion_dist_merge = merge_close_ion_positions_xy(
                    eq_xy,
                    float(args.ion_dist),
                    intensity=res.preprocess.denoised_map,
                )
            else:
                final_xy, n_ion_dist_merge = eq_xy, 0
            n_final_ions = len(final_xy)
            n_after_equilibrium = len(eq_xy)
            print(
                f"  labels={res.n_components}, kept_rects={n_after_drop_merge}, "
                f"after_equilibrium={n_after_equilibrium}, final_ions={n_final_ions}, "
                f"pre_merge_dropped={res.n_rects_dropped_pre_merge}, "
                f"edge_sliver_merges={res.n_edge_sliver_merges}, "
                f"ion_dist_merges={n_ion_dist_merge}, "
                f"boundary={'OK' if res.preprocess.boundary else 'None'}, {elapsed:.2f}s",
            )

            if log_f is not None:
                scale_s = (
                    f"{res.thr_norm_scale:g}"
                    if res.thr_norm_scale is not None
                    else ""
                )
                pct_s = (
                    f"{res.thr_norm_percentile:g}"
                    if res.thr_norm_percentile is not None
                    else ""
                )
                log_f.write(
                    f"{idx:04d}\t{target.name}\t{n_initial}\t{n_after_drop_merge}\t"
                    f"{n_after_equilibrium}\t{n_final_ions}\t"
                    f"{args.threshold:g}\t{0 if args.no_bgsub else 1}\t"
                    f"{1 if args.matched_filter else 0}\t{int(args.connectivity)}\t"
                    f"{int(args.min_area_pixels)}\t"
                    f"{1 if args.split else 0}\t{float(args.max_ysize):g}\t"
                    f"{1 if args.refine_x else 0}\t{float(args.x_profile_threshold):g}\t"
                    f"{1 if args.x_profile_rel_to_max else 0}\t"
                    f"{elapsed:.4f}\t{res.n_components}\t{res.n_rects_dropped_pre_merge}\t"
                    f"{res.n_edge_sliver_merges}\t{float(args.ion_dist):g}\t{n_ion_dist_merge}\t"
                    f"{res.thr_norm}\t{pct_s}\t{scale_s}\n",
                )
                log_f.flush()

            if args.hist:
                final_ion_counts.append(n_final_ions)

            stem = target.stem
            safe = stem.encode("ascii", "replace").decode("ascii")
            thr_s = f"{args.threshold:g}".replace(".", "p").replace("-", "m")
            png = out_dir / f"blob_workflow_{idx:04d}_{safe}_thr{thr_s}.png"
            float_for_thr = (
                res.binarize_float_map
                if res.binarize_float_map is not None
                else res.preprocess.denoised_map
            )
            hist_xlabel: str | None = None
            if res.thr_norm == "p95" and res.thr_norm_scale is not None:
                p = res.thr_norm_percentile if res.thr_norm_percentile is not None else args.thr_norm_pct
                hist_xlabel = (
                    f"z / scale (binarize input); scale = P{p:g}(+) in ellipse ROI = {res.thr_norm_scale:g}"
                )
            elif res.thr_norm == "p95_all" and res.thr_norm_scale is not None:
                p = res.thr_norm_percentile if res.thr_norm_percentile is not None else args.thr_norm_pct
                hist_xlabel = (
                    f"z / scale (binarize input); scale = P{p:g}(signed ROI) = {res.thr_norm_scale:g}"
                )
            if args.plot_pixel_hist:
                OUT_PIXEL_HIST.mkdir(parents=True, exist_ok=True)
                hist_px_path = OUT_PIXEL_HIST / (
                    f"blob_pixel_brightness_hist_{idx:04d}_{safe}_thr{thr_s}.png"
                )
                _write_pixel_brightness_hist(
                    float_for_thr,
                    hist_px_path,
                    boundary=res.preprocess.boundary,
                    frame_idx=int(idx),
                    threshold=float(args.threshold),
                    use_bgsub=not args.no_bgsub,
                    use_matched_filter=args.matched_filter,
                    brightness_xlabel=hist_xlabel,
                )
            visualize_blob_workflow(
                float_for_thr,
                res.binary,
                boundary=res.preprocess.boundary,
                rects=res.rects,
                title=f"[{idx:04d}] {target.name}",
                threshold=float(args.threshold),
                use_bgsub=not args.no_bgsub,
                use_matched_filter=args.matched_filter,
                output_path=png,
                show=args.show,
                n_edge_sliver_merges=res.n_edge_sliver_merges,
                rect_y_split=args.split,
                max_ysize=float(args.max_ysize),
                refine_x=args.refine_x,
                x_profile_threshold=float(args.x_profile_threshold),
                x_profile_rel_to_max=args.x_profile_rel_to_max,
                labeled=res.labeled,
                y_edge_frac=float(args.y_edge_frac),
                merge_band_clip_ellipse=not args.no_merge_band_clip_ellipse,
                ion_xy=final_xy,
                thr_norm=res.thr_norm,
                thr_norm_percentile=res.thr_norm_percentile,
                thr_norm_scale=res.thr_norm_scale,
            )
    finally:
        if log_f is not None:
            log_f.close()

    if args.hist:
        if not final_ion_counts:
            print("Warning: --hist set but no frames processed; skip histogram.")
        else:
            _write_merge_split_hist(final_ion_counts, OUT_BLOB_MERGE_SPLIT_HIST_PNG)


if __name__ == "__main__":
    main()
