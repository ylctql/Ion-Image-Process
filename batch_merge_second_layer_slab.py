"""
对多组数据目录批量执行与

    python merge_ion_centers.py <indices> --add-neighbor-x --second-layer-slab ...

等价的 second-layer-slab 流程；按组保存识别结果、带寻峰标注的合并中心 y 直方图、每帧叠加图；
命令行实时输出进度与离子数；全部完成后汇总绘制离子数/ y 分布直方图。

单目录多帧：``--data-dir``；默认索引 ``0:997``（即 997 帧）。可加 ``--add-neighbor-x`` 等与 merge_ion_centers 一致。
本脚本对**每一帧分别**完成 second-layer-slab：每帧的 y 直方图与 L1/L2/L3 寻峰**仅使用该帧**
的合并中心，再对该帧做条带替换与绘图。这与 ``merge_ion_centers.py 0:997 --second-layer-slab``
一次调用时**把多帧 y 合并成一张直方图**的行为不同。

可选 ``--batch-root``：对每个含 .npy 的子目录，仍按「每帧独立直方图」处理其索引。

用法示例（同一目录 997 帧，帧 0…996）：

  python batch_merge_second_layer_slab.py --add-neighbor-x
  （默认 ``--data-dir`` 为项目下 ``20260305_1727``，与 merge_ion_centers 一致；索引默认 ``0:997``。）

只跑单帧 800 并指定目录：

  python batch_merge_second_layer_slab.py 800 --data-dir D:/data/one_run --add-neighbor-x
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.pipeline import detect_ions
from merge_ion_centers import (  # noqa: E402
    _plot_merged,
    _strip_input_map,
    fuse_detect_strip_by_distance,
    fuse_second_layer_with_others_by_distance,
    merge_centers_hybrid,
)
from output_paths import OUT_AMP_Y_FIT, OUTPUTS_ROOT, PROJECT_ROOT
from second_layer_core import (  # noqa: E402
    _peak_indices_with_padded_ends,
    ions_from_second_layer_row,
    replace_merge_in_xy_slab,
    second_layer_y0_pair_and_slab_hi_mid23,
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _sanitize_dir_name(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name.strip())
    return s[:120] if s else "group"


def _process_one_frame(
    target: Path,
    idx: int,
    *,
    image_loader: Any,
    amp_coef_path: Path,
    clip_ellipse: bool,
    use_matched_filter: bool,
    preprocess: str,
    ex_lo: float,
    ex_hi: float,
    y_edge_frac: float,
    y_fit_frac: float,
    peak_dist: float,
    col_metric: str,
    add_neighbor_x: bool,
    strip_center_mode: str,
    ion_dist: float,
) -> tuple[np.ndarray, Any, dict, list[dict[str, Any]], int, dict] | None:
    image = image_loader(target)
    ions, boundary = detect_ions(
        image,
        fix_theta_zero=True,
        use_matched_filter=use_matched_filter,
        amp_y_coef_path=amp_coef_path,
        amp_y_coef_mode="even",
    )
    if boundary is None:
        logging.warning("[%04d] %s: 无 boundary，跳过", idx, target.name)
        return None

    try:
        strip_map, _ = _strip_input_map(
            image, boundary, preprocess, ions_for_peel=ions,
        )
    except ValueError as e:
        logging.warning("[%04d] %s: %s，跳过", idx, target.name, e)
        return None

    strip_result = outer_y_edge_column_profiles(
        strip_map,
        boundary,
        y_edge_frac,
        clip_ellipse=clip_ellipse,
        col_metric=col_metric,
    )
    merged, stats = merge_centers_hybrid(
        ions,
        boundary,
        strip_map,
        strip_result=strip_result,
        edge_x_lo=ex_lo,
        edge_x_hi=ex_hi,
        peak_dist=float(peak_dist),
        clip_ellipse=clip_ellipse,
        y_fit_frac=float(y_fit_frac),
        add_neighbor_x=add_neighbor_x,
        strip_center_mode=strip_center_mode,
    )
    merged, n_fused = fuse_detect_strip_by_distance(merged, float(ion_dist))
    stats["n_fused_detect_strip"] = n_fused
    return image, boundary, strip_result, merged, n_fused, stats


def _save_merged_y_histogram_slab(
    y_arr: np.ndarray,
    bin_edges: np.ndarray,
    counts: np.ndarray,
    peak_ix: np.ndarray,
    *,
    hist_prominence: float,
    lid1: int,
    lid2: int,
    lid3: int,
    y0_1: int,
    y0_2: int,
    yc2: float,
    yc3: float,
    y_mid: float,
    y_replace_hi: float,
    out_path: Path,
    frame_idx: int | None = None,
) -> None:
    """合并中心 y 直方图 + find_peaks 峰位 + L1/L2/L3 与替换条带上界标注。"""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(
        y_arr,
        bins=bin_edges,
        edgecolor="black",
        linewidth=0.35,
        color="steelblue",
        alpha=0.85,
    )
    peak_centers = 0.5 * (bin_edges[peak_ix] + bin_edges[peak_ix + 1])
    for yi in peak_centers:
        ax.axvline(float(yi), color="crimson", ls="--", lw=0.9, alpha=0.65)
    order = np.argsort(peak_centers)
    sorted_centers = peak_centers[order].astype(np.float64)
    for rank, (lid, color, lw) in enumerate(
        [
            (lid1, "darkgreen", 2.2),
            (lid2, "darkorange", 2.2),
            (lid3, "purple", 2.2),
        ],
        start=1,
    ):
        if 1 <= lid <= len(sorted_centers):
            yc = float(sorted_centers[lid - 1])
            ax.axvline(yc, color=color, ls="-", lw=lw, label=f"line-id={lid} → y≈{yc:.2f}")
    ax.axvline(
        float(y_replace_hi),
        color="black",
        ls="-.",
        lw=1.5,
        label=f"replace slab top y≤{y_replace_hi:.2f} (mid L{lid2}/L{lid3}={y_mid:.2f})",
    )
    ax.set_xlabel("merged center y (pixel)")
    ax.set_ylabel("count")
    title_mid = (
        f"frame [{frame_idx:04d}] | " if frame_idx is not None else ""
    )
    ax.set_title(
        f"Second-layer slab: {title_mid}merged y histogram (this frame only) | "
        f"prominence > {hist_prominence:g} | L1→y0={y0_1}, L2→y0={y0_2}",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_ions_npz(path: Path, merged: list[dict[str, Any]], *, frame_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not merged:
        np.savez_compressed(
            path,
            frame_idx=np.int32(frame_idx),
            n=np.int32(0),
            x0=np.zeros(0, dtype=np.float64),
            y0=np.zeros(0, dtype=np.float64),
            source_obj=np.array([], dtype=object),
        )
        return
    x0 = np.array([float(p["x0"]) for p in merged], dtype=np.float64)
    y0 = np.array([float(p["y0"]) for p in merged], dtype=np.float64)
    src = np.array([str(p.get("source", "")) for p in merged], dtype=object)
    np.savez_compressed(
        path,
        frame_idx=np.int32(frame_idx),
        n=np.int32(len(merged)),
        x0=x0,
        y0=y0,
        source_obj=src,
    )


def _run_one_data_dir(
    data_dir: Path,
    group_key: str,
    selected: list[int],
    files: list[Path],
    out_group: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """对单个目录逐帧 second-layer-slab：每帧仅用本帧合并中心建 y 直方图再寻峰与替换。"""
    t0 = time.perf_counter()
    amp_coef_path: Path = args.amp_coef_path
    clip_ellipse = not args.no_clip_ellipse
    ex_lo, ex_hi = float(args.edge_x_range[0]), float(args.edge_x_range[1])
    if args.profile_x_range is None:
        px_lo, px_hi = 300.0, 600.0
    else:
        px_lo, px_hi = float(args.profile_x_range[0]), float(args.profile_x_range[1])
    if px_lo > px_hi:
        px_lo, px_hi = px_lo, px_hi

    lid1 = int(args.second_layer_line_first)
    lid2 = int(args.second_layer_line_second)
    lid3 = int(args.second_layer_line_third)
    hp = float(args.second_layer_hist_prominence)
    mid_margin = float(args.second_layer_y_cut_pad)

    hist_dir = out_group / "histograms"
    ions_dir = out_group / "ions"
    fig_dir = out_group / "figures"

    def _load(p: Path) -> np.ndarray:
        return np.load(p)

    halfwin = int(args.second_layer_y_halfwin)
    com_n = max(0, int(args.second_layer_com_neighbor_cols))
    ppf = float(args.second_layer_prof_prominence_frac)
    ppd = int(args.second_layer_prof_peak_distance)

    group_ions_total = 0
    per_frame_counts: list[tuple[int, int]] = []
    n_skip_first = 0
    n_skip_second_layer = 0

    for idx in selected:
        target = files[idx]
        got = _process_one_frame(
            target,
            idx,
            image_loader=_load,
            amp_coef_path=amp_coef_path,
            clip_ellipse=clip_ellipse,
            use_matched_filter=not args.no_matched_filter,
            preprocess=args.preprocess,
            ex_lo=ex_lo,
            ex_hi=ex_hi,
            y_edge_frac=float(args.y_edge_frac),
            y_fit_frac=float(args.y_fit_frac),
            peak_dist=float(args.peak_dist),
            col_metric=args.col_metric,
            add_neighbor_x=args.add_neighbor_x,
            strip_center_mode=args.strip_center_mode,
            ion_dist=float(args.ion_dist),
        )
        if got is None:
            n_skip_first += 1
            continue
        image, boundary, strip_result, merged, _n_fused, stats = got
        ys_frame = [float(p["y0"]) for p in merged]
        if not ys_frame:
            logging.warning("[%s] [%04d] 合并中心为空，跳过", group_key, idx)
            n_skip_second_layer += 1
            continue

        y_arr = np.asarray(ys_frame, dtype=np.float64)
        y_lo = int(np.floor(y_arr.min()))
        y_hi = int(np.ceil(y_arr.max()))
        bin_edges = np.arange(y_lo, y_hi + 2, dtype=float)
        counts, _ = np.histogram(y_arr, bins=bin_edges)
        peak_ix = _peak_indices_with_padded_ends(counts, prominence=hp)

        try:
            y0_1, y0_2, yc2, yc3, y_mid, y_replace_hi = second_layer_y0_pair_and_slab_hi_mid23(
                y_arr, bin_edges, hp, lid1, lid2, lid3, mid_margin,
            )
        except RuntimeError as e:
            logging.warning("[%s] [%04d] second-layer 寻峰失败: %s", group_key, idx, e)
            n_skip_second_layer += 1
            continue

        hist_png = hist_dir / f"merged_y_histogram_peaks_{idx:04d}.png"
        _save_merged_y_histogram_slab(
            y_arr,
            bin_edges,
            counts,
            peak_ix,
            hist_prominence=hp,
            lid1=lid1,
            lid2=lid2,
            lid3=lid3,
            y0_1=y0_1,
            y0_2=y0_2,
            yc2=yc2,
            yc3=yc3,
            y_mid=y_mid,
            y_replace_hi=y_replace_hi,
            out_path=hist_png,
            frame_idx=idx,
        )

        im64 = np.asarray(image, dtype=np.float64)
        ions_l1 = ions_from_second_layer_row(
            im64, y0_1, px_lo, px_hi,
            halfwin=halfwin,
            prof_prominence_frac=ppf,
            prof_peak_distance=ppd,
            source="second_layer_L1",
            com_neighbor_cols=com_n,
        )
        ions_l2 = ions_from_second_layer_row(
            im64, y0_2, px_lo, px_hi,
            halfwin=halfwin,
            prof_prominence_frac=ppf,
            prof_peak_distance=ppd,
            source="second_layer_L2",
            com_neighbor_cols=com_n,
        )
        merged2 = list(merged)
        n_before = len(merged2)
        merged2 = replace_merge_in_xy_slab(
            merged2, px_lo, px_hi, 0.0, y_replace_hi, ions_l1 + ions_l2,
        )
        n_removed_slab = n_before - (len(merged2) - len(ions_l1) - len(ions_l2))

        ion_thr = float(args.ion_dist)
        if ion_thr > 0.0:
            merged2, n_sl_fuse = fuse_second_layer_with_others_by_distance(merged2, ion_thr)
        else:
            n_sl_fuse = 0

        stem = Path(target.name).stem
        safe = stem.encode("ascii", "replace").decode("ascii")
        title = (
            f"frame {idx:04d} ({safe}) | second-layer slab (per-frame y hist) | "
            f"L1 n={len(ions_l1)}, L2 n={len(ions_l2)} | "
            f"fused {n_sl_fuse} | total {len(merged2)} | "
            f"detect {stats['n_detect_kept']}/{stats['n_detect_raw']}, strip+{stats['n_strip_used']}"
        )
        out_png = fig_dir / f"ion_centers_merged_{idx:04d}.png"
        _plot_merged(
            image,
            boundary,
            strip_result,
            merged2,
            ex_lo,
            ex_hi,
            out_png,
            title,
            show=False,
        )
        npz_path = ions_dir / f"ions_{idx:04d}.npz"
        _save_ions_npz(npz_path, merged2, frame_idx=idx)

        n_ions = len(merged2)
        group_ions_total += n_ions
        per_frame_counts.append((idx, n_ions))

        logging.info(
            "[%s] [%04d] %s | 离子数=%d | L1=%d L2=%d slab_rm=%d | hist %s",
            group_key,
            idx,
            target.name,
            n_ions,
            len(ions_l1),
            len(ions_l2),
            n_removed_slab,
            hist_png.name,
        )

    elapsed = time.perf_counter() - t0
    n_ok = len(per_frame_counts)
    logging.info(
        "[%s] 完成 | 请求帧 %d | 成功 second-layer %d | 首遍跳过 %d | 寻峰/空中心跳过 %d | "
        "离子总数=%d | %.1fs",
        group_key,
        len(selected),
        n_ok,
        n_skip_first,
        n_skip_second_layer,
        group_ions_total,
        elapsed,
    )

    err: str | None = None
    if n_ok == 0 and len(selected) > 0:
        err = "no successful frames"

    out: dict[str, Any] = {
        "group": group_key,
        "data_dir": str(data_dir),
        "n_frames_requested": len(selected),
        "n_frames_second_layer_ok": n_ok,
        "n_skipped_first_pass": n_skip_first,
        "n_skipped_second_layer": n_skip_second_layer,
        "n_ions_total": group_ions_total,
        "per_frame": [(int(i), int(n)) for i, n in per_frame_counts],
        "seconds": float(elapsed),
    }
    if err:
        out["error"] = err
    return out


def _discover_batch_dirs(batch_root: Path) -> list[Path]:
    subs = [p for p in sorted(batch_root.iterdir()) if p.is_dir()]
    out: list[Path] = []
    for p in subs:
        if any(p.glob("*.npy")):
            out.append(p)
    return out


def _plot_summary(
    summaries: list[dict[str, Any]],
    all_y: np.ndarray,
    all_group_idx: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [s for s in summaries if s.get("n_frames_second_layer_ok", 0) > 0]
    names = [s["group"] for s in valid]
    counts = [int(s["n_ions_total"]) for s in valid]

    if names:
        fig, ax = plt.subplots(figsize=(max(10, 0.12 * len(names)), 4.5))
        x = np.arange(len(names))
        ax.bar(x, counts, color="steelblue", edgecolor="k", linewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("离子数")
        ax.set_title("各组识别离子总数")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_ions_per_group.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all_y.size > 0:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(all_y, bins="auto", color="coral", edgecolor="k", linewidth=0.35, alpha=0.85)
        ax.set_xlabel("y (pixel)")
        ax.set_ylabel("计数（所有组所有离子）")
        ax.set_title(f"合并 y 直方图 | N_ions={all_y.size}")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_all_y_histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all_y.size > 0 and all_group_idx.size == all_y.size:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _, _, _, im = ax.hist2d(
            all_group_idx.astype(np.float64),
            all_y,
            bins=(50, 80),
            cmap="Blues",
            cmin=1,
        )
        ax.set_xlabel("组序号（按批处理顺序）")
        ax.set_ylabel("y (pixel)")
        ax.set_title("组索引 vs y（2D 直方图）")
        fig.colorbar(im, ax=ax, label="计数")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_group_index_vs_y.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    flat: list[tuple[str, int, int]] = []
    for s in summaries:
        gk = str(s.get("group", ""))
        for fi, ni in s.get("per_frame", []):
            flat.append((gk, int(fi), int(ni)))
    if len(flat) >= 2:
        fig, ax = plt.subplots(figsize=(min(28, 0.028 * len(flat) + 2), 4.2))
        xp = np.arange(len(flat))
        ax.bar(xp, [t[2] for t in flat], color="teal", edgecolor="k", linewidth=0.2)
        ax.set_xlabel("处理序号（各组帧顺排）")
        ax.set_ylabel("该帧离子数")
        ax.set_title(f"逐帧离子数 | 共 {len(flat)} 条")
        if len(flat) <= 80:
            ax.set_xticks(xp)
            ax.set_xticklabels([f"{t[1]}" for t in flat], rotation=90, fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / "summary_ions_per_frame.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对每帧单独跑 second-layer-slab（每帧自建 y 直方图）。"
            "多目录时各子目录同样按帧独立处理。"
        ),
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0:997"],
        help="帧索引（与 merge_ion_centers 相同；默认 0:997 即 997 帧 0…996）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "20260305_1727",
        help="单个含 .npy 的数据目录（默认与 merge_ion_centers.py 相同）",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help=(
            "若指定：对该目录下每个含 .npy 的子目录各跑一批；"
            "未指定时仅处理 --data-dir"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出根目录；默认 outputs/batch_merge_slab_<时间戳>",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="同时将日志写入文件")
    parser.add_argument(
        "--amp-coef",
        type=Path,
        default=OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy",
        help="detect_ions 幅度系数 npy",
    )

    parser.add_argument("--edge-x-range", type=float, nargs=2, default=[250.0, 750.0])
    parser.add_argument("--y-edge-frac", type=float, default=0.25)
    parser.add_argument("--y-fit-frac", type=float, default=0.35)
    parser.add_argument("--peak-dist", type=float, default=5.0)
    parser.add_argument("--col-metric", choices=("sum", "mean", "max"), default="mean")
    parser.add_argument("--strip-center-mode", choices=("com", "com_fit", "fit"), default="com")
    parser.add_argument("--add-neighbor-x", action="store_true")
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub", "peel", "peel_bgsub"),
        default="raw",
    )
    parser.add_argument("--no-clip-ellipse", action="store_true")
    parser.add_argument("--no-matched-filter", action="store_true")
    parser.add_argument("--ion-dist", type=float, default=4.0)

    parser.add_argument("--profile-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--second-layer-hist-prominence", type=float, default=5.0)
    parser.add_argument("--second-layer-prof-prominence-frac", type=float, default=0.08)
    parser.add_argument("--second-layer-prof-peak-distance", type=int, default=4)
    parser.add_argument("--second-layer-y-halfwin", type=int, default=3)
    parser.add_argument("--second-layer-com-neighbor-cols", type=int, default=1)
    parser.add_argument("--second-layer-line-first", type=int, default=1)
    parser.add_argument("--second-layer-line-second", type=int, default=2)
    parser.add_argument("--second-layer-line-third", type=int, default=3)
    parser.add_argument("--second-layer-y-cut-pad", type=float, default=1.0)

    args = parser.parse_args()
    args.amp_coef_path = args.amp_coef

    root_out = args.out
    if root_out is None:
        root_out = OUTPUTS_ROOT / f"batch_merge_slab_{time.strftime('%Y%m%d_%H%M%S')}"
    root_out = root_out.resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logging.info("输出根目录: %s", root_out)

    if args.batch_root is not None:
        batch_root = args.batch_root.resolve()
        groups = _discover_batch_dirs(batch_root)
        if not groups:
            raise SystemExit(f"{batch_root} 下未找到含 .npy 的子目录")
        logging.info("批处理目录数: %d（父目录 %s）", len(groups), batch_root)
    else:
        data_dir = args.data_dir.resolve()
        groups = [data_dir]
        logging.info("单目录模式: %s", data_dir)

    summaries: list[dict[str, Any]] = []
    all_y_list: list[float] = []
    all_gidx_list: list[int] = []

    for gix, data_dir in enumerate(groups):
        files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
        if not files:
            logging.warning("跳过（无 .npy）: %s", data_dir)
            continue
        group_key = _sanitize_dir_name(data_dir.name) if args.batch_root else _sanitize_dir_name(data_dir.name)
        out_group = root_out / f"{gix:04d}_{group_key}"

        selected = resolve_indices(args.indices, len(files))
        if not selected:
            logging.warning("[%s] 索引为空，跳过", group_key)
            continue

        logging.info(
            "======== 组 %d/%d: %s | 帧数=%d | 处理索引 %s ========",
            gix + 1,
            len(groups),
            data_dir,
            len(files),
            ",".join(args.indices) if args.indices else "0",
        )

        info = _run_one_data_dir(
            data_dir, group_key, selected, files, out_group, args,
        )
        summaries.append(info)

        if "error" not in info and info.get("per_frame"):
            ions_dir = out_group / "ions"
            for idx, _ in info["per_frame"]:
                p = ions_dir / f"ions_{idx:04d}.npz"
                if p.is_file():
                    z = np.load(p, allow_pickle=True)
                    y0 = z["y0"]
                    all_y_list.extend(y0.astype(np.float64).tolist())
                    all_gidx_list.extend([gix] * int(y0.shape[0]))

    summary_dir = root_out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    all_y = np.asarray(all_y_list, dtype=np.float64)
    all_gidx = np.asarray(all_gidx_list, dtype=np.int32)
    _plot_summary(summaries, all_y, all_gidx, summary_dir)

    grand = int(all_y.size)
    ok_groups = sum(1 for s in summaries if s.get("n_frames_second_layer_ok", 0) > 0)
    logging.info(
        "======== 全部完成 | 处理组数(有输出)=%d | 离子总数=%d | 汇总目录 %s ========",
        ok_groups,
        grand,
        summary_dir,
    )


if __name__ == "__main__":
    main()
