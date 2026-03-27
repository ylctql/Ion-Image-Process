"""
合并 ion_detect（中心区域）与外缘条带列向质心（上下边缘、中间 x 范围）。

规则简述：
- 对落在「条带优先」上下域内、且与某 strip 中心距离 ≤ peak-dist 的 detect 与 strip 成对：在 strip_map 上对二者种子做联合双峰拟合（与 ion_detect 同风格的轴对齐双高斯）。
  若拟合成功且两峰间距 < ion-dist，则输出一点（坐标为两拟合峰均值，source=fused_bimodal，参量继承 detect）；
  若拟合成功且间距 ≥ ion-dist，则保留两拟合峰（source=fused_split）。拟合失败则退回种子坐标算术均值（source=fused_bimodal_fallback）。
- 条带优先域内、未与 strip 配对的 detect 仍丢弃（由 strip/COM 代表该域）；未配对的 strip 与其它位置的 detect 照常保留。
- 其余椭圆内区域仅采用 ion_detect；条带在 x 域外的峰仍丢弃。

示例：
  python merge_ion_centers.py 0
  python merge_ion_centers.py 0 --edge-x-range 250 750 --y-edge-frac 0.25 \\
    --y-fit-frac 0.35 --add-neighbor-x --peak-dist 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_AMP_Y_FIT, OUT_ION_CENTERS_MERGED, PROJECT_ROOT

from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.edge_strip_profile_analysis import fitted_xy_for_auxiliary_strip_peaks
from ion_detect.fitting import fit_joint_two_peaks_at
from ion_detect.pipeline import detect_ions

# 与 detect_ions 默认一致，用于条带域内 detect–strip 联合双峰拟合
_DEFAULT_FIT_HW: tuple[int, int] = (4, 3)
_DEFAULT_SIGMA_FIT_RANGE: tuple[float, float] = (0.3, 3.5)
_BIMODAL_SIGMA_INIT: tuple[float, float] = (1.2, 1.8)


def _greedy_pair_detect_strip(
    n_det: int,
    n_strip: int,
    det_pts: list[dict[str, Any]],
    strip_pts: list[dict[str, Any]],
    peak_dist: float,
) -> list[tuple[int, int]]:
    """在 detect 与 strip 之间贪心配对：每次取距离最小且 ≤ peak_dist 的一对，每点最多参与一次。"""
    if peak_dist <= 0.0 or n_det == 0 or n_strip == 0:
        return []
    thr2 = float(peak_dist) ** 2
    used_d: set[int] = set()
    used_s: set[int] = set()
    pairs: list[tuple[int, int]] = []
    while True:
        best_d2 = thr2 * 1.0000001
        best_ij: tuple[int, int] | None = None
        for di in range(n_det):
            if di in used_d:
                continue
            xd, yd = float(det_pts[di]["x0"]), float(det_pts[di]["y0"])
            for sj in range(n_strip):
                if sj in used_s:
                    continue
                xs, ys = float(strip_pts[sj]["x0"]), float(strip_pts[sj]["y0"])
                d2 = (xd - xs) ** 2 + (yd - ys) ** 2
                if d2 <= thr2 and d2 < best_d2:
                    best_d2 = d2
                    best_ij = (di, sj)
        if best_ij is None:
            break
        di, sj = best_ij
        used_d.add(di)
        used_s.add(sj)
        pairs.append((di, sj))
    return pairs


def _split_entry_from_fit(
    fit_rec: dict[str, Any],
    det_ion: dict[str, Any],
    strip_rec: dict[str, Any],
    py_d: int,
    px_d: int,
) -> dict[str, Any]:
    """双峰保留时：按拟合条目的整数种子归属 detect 或 strip 模板。"""
    if int(fit_rec["_py"]) == py_d and int(fit_rec["_px"]) == px_d:
        out = dict(det_ion)
    else:
        out = {"source": strip_rec["source"]}
    out["x0"] = float(fit_rec["x0"])
    out["y0"] = float(fit_rec["y0"])
    out["sigma_minor"] = float(fit_rec["sigma_minor"])
    out["sigma_major"] = float(fit_rec["sigma_major"])
    out["theta_deg"] = float(fit_rec["theta_deg"])
    out["amplitude"] = float(fit_rec["amplitude"])
    out["source"] = "fused_split"
    if "_sigma_x" in fit_rec:
        out["_sigma_x"] = fit_rec["_sigma_x"]
        out["_sigma_y"] = fit_rec["_sigma_y"]
    return out


def _inside_ellipse(
    x: float, y: float, boundary: tuple[float, float, float, float],
) -> bool:
    cx, cy, a, b = boundary
    if a <= 0 or b <= 0:
        return False
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0


def _zone_top_bottom_strip(
    y: float,
    meta: dict,
) -> str:
    """'top' | 'bottom' | 'middle' 按外缘条带的 y 区间划分（与 strip 掩膜一致）。"""
    y_top_v = float(meta["y_top_vertex"])
    y_below = float(meta["y_below"])
    y_above = float(meta["y_above"])
    y_bot_v = float(meta["y_bot_vertex"])
    if y_top_v <= y <= y_below:
        return "top"
    if y_above <= y <= y_bot_v:
        return "bottom"
    return "middle"


def _in_strip_priority_slab(
    x: float,
    y: float,
    boundary: tuple[float, float, float, float],
    meta: dict,
    edge_x_lo: float,
    edge_x_hi: float,
) -> bool:
    """是否在「条带优先」上下域内（椭圆 ∩ x 条带 ∩ 上/下外缘 y 带）。"""
    if edge_x_lo > edge_x_hi:
        edge_x_lo, edge_x_hi = edge_x_hi, edge_x_lo
    if not (edge_x_lo <= x <= edge_x_hi):
        return False
    if not _inside_ellipse(x, y, boundary):
        return False
    z = _zone_top_bottom_strip(y, meta)
    return z in ("top", "bottom")


def _bgsub_array(arr: np.ndarray, bg_sigma=(10, 30)) -> np.ndarray:
    u = np.asarray(arr, dtype=np.float64)
    bg = gaussian_filter(u, sigma=bg_sigma)
    return u - bg


def _strip_input_map(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    preprocess: str,
    ions_for_peel: list | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """返回送入条带聚合的二维图；peel 模式需要 ions_for_peel。"""
    if preprocess == "raw":
        return image.astype(np.float64), None
    if preprocess == "bgsub":
        return _bgsub_array(image), None
    if preprocess == "peel":
        from ion_detect.gaussian import _accumulate_peel_model

        if not ions_for_peel:
            raise ValueError("preprocess=peel 需要首轮 ions 以构造残差")
        h, w = image.shape
        peel = _accumulate_peel_model(h, w, ions_for_peel, margin_sigma=4.5)
        return image.astype(np.float64) - peel, image.astype(np.float64) - peel
    if preprocess == "peel_bgsub":
        from ion_detect.gaussian import _accumulate_peel_model

        if not ions_for_peel:
            raise ValueError("preprocess=peel_bgsub 需要首轮 ions")
        h, w = image.shape
        peel = _accumulate_peel_model(h, w, ions_for_peel, margin_sigma=4.5)
        resid = image.astype(np.float64) - peel
        return _bgsub_array(resid), resid
    raise ValueError(f"unknown preprocess: {preprocess}")


def merge_centers_hybrid(
    ions: list[dict[str, Any]],
    boundary: tuple[float, float, float, float],
    strip_map: np.ndarray,
    *,
    strip_result: dict,
    edge_x_lo: float,
    edge_x_hi: float,
    peak_dist: float,
    ion_dist: float,
    clip_ellipse: bool,
    y_fit_frac: float | None,
    add_neighbor_x: bool,
    strip_center_mode: str = "com",
    fit_hw: tuple[int, int] | None = None,
    sigma_fit_range: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回合并后的点列表（每项含 x0,y0,source）及统计信息 dict。"""
    meta = strip_result["meta"]
    top_xy, bot_xy = fitted_xy_for_auxiliary_strip_peaks(
        strip_map,
        strip_result,
        boundary,
        peak_dist=peak_dist,
        clip_ellipse=clip_ellipse,
        y_fit_frac=y_fit_frac,
        add_neighbor_x=add_neighbor_x,
        double_peak_fit=False,
        prominence_min=None,
        center_mode=strip_center_mode,
    )

    ex0, ex1 = (edge_x_lo, edge_x_hi) if edge_x_lo <= edge_x_hi else (edge_x_hi, edge_x_lo)

    strip_points: list[dict[str, Any]] = []
    for x, y in top_xy:
        if ex0 <= x <= ex1 and _inside_ellipse(x, y, boundary):
            strip_points.append({"x0": x, "y0": y, "source": "strip_top"})
    for x, y in bot_xy:
        if ex0 <= x <= ex1 and _inside_ellipse(x, y, boundary):
            strip_points.append({"x0": x, "y0": y, "source": "strip_bot"})

    detect_in_slab: list[dict[str, Any]] = []
    detect_elsewhere: list[dict[str, Any]] = []
    for ion in ions:
        x0 = float(ion["x0"])
        y0 = float(ion["y0"])
        d = dict(ion)
        d["source"] = "detect"
        if _in_strip_priority_slab(x0, y0, boundary, meta, ex0, ex1):
            detect_in_slab.append(d)
        else:
            detect_elsewhere.append(d)

    hw = fit_hw if fit_hw is not None else _DEFAULT_FIT_HW
    hw_y, hw_x = int(hw[0]), int(hw[1])
    s_lo, s_hi = (
        sigma_fit_range
        if sigma_fit_range is not None
        else _DEFAULT_SIGMA_FIT_RANGE
    )
    fit_img = np.asarray(strip_map, dtype=np.float64)
    h, w = fit_img.shape
    ion_thr = float(ion_dist)

    pairs = _greedy_pair_detect_strip(
        len(detect_in_slab),
        len(strip_points),
        detect_in_slab,
        strip_points,
        peak_dist,
    )

    used_slab: set[int] = set()
    used_strip: set[int] = set()
    merged_from_pairs: list[dict[str, Any]] = []
    n_fused_bimodal = 0
    n_fused_split = 0
    n_pair_fit_failed = 0

    for di, sj in pairs:
        used_slab.add(di)
        used_strip.add(sj)
        det_d = detect_in_slab[di]
        stp = strip_points[sj]
        py_d = int(np.clip(round(float(det_d["y0"])), 0, h - 1))
        px_d = int(np.clip(round(float(det_d["x0"])), 0, w - 1))
        py_s = int(np.clip(round(float(stp["y0"])), 0, h - 1))
        px_s = int(np.clip(round(float(stp["x0"])), 0, w - 1))

        twop = fit_joint_two_peaks_at(
            fit_img,
            py_d,
            px_d,
            py_s,
            px_s,
            hw_y,
            hw_x,
            float(s_lo),
            float(s_hi),
            h,
            w,
            _BIMODAL_SIGMA_INIT,
        )

        if twop is None or len(twop) != 2:
            n_pair_fit_failed += 1
            fb = dict(det_d)
            fb["x0"] = 0.5 * (float(det_d["x0"]) + float(stp["x0"]))
            fb["y0"] = 0.5 * (float(det_d["y0"]) + float(stp["y0"]))
            fb["source"] = "fused_bimodal_fallback"
            merged_from_pairs.append(fb)
            n_fused_bimodal += 1
            continue

        x_a, y_a = float(twop[0]["x0"]), float(twop[0]["y0"])
        x_b, y_b = float(twop[1]["x0"]), float(twop[1]["y0"])
        sep = float(np.hypot(x_a - x_b, y_a - y_b))

        if ion_thr > 0.0 and sep < ion_thr:
            out = dict(det_d)
            out["x0"] = 0.5 * (x_a + x_b)
            out["y0"] = 0.5 * (y_a + y_b)
            out["source"] = "fused_bimodal"
            merged_from_pairs.append(out)
            n_fused_bimodal += 1
        else:
            for fk in twop:
                merged_from_pairs.append(
                    _split_entry_from_fit(fk, det_d, stp, py_d, px_d)
                )
            n_fused_split += 2

    detect_dropped = 0
    for i in range(len(detect_in_slab)):
        if i not in used_slab:
            detect_dropped += 1

    strip_kept: list[dict[str, Any]] = [
        dict(strip_points[j])
        for j in range(len(strip_points))
        if j not in used_strip
    ]

    merged = detect_elsewhere + strip_kept + merged_from_pairs
    merged.sort(key=lambda d: (d["y0"], d["x0"]))
    stats = {
        "n_detect_raw": len(ions),
        "n_detect_kept": len(detect_elsewhere),
        "n_detect_dropped_strip_zone": detect_dropped,
        "n_detect_slab_paired": len(used_slab),
        "n_strip_top_raw": len(top_xy),
        "n_strip_bot_raw": len(bot_xy),
        "n_strip_used": len(strip_points),
        "edge_x": (ex0, ex1),
        "n_pair_detect_strip": len(pairs),
        "n_fused_bimodal": n_fused_bimodal,
        "n_fused_split_peaks": n_fused_split,
        "n_pair_fit_failed": n_pair_fit_failed,
    }
    return merged, stats


def _plot_merged(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    strip_result: dict,
    merged: list[dict[str, Any]],
    edge_x_lo: float,
    edge_x_hi: float,
    out_path: Path,
    title: str,
    *,
    show: bool,
) -> None:
    h, w = image.shape
    meta = strip_result["meta"]
    cx, cy, a, b = boundary
    y_below = float(meta["y_below"])
    y_above = float(meta["y_above"])
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    lo = float(np.percentile(image, 1))
    hi = float(np.percentile(image, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    ax.imshow(image, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    ell = Ellipse(
        xy=(cx, cy), width=2 * a, height=2 * b, angle=0,
        edgecolor="cyan", facecolor="none", linewidth=1.2, linestyle="--", alpha=0.9,
    )
    ax.add_patch(ell)
    ax.axhline(y_below, color="darkmagenta", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.axhline(y_above, color="darkmagenta", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.axvline(edge_x_lo, color="gold", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axvline(edge_x_hi, color="gold", linestyle="--", linewidth=1.0, alpha=0.9)
    colors = {
        "detect": "dodgerblue",
        "strip_top": "tomato",
        "strip_bot": "lime",
        "fused_mean": "mediumorchid",
        "fused_bimodal": "mediumorchid",
        "fused_bimodal_fallback": "violet",
        "fused_split": "gold",
    }
    for ion in merged:
        src = ion.get("source", "detect")
        c = colors.get(src, "white")
        ax.scatter(
            [ion["x0"]], [ion["y0"]], s=28, marker="o", c=c, edgecolors="k",
            linewidths=0.35, zorder=6,
        )
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["detect"],
               markersize=8, label="detect (body)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["strip_top"],
               markersize=8, label="strip top (COM)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["strip_bot"],
               markersize=8, label="strip bottom (COM)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["fused_bimodal"],
               markersize=8, label="fused bimodal (1 peak)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["fused_split"],
               markersize=8, label="fused split (2 peaks)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["fused_bimodal_fallback"],
               markersize=8, label="bimodal fit fallback", markeredgecolor="k"),
        Line2D([0], [0], color="cyan", linestyle="--", label="boundary ellipse"),
        Line2D([0], [0], color="darkmagenta", linestyle=":", label="y strip / middle split"),
        Line2D([0], [0], color="gold", linestyle="--", label="edge x slab"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ion_detect + 外缘条带 COM 合并离子中心并出图",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引（与 ion_detect 相同）",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "20260305_1727")
    parser.add_argument("--out", type=Path, default=OUT_ION_CENTERS_MERGED)
    parser.add_argument(
        "--edge-x-range",
        type=float,
        nargs=2,
        default=[250.0, 750.0],
        metavar=("X0", "X1"),
        help="上下缘条带优先域的 x 范围（闭区间）；外侧仍只用 detect",
    )
    parser.add_argument(
        "--y-edge-frac",
        type=float,
        default=0.25,
        help="外缘条带几何参数 F（与 edge_strip --y-edge-frac 相同，划分上/中/下域）",
    )
    parser.add_argument(
        "--y-fit-frac",
        type=float,
        default=0.35,
        help="列 y 向 COM 采样条带宽度 F（与 --y-edge-frac 可不同；传给 fitted_xy 的 y_fit_frac）",
    )
    parser.add_argument(
        "--peak-dist",
        type=float,
        default=5.0,
        help="条带 1D 轮廓上辅助峰最小间距（像素）；亦用于与条带域内 detect 配对做联合双峰拟合",
    )
    parser.add_argument(
        "--col-metric",
        choices=("sum", "mean", "max"),
        default="mean",
        help="条带按列聚合度量",
    )
    parser.add_argument(
        "--strip-center-mode",
        choices=("com", "com_fit", "fit"),
        default="com",
        help="条带列 y 中心：com（与示例一致）/ com_fit / fit",
    )
    parser.add_argument(
        "--add-neighbor-x",
        action="store_true",
        help="列 profile 使用 x-1,x,x+1 三列和（与 edge_strip 一致）",
    )
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub", "peel", "peel_bgsub"),
        default="raw",
        help="送入条带聚合的二维图（detect 仍始终用原图 pipeline）",
    )
    parser.add_argument(
        "--no-clip-ellipse",
        action="store_true",
        help="条带不按椭圆裁剪（与 edge_strip --no-clip-ellipse 对应）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="交互显示窗口（默认仅保存 PNG）",
    )
    parser.add_argument(
        "--no-matched-filter",
        action="store_true",
        help="detect_ions 禁用匹配滤波（与 ion_detect --no-matched-filter 一致）",
    )
    parser.add_argument(
        "--ion-dist",
        type=float,
        default=5.0,
        metavar="PX",
        help=(
            "条带域内 detect–strip 经联合双峰拟合后，若两拟合峰间距 < 该值（像素）则并成一点（均值）；"
            "否则保留两拟合峰。≤0 时拟合成功则总保留双峰。勿大于约半格距以免误并相邻离子。"
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

    amp_coef_path = OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy"
    clip_ellipse = not args.no_clip_ellipse
    ex_lo, ex_hi = float(args.edge_x_range[0]), float(args.edge_x_range[1])

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions, boundary = detect_ions(
            image,
            fix_theta_zero=True,
            use_matched_filter=not args.no_matched_filter,
            amp_y_coef_path=amp_coef_path,
            amp_y_coef_mode="even",
        )
        if boundary is None:
            print(f"[{idx:04d}] {target.name}: 无 boundary，跳过")
            continue

        ions_peel = ions
        try:
            strip_map, _ = _strip_input_map(
                image, boundary, args.preprocess, ions_for_peel=ions_peel,
            )
        except ValueError as e:
            print(f"[{idx:04d}] {target.name}: {e}，跳过")
            continue

        strip_result = outer_y_edge_column_profiles(
            strip_map,
            boundary,
            args.y_edge_frac,
            clip_ellipse=clip_ellipse,
            col_metric=args.col_metric,
        )

        y_fit = float(args.y_fit_frac)
        merged, stats = merge_centers_hybrid(
            ions,
            boundary,
            strip_map,
            strip_result=strip_result,
            edge_x_lo=ex_lo,
            edge_x_hi=ex_hi,
            peak_dist=float(args.peak_dist),
            ion_dist=float(args.ion_dist),
            clip_ellipse=clip_ellipse,
            y_fit_frac=y_fit,
            add_neighbor_x=args.add_neighbor_x,
            strip_center_mode=args.strip_center_mode,
        )

        stem = Path(target.name).stem
        safe = stem.encode("ascii", "replace").decode("ascii")
        title = (
            f"frame {idx:04d} ({safe}) merged centers | "
            f"detect kept {stats['n_detect_kept']}/{stats['n_detect_raw']}, "
            f"strip +{stats['n_strip_used']}, pairs {stats['n_pair_detect_strip']} "
            f"(1-peak {stats['n_fused_bimodal']}, 2-peak {stats['n_fused_split_peaks']}, "
            f"fit-fail {stats['n_pair_fit_failed']}; ion-dist={float(args.ion_dist):g}, "
            f"peak-dist={float(args.peak_dist):g}, x slab {stats['edge_x'][0]:.0f}–{stats['edge_x'][1]:.0f})"
        )
        out_png = args.out / f"ion_centers_merged_{idx:04d}.png"
        print(f"\n[{idx:04d}] {target.name}")
        print(
            f"  detect: {stats['n_detect_raw']} raw, "
            f"{stats['n_detect_kept']} kept (body), "
            f"{stats['n_detect_dropped_strip_zone']} dropped (strip zone, no partner), "
            f"{stats['n_detect_slab_paired']} paired for bimodal"
        )
        print(
            f"  strip peaks: top {stats['n_strip_top_raw']}, bot {stats['n_strip_bot_raw']}; "
            f"used in merge {stats['n_strip_used']}"
        )
        print(
            f"  detect+strip pairs: {stats['n_pair_detect_strip']}; "
            f"1-peak out {stats['n_fused_bimodal']}, "
            f"2-peak out {stats['n_fused_split_peaks']}, "
            f"fit fail {stats['n_pair_fit_failed']}"
        )
        print(f"  total merged: {len(merged)}")

        _plot_merged(
            image, boundary, strip_result, merged, ex_lo, ex_hi, out_png, title,
            show=args.show,
        )
        print(f"  saved {out_png}")


if __name__ == "__main__":
    main()
