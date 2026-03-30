"""
合并 ion_detect（中心区域）与外缘条带列向质心（上下边缘、中间 x 范围）。

规则简述：
- 在椭圆内、外缘 y 条带几何（与 edge_strip 一致）且 x 落在 [edge_x_lo, edge_x_hi] 的上下两域内，仅采用条带辅助峰 + COM 的中心，丢弃落在该域内的 ion_detect 结果。
- 其余椭圆内区域仅采用 ion_detect，条带点若落入该区域一般会因几何不可能（条带点在上下条带内）而自然为空集；条带在 x 域外的峰被丢弃。
- 对 detect 与 strip 的中心，若欧氏距离 ≤ ion-dist（默认 4），替换为二者坐标算术平均（贪心每次合并当前最近的一对）；fit 参数等保留自 detect 项。
- ``--second-layer-slab`` 最后将 second_layer_L1/L2 与其馀 center 在 ≤ ion-dist 内同样做坐标平均合并（source=fused_second_layer）。

示例：
  python merge_ion_centers.py 0
  python merge_ion_centers.py 0 --edge-x-range 250 750 --y-edge-frac 0.25 \\
    --y-fit-frac 0.35 --peak-dist 5
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

from output_paths import DEFAULT_DATA_DIR, OUT_ION_CENTERS_MERGED

from ion_detect.cli_helpers import resolve_indices
from ion_detect.edge_strip import outer_y_edge_column_profiles
from ion_detect.edge_strip_profile_analysis import fitted_xy_for_auxiliary_strip_peaks
from ion_detect.pipeline import detect_ions
from second_layer_core import (
    ions_from_second_layer_row,
    replace_merge_in_xy_slab,
    second_layer_y0_pair_and_slab_hi_mid23,
)

_STRIP_SOURCES_FUSE = frozenset({"strip_top", "strip_bot"})
_SECOND_LAYER_SOURCES = frozenset({"second_layer_L1", "second_layer_L2"})
# 仅与 second_layer 配对的“另一侧”：原始 detect/strip/fused_mean，不含 fused_second_layer，避免链式重复合并
_OTHER_FOR_SECOND_LAYER_FUSE = frozenset({"detect", "strip_top", "strip_bot", "fused_mean"})


def fuse_detect_strip_by_distance(
    points: list[dict[str, Any]],
    ion_dist: float,
) -> tuple[list[dict[str, Any]], int]:
    """将 source==detect 与 strip_top/strip_bot 且距离 ≤ ion_dist 的点对合并为坐标均值。

    每次在全部合法点对中取距离最小的一对合并，直至无满足阈值的点对。
    合并后的条目 source 为 ``fused_mean``，x0/y0 为二者平均，其余键自 detect 拷贝。
    """
    if ion_dist <= 0.0 or len(points) < 2:
        return points, 0
    thr2 = float(ion_dist) ** 2
    work = list(points)
    n_fused = 0
    while True:
        best_i: int | None = None
        best_j: int | None = None
        best_d2 = thr2 * 1.0000001  # 仅接受 d2 <= thr2
        for i in range(len(work)):
            if work[i].get("source") != "detect":
                continue
            xi, yi = float(work[i]["x0"]), float(work[i]["y0"])
            for j in range(len(work)):
                if i == j:
                    continue
                if work[j].get("source") not in _STRIP_SOURCES_FUSE:
                    continue
                xj, yj = float(work[j]["x0"]), float(work[j]["y0"])
                d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                if d2 <= thr2 and d2 < best_d2:
                    best_d2 = d2
                    best_i, best_j = i, j
        if best_i is None or best_j is None or best_j == best_i:
            break
        d_det = work[best_i]
        d_strip = work[best_j]
        fused = dict(d_det)
        fused["x0"] = 0.5 * (float(d_det["x0"]) + float(d_strip["x0"]))
        fused["y0"] = 0.5 * (float(d_det["y0"]) + float(d_strip["y0"]))
        fused["source"] = "fused_mean"
        for k in (max(best_i, best_j), min(best_i, best_j)):
            del work[k]
        work.append(fused)
        n_fused += 1
    work.sort(key=lambda d: (d["y0"], d["x0"]))
    return work, n_fused


def fuse_second_layer_with_others_by_distance(
    points: list[dict[str, Any]],
    ion_dist: float,
) -> tuple[list[dict[str, Any]], int]:
    """second_layer_L1/L2 与 detect/strip_top/strip_bot/fused_mean 若距离 ≤ ion_dist，则并成坐标平均。

    每次取距离最小且满足阈值的一对；模板取自非 second_layer 一侧，x0/y0 取均值，source 为
    ``fused_second_layer``。不与其他 fused_second_layer 配对，避免链式平均。
    """
    if ion_dist <= 0.0 or len(points) < 2:
        return points, 0
    thr2 = float(ion_dist) ** 2
    work = list(points)
    n_fused = 0
    while True:
        best_i: int | None = None
        best_j: int | None = None
        best_d2 = thr2 * 1.0000001
        for i in range(len(work)):
            if work[i].get("source") not in _SECOND_LAYER_SOURCES:
                continue
            xi, yi = float(work[i]["x0"]), float(work[i]["y0"])
            for j in range(len(work)):
                if i == j:
                    continue
                if work[j].get("source") not in _OTHER_FOR_SECOND_LAYER_FUSE:
                    continue
                xj, yj = float(work[j]["x0"]), float(work[j]["y0"])
                d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                if d2 <= thr2 and d2 < best_d2:
                    best_d2 = d2
                    best_i, best_j = i, j
        if best_i is None or best_j is None or best_j == best_i:
            break
        p_sl = work[best_i]
        p_ot = work[best_j]
        fused = dict(p_ot)
        fused["x0"] = 0.5 * (float(p_sl["x0"]) + float(p_ot["x0"]))
        fused["y0"] = 0.5 * (float(p_sl["y0"]) + float(p_ot["y0"]))
        fused["source"] = "fused_second_layer"
        for k in (max(best_i, best_j), min(best_i, best_j)):
            del work[k]
        work.append(fused)
        n_fused += 1
    work.sort(key=lambda d: (float(d["y0"]), float(d["x0"])))
    return work, n_fused


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
    clip_ellipse: bool,
    y_fit_frac: float | None,
    strip_center_mode: str = "com",
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
        add_neighbor_x=True,
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

    detect_kept: list[dict[str, Any]] = []
    detect_dropped = 0
    for ion in ions:
        x0 = float(ion["x0"])
        y0 = float(ion["y0"])
        if _in_strip_priority_slab(x0, y0, boundary, meta, ex0, ex1):
            detect_dropped += 1
            continue
        d = dict(ion)
        d["source"] = "detect"
        detect_kept.append(d)

    merged = detect_kept + strip_points
    merged.sort(key=lambda d: (d["y0"], d["x0"]))
    stats = {
        "n_detect_raw": len(ions),
        "n_detect_kept": len(detect_kept),
        "n_detect_dropped_strip_zone": detect_dropped,
        "n_strip_top_raw": len(top_xy),
        "n_strip_bot_raw": len(bot_xy),
        "n_strip_used": len(strip_points),
        "edge_x": (ex0, ex1),
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
        "second_layer_L1": "orange",
        "second_layer_L2": "deeppink",
        "fused_second_layer": "gold",
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

    sources_present = {ion.get("source", "detect") for ion in merged}
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["detect"],
               markersize=8, label="detect (body)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["strip_top"],
               markersize=8, label="strip top (COM)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["strip_bot"],
               markersize=8, label="strip bottom (COM)", markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["fused_mean"],
               markersize=8, label="fused mean (detect+strip)", markeredgecolor="k"),
    ]
    if "second_layer_L1" in sources_present:
        legend_elems.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["second_layer_L1"],
                   markersize=8, label="second layer (line-id 1)", markeredgecolor="k"),
        )
    if "second_layer_L2" in sources_present:
        legend_elems.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["second_layer_L2"],
                   markersize=8, label="second layer (line-id 2)", markeredgecolor="k"),
        )
    if "fused_second_layer" in sources_present:
        legend_elems.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["fused_second_layer"],
                   markersize=8, label="fused mean (second_layer + other)", markeredgecolor="k"),
        )
    legend_elems.extend(
        [
            Line2D([0], [0], color="cyan", linestyle="--", label="boundary ellipse"),
            Line2D([0], [0], color="darkmagenta", linestyle=":", label="y strip / middle split"),
            Line2D([0], [0], color="gold", linestyle="--", label="edge x slab"),
        ]
    )
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
        help="条带 1D 轮廓上辅助峰最小间距（像素）",
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
        "--ion-dist",
        type=float,
        default=4.0,
        metavar="PX",
        help=(
            "detect 与 strip_top/strip_bot 中心距离 ≤ 该值（像素）时合并为坐标均值；"
            "≤0 关闭。注意不要大于约半格距，以免误并相邻离子。"
        ),
    )
    parser.add_argument(
        "--second-layer-slab",
        action="store_true",
        help=(
            "在 profile-x-range 与 y≤(第二/第三 y 直方图峰位中心的中点−margin) 的条带内丢弃 merge，"
            "改用 second_layer（line-first / line-second 两行 COM）；第三峰及更下 y 的 merge 保留"
        ),
    )
    parser.add_argument(
        "--profile-x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("X0", "X1"),
        help=(
            "second_layer x 剖面/替换条带范围（像素）；与 second_layer_ion_peaks 默认一致；"
            "未指定时为 300–600（不再沿用 --edge-x-range 的 250–750）"
        ),
    )
    parser.add_argument(
        "--second-layer-hist-prominence",
        type=float,
        default=5.0,
        help="合并中心 y 直方图 find_peaks prominence（独立脚本中为 --hist-prominence，默认不同）",
    )
    parser.add_argument(
        "--second-layer-prof-prominence-frac",
        type=float,
        default=0.08,
        help="x 剖面相对峰值 prominence 比例",
    )
    parser.add_argument(
        "--second-layer-prof-peak-distance",
        type=int,
        default=4,
        help="x 向峰最小间距（像素）",
    )
    parser.add_argument(
        "--second-layer-y-halfwin",
        type=int,
        default=3,
        help="列方向 COM 半窗（像素）",
    )
    parser.add_argument(
        "--second-layer-com-neighbor-cols",
        type=int,
        default=1,
        metavar="N",
        help="第二层 y 质心在峰值列两侧各并入 N 列（0=仅峰值列；默认 1 与 second_layer_ion_peaks 一致）",
    )
    parser.add_argument(
        "--second-layer-line-first",
        type=int,
        default=1,
        metavar="N",
        help="y 直方图峰序号（1 起）：第一行（较小 y，靠上）",
    )
    parser.add_argument(
        "--second-layer-line-second",
        type=int,
        default=2,
        metavar="N",
        help="y 直方图峰序号（1 起）：第二行（剖面 y0）",
    )
    parser.add_argument(
        "--second-layer-line-third",
        type=int,
        default=3,
        metavar="N",
        help="y 直方图峰序号（1 起）：第三行；与第二行峰位中心的中点作为替换条带下界，第三行 merge 保留在该界之下",
    )
    parser.add_argument(
        "--second-layer-y-cut-pad",
        type=float,
        default=1.0,
        help=(
            "替换条带 inclusive 上界 = (第二峰与第三峰 bin 中心 y 的中点) − 该值（像素）；"
            "略大则更保守、更易完整保留第三峰 merge"
        ),
    )
    args = parser.parse_args()
    OUT_ION_CENTERS_MERGED.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 下没有 npy 文件")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有可处理的索引")

    clip_ellipse = not args.no_clip_ellipse
    ex_lo, ex_hi = float(args.edge_x_range[0]), float(args.edge_x_range[1])
    if args.profile_x_range is None:
        # 与 second_layer_ion_peaks.py 默认 --edge-x-range 一致（第一/第二行剖面用 300–600）
        px_lo, px_hi = 300.0, 600.0
    else:
        px_lo, px_hi = float(args.profile_x_range[0]), float(args.profile_x_range[1])
    if px_lo > px_hi:
        px_lo, px_hi = px_hi, px_lo

    def _process_one_frame(idx: int, target: Path) -> tuple[np.ndarray, Any, dict, list[dict[str, Any]], int, dict] | None:
        image = np.load(target)
        ions, boundary = detect_ions(image)
        if boundary is None:
            print(f"[{idx:04d}] {target.name}: 无 boundary，跳过")
            return None

        ions_peel = ions
        try:
            strip_map, _ = _strip_input_map(
                image, boundary, args.preprocess, ions_for_peel=ions_peel,
            )
        except ValueError as e:
            print(f"[{idx:04d}] {target.name}: {e}，跳过")
            return None

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
            clip_ellipse=clip_ellipse,
            y_fit_frac=y_fit,
            strip_center_mode=args.strip_center_mode,
        )
        merged, n_fused = fuse_detect_strip_by_distance(merged, float(args.ion_dist))
        stats["n_fused_detect_strip"] = n_fused
        return image, boundary, strip_result, merged, n_fused, stats

    if args.second_layer_slab:
        lid1 = int(args.second_layer_line_first)
        lid2 = int(args.second_layer_line_second)
        lid3 = int(args.second_layer_line_third)
        if lid1 < 1 or lid2 < 1 or lid3 < 1:
            raise SystemExit("--second-layer-line-first/second/third 须 >= 1")
        if len({lid1, lid2, lid3}) < 3:
            raise SystemExit("second-layer 三个 line-id 须互不相同")

        merged_by_idx: dict[int, list[dict[str, Any]]] = {}
        meta_by_idx: dict[int, tuple[Any, dict, int, dict]] = {}
        ys_all: list[float] = []

        for idx in selected:
            target = files[idx]
            got = _process_one_frame(idx, target)
            if got is None:
                continue
            image, boundary, strip_result, merged, n_fused, stats = got
            merged_by_idx[idx] = merged
            meta_by_idx[idx] = (boundary, strip_result, n_fused, stats)
            ys_all.extend(float(p["y0"]) for p in merged)

        if not ys_all:
            raise SystemExit("second-layer-slab: 无合并中心，无法建 y 直方图")

        y_arr = np.asarray(ys_all, dtype=np.float64)
        y_lo = int(np.floor(y_arr.min()))
        y_hi = int(np.ceil(y_arr.max()))
        bin_edges = np.arange(y_lo, y_hi + 2, dtype=float)
        hp = float(args.second_layer_hist_prominence)
        mid_margin = float(args.second_layer_y_cut_pad)
        try:
            y0_1, y0_2, yc2, yc3, y_mid, y_replace_hi = second_layer_y0_pair_and_slab_hi_mid23(
                y_arr, bin_edges, hp, lid1, lid2, lid3, mid_margin,
            )
        except RuntimeError as e:
            raise SystemExit(f"second-layer-slab: {e}") from e

        print(
            f"\nsecond-layer-slab: line-id {lid1}→y0={y0_1}, {lid2}→y0={y0_2} | "
            f"peak centers y≈{yc2:.2f} (L{lid2}), {yc3:.2f} (L{lid3}) → mid={y_mid:.2f}, "
            f"replace slab y≤{y_replace_hi:.2f} (mid−{mid_margin:g}) | "
            f"x [{px_lo:.0f},{px_hi:.0f}] px",
        )

        halfwin = int(args.second_layer_y_halfwin)
        com_n = max(0, int(args.second_layer_com_neighbor_cols))
        ppf = float(args.second_layer_prof_prominence_frac)
        ppd = int(args.second_layer_prof_peak_distance)

        for idx in selected:
            if idx not in merged_by_idx:
                continue
            target = files[idx]
            merged = list(merged_by_idx[idx])
            boundary, strip_result, n_fused, stats = meta_by_idx[idx]
            image = np.load(target)
            im = np.asarray(image, dtype=np.float64)

            ions_l1 = ions_from_second_layer_row(
                im, y0_1, px_lo, px_hi,
                halfwin=halfwin,
                prof_prominence_frac=ppf,
                prof_peak_distance=ppd,
                source="second_layer_L1",
                com_neighbor_cols=com_n,
            )
            ions_l2 = ions_from_second_layer_row(
                im, y0_2, px_lo, px_hi,
                halfwin=halfwin,
                prof_prominence_frac=ppf,
                prof_peak_distance=ppd,
                source="second_layer_L2",
                com_neighbor_cols=com_n,
            )
            n_before = len(merged)
            merged = replace_merge_in_xy_slab(
                merged, px_lo, px_hi, 0.0, y_replace_hi, ions_l1 + ions_l2,
            )
            n_removed_slab = n_before - (len(merged) - len(ions_l1) - len(ions_l2))

            ion_thr = float(args.ion_dist)
            if ion_thr > 0.0:
                merged, n_sl_fuse = fuse_second_layer_with_others_by_distance(merged, ion_thr)
            else:
                n_sl_fuse = 0

            stem = Path(target.name).stem
            safe = stem.encode("ascii", "replace").decode("ascii")
            title = (
                f"frame {idx:04d} ({safe}) | second-layer slab replace | "
                f"L1 n={len(ions_l1)}, L2 n={len(ions_l2)}, y≤{y_replace_hi:.1f} "
                f"(mid L{lid2}/L{lid3}={y_mid:.1f}) | "
                f"second_layer↔other fused {n_sl_fuse} (ion-dist≤{ion_thr:g}) | "
                f"detect kept {stats['n_detect_kept']}/{stats['n_detect_raw']}, "
                f"strip +{stats['n_strip_used']}, fused {n_fused} | "
                f"total points {len(merged)}"
            )
            out_png = OUT_ION_CENTERS_MERGED / f"ion_centers_merged_{idx:04d}.png"
            print(f"\n[{idx:04d}] {target.name}")
            print(
                f"  detect: {stats['n_detect_raw']} raw, "
                f"{stats['n_detect_kept']} kept, "
                f"{stats['n_detect_dropped_strip_zone']} dropped in strip-priority zones",
            )
            print(
                f"  strip peaks: top {stats['n_strip_top_raw']}, bot {stats['n_strip_bot_raw']}; "
                f"used in merge {stats['n_strip_used']}",
            )
            print(f"  detect+strip fused (mean within ion-dist): {n_fused}")
            print(
                f"  second-layer: removed {n_removed_slab} merged in slab "
                f"[x={px_lo:.0f}–{px_hi:.0f}, y=0–{y_replace_hi:.1f}], "
                f"added L1={len(ions_l1)} L2={len(ions_l2)}",
            )
            if ion_thr > 0.0:
                print(f"  second_layer L1/L2 ↔ other fused (≤ion-dist): {n_sl_fuse}")
            print(f"  total plotted: {len(merged)}")

            _plot_merged(
                image, boundary, strip_result, merged, ex_lo, ex_hi, out_png, title,
                show=args.show,
            )
            print(f"  saved {out_png}")
    else:
        for idx in selected:
            target = files[idx]
            got = _process_one_frame(idx, target)
            if got is None:
                continue
            image, boundary, strip_result, merged, n_fused, stats = got

            stem = Path(target.name).stem
            safe = stem.encode("ascii", "replace").decode("ascii")
            title = (
                f"frame {idx:04d} ({safe}) merged centers | "
                f"detect kept {stats['n_detect_kept']}/{stats['n_detect_raw']}, "
                f"strip +{stats['n_strip_used']}, fused {n_fused} "
                f"(ion-dist={float(args.ion_dist):g}, x slab {stats['edge_x'][0]:.0f}–{stats['edge_x'][1]:.0f})"
            )
            out_png = OUT_ION_CENTERS_MERGED / f"ion_centers_merged_{idx:04d}.png"
            print(f"\n[{idx:04d}] {target.name}")
            print(
                f"  detect: {stats['n_detect_raw']} raw, "
                f"{stats['n_detect_kept']} kept, "
                f"{stats['n_detect_dropped_strip_zone']} dropped in strip-priority zones",
            )
            print(
                f"  strip peaks: top {stats['n_strip_top_raw']}, bot {stats['n_strip_bot_raw']}; "
                f"used in merge {stats['n_strip_used']}",
            )
            print(f"  detect+strip fused (mean within ion-dist): {n_fused}")
            print(f"  total merged: {len(merged)}")

            _plot_merged(
                image, boundary, strip_result, merged, ex_lo, ex_hi, out_png, title,
                show=args.show,
            )
            print(f"  saved {out_png}")


if __name__ == "__main__":
    main()
