"""
根据 ``outputs/blob`` 下的 merge_split 类 TSV 日志，筛选指定 ``final_ions`` 的行，
按切片选取子集，从离子坐标目录加载对应 ``.npy``，计算两两 normalized cdistance 并绘制直方图。

两两距离会缓存到 ``outputs/histogram/{日志文件名stem}-{离子数}.npz``（含每对的
``frame_idx_a`` / ``frame_idx_b`` / ``distances``）。下次若构型序列与参数一致则直接读缓存再绘图。

运行目录建议为 ``Ion-Image-Process`` 根目录::

    python cdist/log_cdistance_hist.py
    python cdist/log_cdistance_hist.py --indices 0:50 ::3 --ion-count 1727
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.cli_helpers import resolve_indices
from output_paths import OUT_BLOB_PIPELINE, OUT_HISTOGRAM, OUT_ION_POS

# 同目录 dist.py（避免与标准库名冲突，用 importlib 加载）
_spec = importlib.util.spec_from_file_location("ionimage_cdist_dist", _SCRIPT_DIR / "dist.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法加载 cdist/dist.py")
_ion_cdist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ion_cdist)
compute_all_pair_distances_from_coords = _ion_cdist.compute_all_pair_distances_from_coords
plot_histogram = _ion_cdist.plot_histogram

# 与 ``dist.normalized_config_distance`` 默认一致；写入缓存供校验
_DEFAULT_CENTER_CENTROIDS: bool = True


def load_merge_split_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.is_file():
        raise FileNotFoundError(f"日志不存在: {log_path}")
    rows: list[dict[str, str]] = []
    with log_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "frame_idx" not in reader.fieldnames:
            raise ValueError(f"无法解析表头（需含 frame_idx 等 TSV 列）: {log_path}")
        if "final_ions" not in reader.fieldnames or "file" not in reader.fieldnames:
            raise ValueError(f"日志缺少 final_ions 或 file 列: {log_path}")
        for row in reader:
            if not row.get("frame_idx", "").strip():
                continue
            rows.append(row)
    return rows


def filter_rows_by_final_ions(rows: list[dict[str, str]], ion_count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        try:
            n = int(str(r["final_ions"]).strip())
        except (KeyError, ValueError):
            continue
        if n == ion_count:
            out.append(r)
    return out


def load_coords_for_row(row: dict[str, str], pos_dir: Path, expected_ions: int) -> np.ndarray | None:
    name = str(row["file"]).strip()
    if not name:
        return None
    path = pos_dir / name
    if not path.is_file():
        print(f"[skip] 无坐标文件: {path} (frame_idx={row.get('frame_idx')})")
        return None
    coords = np.asarray(np.load(path), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        print(f"[skip] 坐标形状非 (N,2): {path} shape={coords.shape}")
        return None
    if coords.shape[0] != expected_ions:
        print(
            f"[skip] 离子数与期望不符: {path} N={coords.shape[0]} 期望 {expected_ions} "
            f"(log final_ions={row.get('final_ions')})",
        )
        return None
    return coords


def cdist_pairs_npz_path(log_path: Path, ion_count: int) -> Path:
    """两两晶格距离缓存：``{日志文件名不含扩展名}-{离子数}.npz``。"""
    return OUT_HISTOGRAM / f"{log_path.stem}-{ion_count}.npz"


def _entries_frame_order(entries: list[tuple[dict[str, str], np.ndarray]]) -> list[str]:
    return [str(r.get("frame_idx", "")).strip() for r, _ in entries]


def _unicode_np(strings: list[str], *, max_len: int = 64) -> np.ndarray:
    return np.asarray(strings, dtype=f"U{max_len}")


def try_load_pairwise_cdist_cache(
    path: Path,
    *,
    log_stem: str,
    ion_count: int,
    frame_order: list[str],
    center_centroids: bool,
) -> list[float] | None:
    """若缓存存在且与当前构型序列、离子数、度量一致，返回距离列表；否则返回 None。"""
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except OSError as e:
        print(f"[warn] 无法读取缓存 {path}: {e}，将重新计算")
        return None

    need = {
        "ion_count",
        "log_stem",
        "center_centroids",
        "frame_order",
        "frame_idx_a",
        "frame_idx_b",
        "distances",
    }
    if not need.issubset(z.files):
        print(f"[warn] 缓存字段不完整，将重新计算: {path}")
        return None

    if int(z["ion_count"]) != ion_count:
        print(f"[warn] 缓存 ion_count 与当前不一致，将重新计算: {path}")
        return None
    cached_stem = str(np.asarray(z["log_stem"]).item())
    if cached_stem != log_stem:
        print(f"[warn] 缓存 log_stem 与当前不一致，将重新计算: {path}")
        return None
    if bool(z["center_centroids"]) != center_centroids:
        print(f"[warn] 缓存 center_centroids 与当前不一致，将重新计算: {path}")
        return None

    cached_order = [str(x) for x in np.asarray(z["frame_order"]).tolist()]
    if cached_order != frame_order:
        print(f"[warn] 缓存构型序列与当前不一致（例如 --indices 不同），将重新计算: {path}")
        return None

    n = len(frame_order)
    n_pairs = n * (n - 1) // 2
    dist = np.asarray(z["distances"], dtype=np.float64)
    fa = np.asarray(z["frame_idx_a"])
    fb = np.asarray(z["frame_idx_b"])
    if dist.shape != (n_pairs,) or fa.shape != (n_pairs,) or fb.shape != (n_pairs,):
        print(f"[warn] 缓存数组形状异常，将重新计算: {path}")
        return None

    pairs = list(combinations(range(n), 2))
    for k, (i, j) in enumerate(pairs):
        if str(fa[k]) != frame_order[i] or str(fb[k]) != frame_order[j]:
            print(f"[warn] 缓存内 frame_idx 与顺序不对应，将重新计算: {path}")
            return None

    print(f"[info] 使用已缓存的两两距离: {path}")
    return [float(x) for x in dist.tolist()]


def save_pairwise_cdist_cache(
    path: Path,
    *,
    log_path: Path,
    ion_count: int,
    frame_order: list[str],
    distances: list[float],
    center_centroids: bool,
) -> None:
    n = len(frame_order)
    pairs = list(combinations(range(n), 2))
    fa = [frame_order[i] for i, j in pairs]
    fb = [frame_order[j] for i, j in pairs]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ion_count=np.int32(ion_count),
        log_stem=np.asarray(log_path.stem),
        center_centroids=np.bool_(center_centroids),
        frame_order=_unicode_np(frame_order),
        frame_idx_a=_unicode_np(fa),
        frame_idx_b=_unicode_np(fb),
        distances=np.asarray(distances, dtype=np.float64),
    )
    print(f"[info] 已写入两两距离缓存: {path}")


def default_hist_path(log_path: Path, ion_count: int, n_configs: int) -> Path:
    return (
        OUT_HISTOGRAM
        / f"cdist_hist_{log_path.stem}_finalions{ion_count}_configs{n_configs}.png"
    )


def default_min_pair_path(
    log_path: Path,
    ion_count: int,
    frame_a: str,
    frame_b: str,
) -> Path:
    return (
        OUT_HISTOGRAM
        / f"cdist_minpair_{log_path.stem}_finalions{ion_count}_f{frame_a}_f{frame_b}.png"
    )


def argmin_pair_distance(distances: list[float], n: int) -> tuple[int, int, float]:
    """返回使距离最小的构型对索引 (i, j) 及该距离值（与 ``combinations`` 顺序一致）。"""
    pairs = list(combinations(range(n), 2))
    if len(distances) != len(pairs):
        raise RuntimeError("内部错误: 距离条数与构型对数不一致")
    k = int(np.argmin(np.asarray(distances, dtype=np.float64)))
    i, j = pairs[k]
    return i, j, float(distances[k])


def plot_min_distance_pair_overlay(
    row_a: dict[str, str],
    coords_a: np.ndarray,
    row_b: dict[str, str],
    coords_b: np.ndarray,
    *,
    d_min: float,
    output_path: Path | None,
    show: bool,
) -> None:
    """Overlay two frames' ion positions (red / blue); title shows log frame_idx (English labels for matplotlib)."""
    fa = str(row_a.get("frame_idx", "")).strip()
    fb = str(row_b.get("frame_idx", "")).strip()
    fig, ax = plt.subplots(figsize=(9, 8))
    ca = np.asarray(coords_a, dtype=np.float64)
    cb = np.asarray(coords_b, dtype=np.float64)
    ax.scatter(
        ca[:, 0],
        ca[:, 1],
        c="red",
        s=10,
        alpha=0.55,
        edgecolors="none",
        label=f"Frame {fa} (red)",
    )
    ax.scatter(
        cb[:, 0],
        cb[:, 1],
        c="blue",
        s=10,
        alpha=0.55,
        edgecolors="none",
        label=f"Frame {fb} (blue)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Closest pair by config distance: frame_idx={fa} vs frame_idx={fb}\n"
        f"normalized cdistance = {d_min:.6f} px "
        f"(same metric as histogram: centroid-aligned, mean nearest-neighbor)"
    )
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"[saved] min-pair overlay: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 merge_split 日志，按 final_ions 与切片筛选帧，计算坐标两两 cdistance 并画直方图。",
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=OUT_BLOB_PIPELINE,
        help="日志所在目录（默认: outputs/blob）",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="merge_split.log",
        help="日志文件名（默认: merge_split.log）",
    )
    parser.add_argument(
        "--ion-count",
        type=int,
        default=1727,
        help="只保留日志中 final_ions 等于该值的行（默认: 1727）",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        default=None,
        metavar="SLICE",
        help="对「筛选后」行序列的索引切片，语法与 blob_cli --indices 相同（如 0:20 、 0,5,10 、 ::2）；默认全部",
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=OUT_ION_POS,
        help="离子坐标 .npy 目录，文件名需与日志 file 列一致（默认: outputs/IonPos）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="直方图输出路径（默认: outputs/histogram/cdist_hist_<logstem>_...png）",
    )
    parser.add_argument("--bins", type=int, default=50, help="直方图 bin 数")
    parser.add_argument("--show", action="store_true", help="保存后显示直方图窗口")
    parser.add_argument(
        "--min-pair-output",
        type=Path,
        default=None,
        help="最小距离两帧红蓝叠加图保存路径（默认 outputs/histogram/cdist_minpair_...）",
    )
    parser.add_argument(
        "--no-show-min-pair",
        action="store_true",
        help="不弹窗显示最小距离两帧叠加图（仍会保存 PNG）",
    )
    args = parser.parse_args()

    log_path = args.blob_dir / args.log_name
    rows = load_merge_split_rows(log_path)
    filtered = filter_rows_by_final_ions(rows, args.ion_count)
    if not filtered:
        raise SystemExit(
            f"无匹配行: final_ions={args.ion_count}（共读取日志 {len(rows)} 行），请检查 --ion-count 或日志内容",
        )

    if args.indices is None:
        selected_rows = filtered
    else:
        if not args.indices:
            raise SystemExit("--indices 需要至少一段切片（省略 --indices 表示使用全部筛选结果）")
        idxs = resolve_indices(list(args.indices), len(filtered))
        if not idxs:
            raise SystemExit("切片结果为空，请检查 --indices 与当前筛选行数")
        selected_rows = [filtered[i] for i in idxs]

    if not args.pos_dir.is_dir():
        raise FileNotFoundError(f"坐标目录不存在: {args.pos_dir}")

    entries: list[tuple[dict[str, str], np.ndarray]] = []
    for row in selected_rows:
        c = load_coords_for_row(row, args.pos_dir, args.ion_count)
        if c is not None:
            entries.append((row, c))

    if len(entries) < 2:
        raise SystemExit(
            f"有效坐标构型不足 2 个（当前 {len(entries)}），请检查 --pos-dir 与日志 file 列是否一致",
        )

    coords_list = [c for _, c in entries]
    frame_order = _entries_frame_order(entries)
    cache_path = cdist_pairs_npz_path(log_path, args.ion_count)

    distances = try_load_pairwise_cdist_cache(
        cache_path,
        log_stem=log_path.stem,
        ion_count=args.ion_count,
        frame_order=frame_order,
        center_centroids=_DEFAULT_CENTER_CENTROIDS,
    )
    if distances is None:
        distances = compute_all_pair_distances_from_coords(
            coords_list,
            center_centroids=_DEFAULT_CENTER_CENTROIDS,
        )
        save_pairwise_cdist_cache(
            cache_path,
            log_path=log_path,
            ion_count=args.ion_count,
            frame_order=frame_order,
            distances=distances,
            center_centroids=_DEFAULT_CENTER_CENTROIDS,
        )

    out = args.output or default_hist_path(log_path, args.ion_count, len(coords_list))
    plot_histogram(
        distances,
        config_count=len(coords_list),
        output_path=out,
        bins=args.bins,
        show=args.show,
        centroid_aligned=_DEFAULT_CENTER_CENTROIDS,
    )

    i_min, j_min, d_min = argmin_pair_distance(distances, len(coords_list))

    row_a, coords_a = entries[i_min]
    row_b, coords_b = entries[j_min]
    fa = str(row_a.get("frame_idx", "")).strip()
    fb = str(row_b.get("frame_idx", "")).strip()
    pair_out = args.min_pair_output or default_min_pair_path(
        log_path, args.ion_count, fa, fb
    )
    plot_min_distance_pair_overlay(
        row_a,
        coords_a,
        row_b,
        coords_b,
        d_min=d_min,
        output_path=pair_out,
        show=not args.no_show_min_pair,
    )

    print(f"[info] log={log_path} final_ions={args.ion_count} configs={len(coords_list)} pairs={len(distances)}")
    print(
        f"[info] 最小构型距离: {d_min:.6f} px → frame_idx={fa} vs frame_idx={fb} "
        f"（日志行内 file: {row_a.get('file')} | {row_b.get('file')}）",
    )


if __name__ == "__main__":
    main()
