"""
从 blob merge_split 类 TSV 日志中筛选指定离子数的行；``file`` 列为**原始帧** ``.npy`` 文件名。
按日志中的阈值与 blob 参数调用与 ``python -m ion_detect.blob_cli`` 相同的识别流程，校验离子数后写入
``outputs/IonPos``，供 ``cdist/dist.py`` 使用。

建议在 Ion-Image-Process 根目录执行::

    python tools/extract_ionpos_from_blob_log.py --log-name merge_split_20260414-1727-1.log --ion-count 1727
    python tools/extract_ionpos_from_blob_log.py --coords-stage equilibrium
        # 若日志列为 y-split/refine 后、ion_dist 合并前的离子数，用 equilibrium 校验并保存该阶段坐标

    python tools/extract_ionpos_from_blob_log.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.blob_ion_positions import (  # noqa: E402
    ion_equilibrium_positions_xy,
    merge_close_ion_positions_xy,
)
from ion_detect.blob_workflow import run_blob_workflow  # noqa: E402
from output_paths import DEFAULT_DATA_DIR, OUT_BLOB_PIPELINE, OUT_ION_POS  # noqa: E402


@dataclass(frozen=True)
class BlobReplayParams:
    threshold: float
    use_bgsub: bool
    use_matched_filter: bool
    connectivity: Literal[4, 8]
    min_area_pixels: int
    merge_small_rects: bool
    y_edge_frac: float
    min_edge_ysize: float
    merge_band_clip_ellipse: bool
    pre_merge_drop: bool
    split: bool
    max_ysize: float
    refine_x: bool
    x_profile_threshold: float
    x_profile_rel_to_max: bool
    ion_dist: float
    thr_norm: Literal["none", "p95", "p95_all"]
    thr_norm_percentile: float


def _parse_int(s: str | None, default: int) -> int:
    if s is None or not str(s).strip():
        return default
    return int(float(str(s).strip()))


def _parse_float(s: str | None, default: float) -> float:
    if s is None or not str(s).strip():
        return default
    return float(str(s).strip())


def _parse_bool01(s: str | None, default: bool) -> bool:
    if s is None or not str(s).strip():
        return default
    return bool(int(float(str(s).strip())))


def _parse_thr_norm(s: str | None) -> Literal["none", "p95", "p95_all"]:
    if s is None or not str(s).strip():
        return "p95"
    v = str(s).strip().lower()
    if v in ("none", "p95", "p95_all"):
        return v  # type: ignore[return-value]
    return "p95"


def detect_count_column(fieldnames: list[str] | None, override: str | None) -> str:
    if override:
        if override not in (fieldnames or []):
            raise ValueError(f"参数 --count-column {override!r} 不在日志表头中")
        return override
    if not fieldnames:
        raise ValueError("日志无表头")
    for key in ("final_ions", "after_equilibrium", "after_y_split"):
        if key in fieldnames:
            return key
    raise ValueError(
        "无法自动识别离子数列：请使用 --count-column 指定（例如 final_ions 或 after_y_split）",
    )


def load_log_rows(log_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not log_path.is_file():
        raise FileNotFoundError(f"日志不存在: {log_path}")
    rows: list[dict[str, str]] = []
    with log_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "frame_idx" not in reader.fieldnames:
            raise ValueError(f"无法解析表头（需含 frame_idx）: {log_path}")
        if "file" not in reader.fieldnames:
            raise ValueError(f"日志缺少 file 列: {log_path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if not row.get("frame_idx", "").strip():
                continue
            rows.append(row)
    return rows, fieldnames


def params_from_row(row: dict[str, str]) -> BlobReplayParams:
    th = row.get("threshold")
    if th is None or not str(th).strip():
        raise ValueError("日志行缺少 threshold")
    threshold = float(str(th).strip())

    return BlobReplayParams(
        threshold=threshold,
        use_bgsub=_parse_bool01(row.get("bgsub"), True),
        use_matched_filter=_parse_bool01(row.get("matched_filter"), False),
        connectivity=4 if _parse_int(row.get("connectivity"), 4) == 4 else 8,
        min_area_pixels=max(1, _parse_int(row.get("min_area_pixels"), 1)),
        merge_small_rects=not _parse_bool01(row.get("no_merge_small_rects"), False),
        y_edge_frac=_parse_float(row.get("y_edge_frac"), 0.35),
        min_edge_ysize=_parse_float(row.get("min_edge_ysize"), 5.0),
        merge_band_clip_ellipse=not _parse_bool01(row.get("no_merge_band_clip_ellipse"), False),
        pre_merge_drop=not _parse_bool01(row.get("no_pre_merge_drop"), False),
        split=_parse_bool01(row.get("split_on"), True),
        max_ysize=_parse_float(row.get("max_ysize"), 9.0),
        refine_x=_parse_bool01(row.get("refine_x"), True),
        x_profile_threshold=_parse_float(row.get("x_profile_threshold"), 0.4),
        x_profile_rel_to_max=_parse_bool01(row.get("x_profile_rel_to_max"), False),
        ion_dist=_parse_float(row.get("ion_dist"), 5.0),
        thr_norm=_parse_thr_norm(row.get("thr_norm")),
        thr_norm_percentile=_parse_float(row.get("thr_norm_pct"), 1.0),
    )


def positions_to_array(pts: list[tuple[float, float]]) -> np.ndarray:
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def run_blob_detection(
    image: np.ndarray,
    p: BlobReplayParams,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (equilibrium_xy, final_xy)，均为 (N,2) float64。"""
    conn: Literal[4, 8] = 4 if p.connectivity == 4 else 8
    res = run_blob_workflow(
        image,
        p.threshold,
        use_bgsub=p.use_bgsub,
        use_matched_filter=p.use_matched_filter,
        connectivity=conn,
        min_area_pixels=int(p.min_area_pixels),
        merge_small_rects=p.merge_small_rects,
        y_edge_frac=float(p.y_edge_frac),
        min_edge_ysize=float(p.min_edge_ysize),
        merge_band_clip_ellipse=p.merge_band_clip_ellipse,
        pre_merge_drop_max_span=None if not p.pre_merge_drop else 1.0,
        thr_norm=p.thr_norm,
        thr_norm_percentile=float(p.thr_norm_percentile),
    )
    eq_xy = ion_equilibrium_positions_xy(
        res.rects,
        res.binary,
        labeled=res.labeled,
        split=p.split,
        max_ysize=float(p.max_ysize),
        refine_x=p.refine_x,
        x_profile_threshold=float(p.x_profile_threshold),
        x_profile_rel_to_max=p.x_profile_rel_to_max,
        intensity=res.preprocess.denoised_map,
    )
    if float(p.ion_dist) > 0.0:
        final_list, _ = merge_close_ion_positions_xy(
            eq_xy,
            float(p.ion_dist),
            intensity=res.preprocess.denoised_map,
        )
    else:
        final_list = list(eq_xy)
    return positions_to_array(eq_xy), positions_to_array(final_list)


def filter_rows_by_count(
    rows: list[dict[str, str]],
    count_col: str,
    ion_count: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        try:
            n = int(str(r[count_col]).strip())
        except (KeyError, ValueError):
            continue
        if n == ion_count:
            out.append(r)
    return out


def dedupe_rows_by_file(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """同名 file 保留最后一行（与重复追加的日志一致）。"""
    by_name: dict[str, dict[str, str]] = {}
    for r in rows:
        name = str(r["file"]).strip()
        if name:
            by_name[name] = r
    return list(by_name.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按日志筛选离子数，从原始 .npy 重跑 blob 识别，校验后写入 outputs/IonPos。",
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
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="原始帧 .npy 所在目录（默认: output_paths.DEFAULT_DATA_DIR）",
    )
    parser.add_argument(
        "--ion-count",
        type=int,
        default=1727,
        metavar="N",
        help="只处理日志中「离子数列」等于 N 的行（默认: 1727）",
    )
    parser.add_argument(
        "--count-column",
        type=str,
        default=None,
        metavar="COL",
        help="离子数列名；省略则自动选用 final_ions / after_equilibrium / after_y_split 中存在的列",
    )
    parser.add_argument(
        "--coords-stage",
        choices=("final", "equilibrium"),
        default="final",
        help="校验与保存哪一阶段坐标：final=ion_dist 合并后（默认）；equilibrium=合并前（适合仅含 after_y_split 的旧日志）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_ION_POS,
        help="输出目录（默认: outputs/IonPos）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将处理的文件与参数，不写 npy",
    )
    parser.add_argument(
        "--no-clean-dest",
        action="store_true",
        help="不在写入前删除输出目录中不在本次结果内的 .npy",
    )
    parser.add_argument(
        "--ion-dist",
        type=float,
        default=None,
        metavar="D",
        help="覆盖日志中的 ion_dist（像素；≤0 关闭合并）",
    )
    parser.add_argument(
        "--no-bgsub",
        action="store_true",
        help="覆盖日志：不做高斯减背景",
    )
    parser.add_argument(
        "--matched-filter",
        action="store_true",
        help="覆盖日志：启用匹配滤波",
    )
    args = parser.parse_args()

    log_path = args.blob_dir / args.log_name
    rows, fieldnames = load_log_rows(log_path)
    count_col = detect_count_column(fieldnames, args.count_column)
    filtered = filter_rows_by_count(rows, count_col, args.ion_count)
    if not filtered:
        raise SystemExit(
            f"无匹配行: {count_col}={args.ion_count}（日志共 {len(rows)} 行），请检查 --ion-count 或 --count-column",
        )
    unique_rows = dedupe_rows_by_file(filtered)
    print(
        f"[信息] 日志={log_path} 列 {count_col}={args.ion_count} "
        f"匹配 {len(filtered)} 行，按 file 去重后 {len(unique_rows)} 个",
    )
    if count_col == "after_y_split" and args.coords_stage == "final":
        print(
            "[提示] 当前日志列为 after_y_split（多为 y 分割与 x 细化之后、ion_dist 合并之前的离子数）。"
            "若默认阶段 final 校验失败，请加参数 --coords-stage equilibrium。",
        )

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    failed: list[str] = []

    for row in unique_rows:
        fname = str(row["file"]).strip()
        src = args.data_dir / fname
        if not src.is_file():
            print(f"[跳过] 未找到原始帧: {src}")
            failed.append(fname)
            continue
        try:
            base = params_from_row(row)
        except ValueError as e:
            print(f"[跳过] {fname} 日志参数解析失败: {e}")
            failed.append(fname)
            continue
        if args.ion_dist is not None:
            base = replace(base, ion_dist=float(args.ion_dist))
        if args.no_bgsub:
            base = replace(base, use_bgsub=False)
        if args.matched_filter:
            base = replace(base, use_matched_filter=True)

        if args.dry_run:
            print(
                f"  [试运行] {fname} threshold={base.threshold:g} bgsub={base.use_bgsub} "
                f"阶段={args.coords_stage}",
            )
            continue

        raw = np.load(src)
        if raw.ndim == 3:
            raw = raw.mean(axis=0)
        image = np.asarray(raw, dtype=np.float64)

        t0 = time.perf_counter()
        try:
            eq_arr, fin_arr = run_blob_detection(image, base)
        except Exception as e:
            print(f"[失败] {fname} 识别过程异常: {e}")
            failed.append(fname)
            continue
        elapsed = time.perf_counter() - t0

        if args.coords_stage == "final":
            out_coords = fin_arr
        else:
            out_coords = eq_arr

        if out_coords.shape[0] != args.ion_count:
            print(
                f"[失败] {fname} 离子数校验未通过: 阶段={args.coords_stage} 得到 {out_coords.shape[0]}，"
                f"期望 {args.ion_count}（耗时 {elapsed:.2f}s）；可尝试 --coords-stage equilibrium 或调整 --ion-dist",
            )
            failed.append(fname)
            continue

        if out_coords.ndim != 2 or out_coords.shape[1] != 2:
            print(f"[失败] {fname} 坐标数组形状异常: {out_coords.shape}")
            failed.append(fname)
            continue

        dst = args.out_dir / fname
        np.save(dst, out_coords)
        written.append(fname)
        print(
            f"[成功] {fname} -> {dst}  N={out_coords.shape[0]}  阶段={args.coords_stage}  ({elapsed:.2f}s)",
        )

    if args.dry_run:
        print("[完成] 试运行结束，未写入文件")
        return

    if not args.no_clean_dest:
        keep = set(written)
        for p in args.out_dir.glob("*.npy"):
            if p.name not in keep:
                p.unlink()
                print(f"  [已删除] 输出目录多余文件 {p.name}")

    print(
        f"[汇总] 成功 {len(written)}，失败或跳过 {len(failed)}，输出目录 {args.out_dir}",
    )
    if not written:
        raise SystemExit("未成功写入任何坐标 .npy 文件")


if __name__ == "__main__":
    main()
