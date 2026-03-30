"""python -m ion_detect.blob_cli — 连通域 + 轴对齐最小外接矩形批处理。"""
from __future__ import annotations

import argparse
from typing import Literal
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import DEFAULT_DATA_DIR, OUT_BLOB_CONNECTED

from ion_detect.blob_viz import visualize_blob_workflow
from ion_detect.blob_workflow import run_blob_workflow
from ion_detect.cli_helpers import resolve_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_blob_workflow：默认仅减背景（可加 --matched-filter）→ boundary → 二值化 → 连通域 → 轴对齐矩形；"
        "输出上下两幅子图 PNG（上：灰度+矩形；下：二值+矩形）",
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
        required=True,
        metavar="T",
        help="二值化阈值（作用在 --binarize-on 所选图上）",
    )
    parser.add_argument(
        "--binarize-on",
        choices=("denoised_map", "signal"),
        default="denoised_map",
        help="denoised_map=在 denoised_map 上阈值（默认无匹配滤波时等同 bgsub）；signal=始终在 signal 上阈值",
    )
    parser.add_argument(
        "--no-bgsub",
        action="store_true",
        help="不做高斯减背景：signal=原图浮点，boundary/二值化均基于此（默认做 bgsub）",
    )
    parser.add_argument(
        "--matched-filter",
        action="store_true",
        help="对 signal 做与 detect_ions 相同的匹配滤波；denoised_map 为滤波图，否则 denoised_map=signal",
    )
    parser.add_argument("--binarize-strict", action="store_true", help="前景为 >T 而非 >=T")
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="连通域邻接：4 或 8",
    )
    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=1,
        metavar="N",
        help="忽略像素数 < N 的连通域",
    )
    parser.add_argument("--show", action="store_true", help="弹窗（仍会保存 PNG）")
    parser.add_argument(
        "--no-boundary-ellipse",
        action="store_true",
        help="左图不绘制 crystal boundary 椭圆",
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

    for idx in selected:
        target = files[idx]
        print(f"\n[{idx:04d}] {target.name}")
        image = np.load(target)
        if image.ndim == 3:
            image = image.mean(axis=0)
        image = np.asarray(image, dtype=np.float64)

        conn: Literal[4, 8] = 4 if args.connectivity == 4 else 8
        t0 = time.perf_counter()
        res = run_blob_workflow(
            image,
            args.threshold,
            use_bgsub=not args.no_bgsub,
            use_matched_filter=args.matched_filter,
            binarize_on=args.binarize_on,
            ge=not args.binarize_strict,
            connectivity=conn,
            min_area_pixels=int(args.min_area_pixels),
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  labels={res.n_components}, kept_rects={len(res.rects)}, "
            f"boundary={'OK' if res.preprocess.boundary else 'None'}, {elapsed:.2f}s",
        )

        stem = target.stem
        safe = stem.encode("ascii", "replace").decode("ascii")
        thr_s = f"{args.threshold:g}".replace(".", "p").replace("-", "m")
        png = out_dir / f"blob_workflow_{idx:04d}_{safe}_thr{thr_s}.png"
        visualize_blob_workflow(
            image,
            res.binary,
            boundary=res.preprocess.boundary,
            rects=res.rects,
            title=f"[{idx:04d}] {target.name} thr={args.threshold:g} src={args.binarize_on}",
            output_path=png,
            show=args.show,
            draw_boundary=not args.no_boundary_ellipse,
        )


if __name__ == "__main__":
    main()
