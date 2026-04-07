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
        required=True,
        metavar="T",
        help="二值化阈值（作用在：可选 bgsub 后的 signal，再可选匹配滤波后的浮点图上）",
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
            res.preprocess.denoised_map,
            res.binary,
            boundary=res.preprocess.boundary,
            rects=res.rects,
            title=f"[{idx:04d}] {target.name}",
            threshold=float(args.threshold),
            use_bgsub=not args.no_bgsub,
            use_matched_filter=args.matched_filter,
            output_path=png,
            show=args.show,
        )


if __name__ == "__main__":
    main()
