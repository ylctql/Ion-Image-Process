"""python -m ion_detect 入口。"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from output_paths import (
    DEFAULT_DATA_DIR,
    OUT_BGSUB_BIN_IMGS,
    OUT_BGSUB_IMGS,
    OUT_ION_DETECT_IMGS,
    OUT_ION_POS,
)

from .cli_helpers import resolve_indices
from .pipeline import detect_ions
from .viz import (
    print_summary,
    visualize,
    visualize_bgsub,
    visualize_bgsub_binarized,
)


def main():
    parser = argparse.ArgumentParser(description="离子检测与椭圆拟合可视化")
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help=(
            "要处理的索引规范，支持整数和 numpy 风格切片，并支持并集。"
            "例如: 0 5 -1 ::3,0:10"
        ),
    )
    parser.add_argument(
        "--save-pos",
        action="store_true",
        help="保存离子中心坐标到 .npy 文件。",
    )
    parser.add_argument(
        "--save-bgsub-img",
        action="store_true",
        help="保存减高斯背景后的 signal 图 (bgsub, 与 detect_ions 首轮检测用图一致)。",
    )
    parser.add_argument(
        "--bgsub-binarize-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "对减高斯背景后的 bgsub (与 detect_ions 首轮 signal 一致) 按阈值 T 二值化，"
            "并保存两张独立 PNG：*_bgsub.png 与 *_mask.png。不指定则不生成。"
        ),
    )
    parser.add_argument(
        "--bgsub-binarize-strict",
        action="store_true",
        help="与 --bgsub-binarize-threshold 合用：前景为 bgsub > T（不含等于）。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "弹窗显示检测图与（若有）bgsub / 二值化；关闭当前窗口后继续下一帧。"
            "仍会保存 PNG（与默认一致）。"
        ),
    )
    args = parser.parse_args()

    data_dir = DEFAULT_DATA_DIR
    out_dir = OUT_ION_DETECT_IMGS
    out_dir.mkdir(parents=True, exist_ok=True)
    bgsub_dir = OUT_BGSUB_IMGS
    bgsub_bin_dir = OUT_BGSUB_BIN_IMGS
    pos_dir = OUT_ION_POS
    if args.save_pos:
        pos_dir.mkdir(exist_ok=True)

    want_bgsub_bin = args.bgsub_binarize_threshold is not None
    if args.bgsub_binarize_strict and not want_bgsub_bin:
        print("警告: --bgsub-binarize-strict 已忽略 (需同时指定 --bgsub-binarize-threshold)。")
    want_bgsub_save = bool(args.save_bgsub_img)
    return_bgsub = want_bgsub_save or want_bgsub_bin

    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    print(f"共找到 {len(files)} 个 npy 文件")
    selected_indices = resolve_indices(args.indices, len(files))
    if not selected_indices:
        raise ValueError("没有可处理的索引。请检查输入的 indices。")

    for idx in selected_indices:
        target = files[idx]
        print(f"\n[{idx:04d}] 加载: {target.name}")
        image = np.load(target)

        print("正在检测离子...")
        t0 = time.perf_counter()
        if return_bgsub:
            ions, boundary, bgsub_map = detect_ions(image, return_bgsub=True)
        else:
            ions, boundary = detect_ions(image)
            bgsub_map = None
        elapsed = time.perf_counter() - t0
        print(f"检测耗时: {elapsed:.2f} 秒")
        if boundary:
            cx, cy, a, b = boundary
            print(f"晶格边界椭圆: 中心=({cx:.1f}, {cy:.1f}), "
                  f"半轴=({a:.1f}, {b:.1f})")

        print_summary(ions)

        if args.save_pos:
            positions = np.array([[ion["x0"], ion["y0"]] for ion in ions], dtype=np.float64)
            if positions.size == 0:
                positions = np.empty((0, 2), dtype=np.float64)
            pos_path = pos_dir / target.name
            np.save(pos_path, positions)
            print(f"[已保存离子中心] {pos_path}")

        out_path = out_dir / f"ion_ellipses_{idx:04d}.png"
        visualize(
            image,
            ions,
            n_sigma=2.0,
            title=f"[{idx:04d}] {target.name}",
            output_path=out_path,
            boundary=boundary,
            show=args.show,
        )

        if want_bgsub_save and bgsub_map is not None:
            bgsub_dir.mkdir(parents=True, exist_ok=True)
            bgsub_path = bgsub_dir / f"ion_bgsub_{idx:04d}.png"
            visualize_bgsub(
                bgsub_map,
                ions,
                n_sigma=2.0,
                title=f"[{idx:04d}] {target.name}",
                output_path=bgsub_path,
                boundary=boundary,
                show=args.show,
            )

        if want_bgsub_bin and bgsub_map is not None:
            bgsub_bin_dir.mkdir(parents=True, exist_ok=True)
            thr = float(args.bgsub_binarize_threshold)
            thr_fn = f"{thr:g}".replace(".", "p").replace("-", "m")
            bin_path = bgsub_bin_dir / f"ion_bgsub_binary_{idx:04d}_thr{thr_fn}.png"
            visualize_bgsub_binarized(
                bgsub_map,
                thr,
                ions,
                n_sigma=2.0,
                title=f"[{idx:04d}] {target.name}",
                output_path=bin_path,
                boundary=boundary,
                show=args.show,
                ge=not args.bgsub_binarize_strict,
            )


if __name__ == "__main__":
    main()
