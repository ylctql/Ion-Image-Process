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

from .blob_ion_positions import merge_close_ion_positions_xy
from .cli_helpers import resolve_frame_paths_from_args, resolve_indices
from .frame_io import list_frame_files, load_frame
from .pipeline import detect_ions
from .viz import (
    print_summary,
    visualize_bgsub_binarized_markers,
    visualize_bgsub_markers,
    visualize_ion_positions_markers,
)


def _detect_ions_kwargs_from_args(args: argparse.Namespace) -> dict:
    """与 ``pipeline.detect_ions`` 默认一致，由 CLI 覆盖。"""
    return {
        "bg_sigma": (float(args.bg_sigma_y), float(args.bg_sigma_x)),
        "peak_size": (int(args.peak_size_y), int(args.peak_size_x)),
        "rel_threshold": float(args.rel_threshold),
        "fit_hw": (int(args.fit_hw_y), int(args.fit_hw_x)),
        "sigma_range": (float(args.sigma_min), float(args.sigma_max)),
        "refine": bool(args.refine),
        "fix_theta_zero": bool(args.fix_theta_zero),
    }


def main():
    parser = argparse.ArgumentParser(
        description="离子检测与椭圆拟合可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=None,
        help=(
            "要处理的索引规范，支持整数和 numpy 风格切片，并支持并集。"
            "例如: 0 5 -1 ::3,0:10。"
            "若使用 --file 指定帧，则忽略本参数（文件名优先）。"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help=(
            "帧数据目录（含 .npy / 栅格图）。默认与 output_paths.DEFAULT_DATA_DIR 相同"
            "（一般为项目上级目录下的 20260305_1727）。"
        ),
    )
    parser.add_argument(
        "--file",
        dest="frame_files",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "指定待处理帧；可多次使用。"
            "若参数含路径（子目录或绝对路径），则直接使用该路径，不再与 --data-dir 拼接；"
            "若仅为文件名/stem，则在 --data-dir 下解析。"
            "一旦提供，仅处理所列文件，且优先于位置参数 indices。"
        ),
    )
    parser.add_argument(
        "--save-pos",
        action="store_true",
        help=(
            "保存识别最终结果（与输出图中洋红 + 一致，ion-dist 合并后）的离子中心到 .npy。"
            "文件名：各帧源文件去扩展名 + .npy；目录见 --pos-out-dir。"
        ),
    )
    parser.add_argument(
        "--pos-out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "仅与 --save-pos 合用：坐标输出目录，默认与 output_paths.OUT_ION_POS 相同。"
            "若指定则先 expanduser/resolve 并自动创建目录。"
        ),
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
    parser.add_argument(
        "--ion-dist",
        type=float,
        default=4.0,
        metavar="D",
        help=(
            "检测后对离子中心做近距合并：欧氏距离严格小于 D 的一对合并为两点中点（贪心选最近对）；"
            "≤0 时不合并。输出图为洋红 + 标记（与 blob_cli 一致）。"
        ),
    )
    det = parser.add_argument_group(
        "detect_ions（峰识别 / 高斯拟合，默认与 pipeline.detect_ions 一致）",
    )
    det.add_argument(
        "--bg-sigma-y",
        type=float,
        default=10.0,
        metavar="Y",
        help="高斯背景估计 sigma 的 y 向分量",
    )
    det.add_argument(
        "--bg-sigma-x",
        type=float,
        default=30.0,
        metavar="X",
        help="高斯背景估计 sigma 的 x 向分量",
    )
    det.add_argument(
        "--peak-size-y",
        type=int,
        default=5,
        metavar="NY",
        help="detect_map 上局部极大检测窗口高度（y）",
    )
    det.add_argument(
        "--peak-size-x",
        type=int,
        default=9,
        metavar="NX",
        help="detect_map 上局部极大检测窗口宽度（x）",
    )
    det.add_argument(
        "--rel-threshold",
        type=float,
        default=0.025,
        metavar="R",
        help="峰检测相对阈值（相对 detect_map 最大值）",
    )
    det.add_argument(
        "--fit-hw-y",
        type=int,
        default=3,
        metavar="HY",
        help="高斯拟合半窗口（y）",
    )
    det.add_argument(
        "--fit-hw-x",
        type=int,
        default=4,
        metavar="HX",
        help="高斯拟合半窗口（x）",
    )
    det.add_argument(
        "--sigma-min",
        type=float,
        default=0.3,
        metavar="S0",
        help="允许的高斯 sigma 下限（短/长轴中较小者与较大者均约束在此区间）",
    )
    det.add_argument(
        "--sigma-max",
        type=float,
        default=3.5,
        metavar="S1",
        help="允许的高斯 sigma 上限",
    )
    det.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        default=True,
        help="关闭两阶段精修",
    )
    det.add_argument(
        "--no-fix-theta-zero",
        dest="fix_theta_zero",
        action="store_false",
        default=True,
        help="不固定 θ=0，允许拟合旋转椭圆",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = OUT_ION_DETECT_IMGS
    out_dir.mkdir(parents=True, exist_ok=True)
    bgsub_dir = OUT_BGSUB_IMGS
    bgsub_bin_dir = OUT_BGSUB_BIN_IMGS
    if args.pos_out_dir is not None and not args.save_pos:
        print("警告: --pos-out-dir 已忽略（需同时指定 --save-pos）。")
    pos_dir = OUT_ION_POS
    if args.save_pos:
        pos_dir = (
            Path(args.pos_out_dir).expanduser().resolve()
            if args.pos_out_dir is not None
            else OUT_ION_POS
        )
        pos_dir.mkdir(parents=True, exist_ok=True)

    want_bgsub_bin = args.bgsub_binarize_threshold is not None
    if args.bgsub_binarize_strict and not want_bgsub_bin:
        print("警告: --bgsub-binarize-strict 已忽略 (需同时指定 --bgsub-binarize-threshold)。")
    want_bgsub_save = bool(args.save_bgsub_img)
    return_bgsub = want_bgsub_save or want_bgsub_bin

    print(f"数据目录: {data_dir}")

    indices_arg = args.indices if args.indices else ["0"]
    if args.frame_files:
        selected = resolve_frame_paths_from_args(args.frame_files, data_dir)
        print(f"将处理 {len(selected)} 个指定帧。")
        if args.indices:
            print("提示: 已指定 --file，忽略位置参数 indices。")
    else:
        files = list_frame_files(data_dir)
        if not files:
            raise FileNotFoundError(
                f"目录中无可用帧文件: {data_dir}（需要 .npy 或栅格图: .jpg/.jpeg/.png 等）",
            )
        n_npy = sum(1 for p in files if p.suffix.lower() == ".npy")
        n_raster = len(files) - n_npy
        print(f"共找到 {len(files)} 个帧文件（.npy: {n_npy}，栅格图: {n_raster}）")
        selected_indices = resolve_indices(indices_arg, len(files))
        if not selected_indices:
            raise ValueError("没有可处理的索引。请检查输入的 indices。")
        selected = [(i, files[i]) for i in selected_indices]

    for seq, (idx, target) in enumerate(selected):
        tag = idx if idx is not None else seq
        print(f"\n[{tag:04d}] 加载: {target}")
        image = load_frame(target)

        print("正在检测离子...")
        t0 = time.perf_counter()
        dkw = _detect_ions_kwargs_from_args(args)
        if return_bgsub:
            ions, boundary, bgsub_map = detect_ions(image, return_bgsub=True, **dkw)
        else:
            ions, boundary = detect_ions(image, **dkw)
            bgsub_map = None
        elapsed = time.perf_counter() - t0
        print(f"检测耗时: {elapsed:.2f} 秒")
        if boundary:
            cx, cy, a, b = boundary
            print(f"晶格边界椭圆: 中心=({cx:.1f}, {cy:.1f}), "
                  f"半轴=({a:.1f}, {b:.1f})")

        print_summary(ions)

        raw_xy = [(float(ion["x0"]), float(ion["y0"])) for ion in ions]
        merged_xy, n_ion_merges = merge_close_ion_positions_xy(
            raw_xy,
            float(args.ion_dist),
            pairwise_midpoint=True,
        )
        print(
            f"ion-dist 合并 (阈值={float(args.ion_dist):g} px): "
            f"{n_ion_merges} 次合并 -> {len(merged_xy)} 个位置",
        )

        if args.save_pos:
            positions = np.array(merged_xy, dtype=np.float64)
            if positions.size == 0:
                positions = np.empty((0, 2), dtype=np.float64)
            pos_path = pos_dir / f"{target.stem}.npy"
            np.save(pos_path, positions)
            print(f"[已保存离子中心] {pos_path}")

        out_path = out_dir / f"ion_positions_{tag:04d}.png"
        visualize_ion_positions_markers(
            image,
            merged_xy,
            boundary=boundary,
            title=f"[{tag:04d}] {target.name}",
            output_path=out_path,
            show=args.show,
        )

        if want_bgsub_save and bgsub_map is not None:
            bgsub_dir.mkdir(parents=True, exist_ok=True)
            bgsub_path = bgsub_dir / f"ion_bgsub_{tag:04d}.png"
            visualize_bgsub_markers(
                bgsub_map,
                merged_xy,
                boundary=boundary,
                title=f"[{tag:04d}] {target.name}",
                output_path=bgsub_path,
                show=args.show,
            )

        if want_bgsub_bin and bgsub_map is not None:
            bgsub_bin_dir.mkdir(parents=True, exist_ok=True)
            thr = float(args.bgsub_binarize_threshold)
            thr_fn = f"{thr:g}".replace(".", "p").replace("-", "m")
            bin_path = bgsub_bin_dir / f"ion_bgsub_binary_{tag:04d}_thr{thr_fn}.png"
            visualize_bgsub_binarized_markers(
                bgsub_map,
                thr,
                merged_xy,
                title=f"[{tag:04d}] {target.name}",
                output_path=bin_path,
                boundary=boundary,
                show=args.show,
                ge=not args.bgsub_binarize_strict,
            )


if __name__ == "__main__":
    main()
