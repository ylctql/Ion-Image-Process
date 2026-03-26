"""python -m ion_detect 入口。"""
import argparse
import sys
import numpy as np
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from output_paths import (
    OUT_AMP_Y_FIT,
    OUT_ION_DETECT_IMGS,
    OUT_ION_POS,
    OUT_RESIDUAL_IMGS,
)

from .cli_helpers import resolve_indices
from .pipeline import detect_ions
from .viz import print_summary, visualize, visualize_peel_residual


_PEEL_UNSET = object()  # --peak-peel 未传入


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
        "--pos-dir",
        type=Path,
        default=None,
        help="离子中心保存目录，默认 outputs/IonPos。",
    )
    parser.add_argument(
        "--use-y-thresh-comp",
        action="store_true",
        help="启用 y 方向亮度模型补偿阈值(降低边缘较暗区域的判定门槛)。",
    )
    parser.add_argument(
        "--amp-coef-path",
        type=Path,
        default=None,
        help=(
            "y 方向亮度拟合系数文件(.npy)。"
            "默认使用 outputs/amp_y_fit/amp_vs_y_coef_10.npy。"
        ),
    )
    parser.add_argument(
        "--amp-coef-mode",
        type=str,
        choices=("even", "poly2"),
        default="even",
        help="系数解释方式: even=[a0,a2,a4], poly2=[p2,p1,p0]。",
    )
    parser.add_argument(
        "--comp-floor",
        type=float,
        default=0.2,
        help="阈值缩放下限，越小边缘越容易被检出(也更易引入噪声)。",
    )
    parser.add_argument(
        "--fix-theta-zero",
        action="store_true",
        help="高斯拟合固定椭圆转角 θ=0 (轴对齐 x/y，不拟合旋转)。",
    )
    parser.add_argument(
        "--no-matched-filter",
        action="store_true",
        help="禁用匹配滤波; 峰检测直接使用减背景后的信号 (默认启用匹配滤波)。",
    )
    parser.add_argument(
        "--joint-pair-y-gap",
        type=float,
        default=None,
        metavar="DY",
        help=(
            "启用 y 向近邻双峰联合拟合 (N=2, θ=0): 两候选峰 |Δy|≤DY 且 |Δx| 在 "
            "--joint-pair-x-gap 内则合并 ROI 同时拟合; 失败则回退单峰。"
        ),
    )
    parser.add_argument(
        "--joint-pair-x-gap",
        type=float,
        default=None,
        metavar="DX",
        help="联合配对允许的 |Δx| (像素); 省略时默认 max(4, 拟合半宽 hw_x)。",
    )
    parser.add_argument(
        "--peak-peel",
        nargs="?",
        const="",
        default=_PEEL_UNSET,
        metavar="MODE",
        help=(
            "启用峰值剥离: 省略 MODE 为全图第二轮; MODE=edge 为仅 y 向边缘带第二轮 "
            "(大小写不敏感); 其他 MODE 视为全图第二轮。若下一参数是帧下标，请写在其前或使用 "
            "--peak-peel=edge。"
        ),
    )
    parser.add_argument(
        "--peak-peel-min-sep",
        type=float,
        default=2.0,
        metavar="PX",
        help="剥离轮新峰与已有峰中心的最小距离 (像素)。",
    )
    parser.add_argument(
        "--peak-peel-y-edge-frac",
        type=float,
        default=0.25,
        metavar="F",
        help="y 向边缘带: 仅保留 |y-cy|/b ≥ 1-F 的候选, 默认 0.25 即 ≥0.75。",
    )
    parser.add_argument(
        "--peak-peel-rel-threshold",
        type=float,
        default=None,
        metavar="R",
        help="剥离第二轮相对阈值 (不传则与首轮相同); 略提高可抑制残差弱峰。",
    )
    parser.add_argument(
        "--peak-peel-min-amp-frac",
        type=float,
        default=None,
        metavar="Q",
        help="第二轮峰振幅须 ≥ Q×首轮振幅中位数 (抑制弱伪峰)。",
    )
    parser.add_argument(
        "--save-residual-img",
        action="store_true",
        help="保存 peak-peel 残差图 (须配合 --peak-peel; 首轮无离子时不生成)。",
    )
    parser.add_argument(
        "--residual-img-dir",
        type=Path,
        default=None,
        help="残差图保存目录，默认 outputs/residual_imgs。",
    )
    args = parser.parse_args()

    if args.peak_peel is _PEEL_UNSET:
        peak_peel = False
        peak_peel_y_edges_only = False
    else:
        peak_peel = True
        peak_peel_y_edges_only = str(args.peak_peel).strip().lower() == "edge"

    project_root = _PROJECT_ROOT
    data_dir = project_root / "20260305_1727"
    out_dir = OUT_ION_DETECT_IMGS
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = args.residual_img_dir or OUT_RESIDUAL_IMGS
    pos_dir = args.pos_dir or OUT_ION_POS
    default_amp_coef_path = OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy"
    amp_coef_path = args.amp_coef_path or default_amp_coef_path
    if args.save_pos:
        pos_dir.mkdir(exist_ok=True)

    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    print(f"共找到 {len(files)} 个 npy 文件")
    selected_indices = resolve_indices(args.indices, len(files))
    if not selected_indices:
        raise ValueError("没有可处理的索引。请检查输入的 indices。")

    if args.save_residual_img and not peak_peel:
        print("警告: --save-residual-img 已忽略 (需同时使用 --peak-peel)。")
    want_residual = bool(args.save_residual_img and peak_peel)

    detect_kw = dict(
        use_y_threshold_comp=args.use_y_thresh_comp,
        amp_y_coef_path=amp_coef_path,
        amp_y_coef_mode=args.amp_coef_mode,
        comp_floor=args.comp_floor,
        fix_theta_zero=args.fix_theta_zero,
        use_matched_filter=not args.no_matched_filter,
        joint_pair_y_gap=args.joint_pair_y_gap,
        joint_pair_x_gap=args.joint_pair_x_gap,
        peak_peel=peak_peel,
        peak_peel_min_sep=args.peak_peel_min_sep,
        peak_peel_y_edges_only=peak_peel_y_edges_only,
        peak_peel_y_edge_frac=args.peak_peel_y_edge_frac,
        peak_peel_rel_threshold=args.peak_peel_rel_threshold,
        peak_peel_min_amp_frac=args.peak_peel_min_amp_frac,
        peak_peel_margin_sigma=4.5,
    )

    for idx in selected_indices:
        target = files[idx]
        print(f"\n[{idx:04d}] 加载: {target.name}")
        image = np.load(target)

        print("正在检测离子...")
        t0 = time.perf_counter()
        if want_residual:
            ions, boundary, peel_residual = detect_ions(
                image, **detect_kw, return_peel_residual=True,
            )
        else:
            ions, boundary = detect_ions(image, **detect_kw)
            peel_residual = None
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
        visualize(image, ions, n_sigma=2.0,
                  title=f"[{idx:04d}] {target.name}",
                  output_path=out_path,
                  show_zoom=True, boundary=boundary)

        if want_residual:
            if peel_residual is None:
                print("[残差图] 首轮无检出离子, 跳过保存。")
            else:
                residual_dir.mkdir(parents=True, exist_ok=True)
                residual_path = residual_dir / f"peak_peel_residual_{idx:04d}.png"
                visualize_peel_residual(
                    peel_residual,
                    title=f"[{idx:04d}] {target.name}",
                    output_path=residual_path,
                    boundary=boundary,
                    reference_image=image,
                    peak_peel_y_edges_only=peak_peel_y_edges_only,
                    peak_peel_y_edge_frac=args.peak_peel_y_edge_frac,
                )


if __name__ == "__main__":
    main()
