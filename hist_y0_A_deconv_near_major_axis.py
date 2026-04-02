"""
对多帧离子图运行与 ``detect_ions`` 相同的流水线，筛出 ``|y0 - cy| <= tol`` 的离子
（boundary 长轴沿 x，过长轴为 ``y = cy``）。

对每个入选离子：
- 取识别椭圆（与 ``visualize`` 一致：``n_sigma`` × ``sigma_minor`` / ``sigma_major``、
  ``theta_deg``）的轴对齐外接正方形，长宽各向外扩 1 像素得到基础 ROI；可用
  ``--deconv-roi-pad`` / ``--deconv-min-half`` 再扩大反卷积区域，减轻小块上 FFT
  周期边界与 PSF 裁剪带来的误差；
- 在 ROI 上对原始图 ``image`` 做维纳式反卷积，核为 ``exp(-(x^2+y^2)/2)``（离散网格）；
- 将反卷积结果用 ``f(y) = c0 / sqrt(A^2 - y^2)`` 拟合（``y`` 为相对识别中心 ``y0`` 的像素偏移）；
- 汇总所有帧的 ``(y0, A)``，画散点图并对 ``A ~ y0`` 做线性拟合。

每识别完一帧保存标记图（与 ``visualize(..., near_major_axis_tol=...)`` 相同逻辑）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.optimize import curve_fit

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.boundary import offset_perpendicular_to_boundary_major_axis
from ion_detect.cli_helpers import resolve_indices
from ion_detect.pipeline import detect_ions
from ion_detect.viz import visualize
from output_paths import DEFAULT_DATA_DIR, OUT_HISTOGRAM


def ellipse_axis_aligned_half_extents(
    n_sigma: float,
    sigma_minor: float,
    sigma_major: float,
    theta_deg: float,
) -> tuple[float, float]:
    """与 matplotlib Ellipse(width=2*n_sigma*σ_minor, height=2*n_sigma*σ_major, angle) 一致的 AABB 半宽。"""
    w2 = float(n_sigma) * float(sigma_minor)
    h2 = float(n_sigma) * float(sigma_major)
    th = np.deg2rad(float(theta_deg))
    c, s = np.cos(th), np.sin(th)
    ex = float(np.sqrt((w2 * c) ** 2 + (h2 * s) ** 2))
    ey = float(np.sqrt((w2 * s) ** 2 + (h2 * c) ** 2))
    return ex, ey


def roi_square_bounds(
    x0: float,
    y0: float,
    half_side: float,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    """外接正方形半宽 ``half_side``（已含扩边），返回 ``y1,y2,x1,x2`` 切片 ``[y1:y2, x1:x2]``。"""
    hs = float(half_side)
    x1 = max(0, int(np.floor(x0 - hs)))
    x2 = min(w, int(np.ceil(x0 + hs)) + 1)
    y1 = max(0, int(np.floor(y0 - hs)))
    y2 = min(h, int(np.ceil(y0 + hs)) + 1)
    return y1, y2, x1, x2


def build_kernel_exp_r2_half(radius: int) -> np.ndarray:
    """``exp(-(x^2+y^2)/2)``，整数坐标 ``x,y in [-radius, radius]``。"""
    r = int(radius)
    ax = np.arange(-r, r + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return np.exp(-0.5 * (xx ** 2 + yy ** 2))


def _crop_center_2d(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """取 ``arr`` 中心与 ``out_shape`` 同尺寸的子块（``arr`` 须不小于目标尺寸）。"""
    oh, ow = out_shape
    ah, aw = arr.shape
    if ah < oh or aw < ow:
        raise ValueError(f"kernel {arr.shape} smaller than patch {out_shape}")
    y0 = (ah - oh) // 2
    x0 = (aw - ow) // 2
    return arr[y0 : y0 + oh, x0 : x0 + ow].astype(np.float64, copy=False)


def _embed_kernel_centered(kernel: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """小核置于画布中心后 ``ifftshift``，便于 ``fft2`` 与卷积定理一致。"""
    oh, ow = out_shape
    kh, kw = kernel.shape
    if kh > oh or kw > ow:
        kernel = _crop_center_2d(kernel, (oh, ow))
        kh, kw = kernel.shape
    canvas = np.zeros((oh, ow), dtype=np.float64)
    y0 = (oh - kh) // 2
    x0 = (ow - kw) // 2
    canvas[y0 : y0 + kh, x0 : x0 + kw] = np.asarray(kernel, dtype=np.float64)
    return np.fft.ifftshift(canvas)


def wiener_deconv2d(patch: np.ndarray, kernel_small: np.ndarray, reg: float) -> np.ndarray:
    """简易维纳反卷积：``F = conj(H) / (|H|^2 + reg) * G``。"""
    patch = np.asarray(patch, dtype=np.float64)

    H = fft2(_embed_kernel_centered(kernel_small, patch.shape))
    G = fft2(patch)
    denom = (np.abs(H) ** 2) + float(reg)
    F = np.conj(H) / denom * G
    out = np.real(ifft2(F))
    return out


def fit_inv_sqrt_profile(
    yy_rel: np.ndarray,
    vals: np.ndarray,
    *,
    a_lo_delta: float = 0.3,
    a_max_extra: float = 12.0,
) -> tuple[float, float] | None:
    """``vals ≈ c0 / sqrt(A^2 - yy_rel^2)``，``yy_rel`` 为相对中心的行偏移。

    **参数 A 不可判据**：当 ``A >> max|y|`` 时 ``sqrt(A^2-y^2) ≈ A``，模型在 ROI 内
    几乎与 ``y`` 无关；若反卷积图块状、无明显脊形，最小二乘会把 ``A`` 顶到上界，
    得到上百像素量级的假拟合。因此上界取 ``max|y| + a_max_extra``（默认约十余像素），
    迫使 ``f(y)`` 在 ROI 内有可辨识的 ``y`` 依赖；但仍需结合物理检查残差。
    """

    def model(y: np.ndarray, c0: float, A: float) -> np.ndarray:
        inner = np.maximum(A ** 2 - y ** 2, 1e-18)
        return c0 / np.sqrt(inner)

    y = np.asarray(yy_rel, dtype=np.float64).ravel()
    v = np.asarray(vals, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(v)
    if int(ok.sum()) < 6:
        return None
    y, v = y[ok], v[ok]
    pos = v > 0
    if int(pos.sum()) >= 6:
        y, v = y[pos], v[pos]
    else:
        v = np.maximum(v, 1e-9)
    ymax = float(np.max(np.abs(y)))
    if ymax < 1e-9:
        return None
    a_lo = ymax + float(a_lo_delta)
    a_hi = ymax + float(a_max_extra)
    if a_hi <= a_lo + 1e-6:
        return None
    A0 = float(np.clip(ymax + 0.75, a_lo + 1e-3, a_hi - 1e-3))
    c0_guess = float(np.median(v) * A0)
    try:
        popt, _ = curve_fit(
            model,
            y,
            v,
            p0=(c0_guess, A0),
            bounds=((1e-12, a_lo), (np.inf, a_hi)),
            maxfev=20000,
        )
        c0, A = float(popt[0]), float(popt[1])
        if not (np.isfinite(c0) and np.isfinite(A)):
            return None
        return c0, A
    except (ValueError, RuntimeError):
        return None


def process_frame(
    image: np.ndarray,
    ions: list,
    boundary: tuple,
    *,
    tol: float,
    n_sigma_roi: float,
    kernel_radius: int,
    wiener_reg: float,
    a_max_extra: float,
    deconv_roi_pad: float,
    deconv_min_half: float,
) -> tuple[list[float], list[float]]:
    """返回本帧入选离子的 ``y0`` 列表与 ``A`` 列表。"""
    h, w = image.shape
    img = np.asarray(image, dtype=np.float64)
    ksmall = build_kernel_exp_r2_half(kernel_radius)
    y0_list: list[float] = []
    A_list: list[float] = []

    for ion in ions:
        off = offset_perpendicular_to_boundary_major_axis(
            ion["x0"], ion["y0"], boundary,
        )
        if off is None or abs(off) > tol:
            continue

        ex, ey = ellipse_axis_aligned_half_extents(
            n_sigma_roi,
            float(ion["sigma_minor"]),
            float(ion["sigma_major"]),
            float(ion["theta_deg"]),
        )
        half_side = max(ex, ey) + 1.0 + float(deconv_roi_pad)
        if float(deconv_min_half) > 0.0:
            half_side = max(half_side, float(deconv_min_half))
        x0 = float(ion["x0"])
        y0 = float(ion["y0"])
        y1, y2, x1, x2 = roi_square_bounds(x0, y0, half_side, h, w)
        if y2 - y1 < 3 or x2 - x1 < 3:
            continue

        patch = img[y1:y2, x1:x2].copy()
        dec = wiener_deconv2d(patch, ksmall, wiener_reg)

        jj, ii = np.mgrid[y1:y2, x1:x2]
        yy_rel = jj.astype(np.float64) - y0
        fit_r = fit_inv_sqrt_profile(yy_rel, dec, a_max_extra=a_max_extra)
        if fit_r is None:
            continue
        _, A = fit_r
        y0_list.append(y0)
        A_list.append(A)

    return y0_list, A_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="长轴附近离子：ROI 反卷积 + 1/sqrt(A²-y²) 拟合；y0–A 散点与线性拟合",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引（与 python -m ion_detect 相同）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f".npy 数据目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--major-axis-tol",
        type=float,
        default=30.0,
        help="|y0 - cy| 上限 (像素)，默认 30",
    )
    parser.add_argument(
        "--n-sigma-roi",
        type=float,
        default=2.0,
        help="ROI 椭圆与 visualize 一致：半轴 = n_sigma × σ，默认 2",
    )
    parser.add_argument(
        "--kernel-radius",
        type=int,
        default=12,
        help="核 exp(-(x²+y²)/2) 的整数半径（网格边长 2r+1），默认 12",
    )
    parser.add_argument(
        "--wiener-reg",
        type=float,
        default=1e-3,
        help="维纳分母正则 |H|² + reg，默认 1e-3",
    )
    parser.add_argument(
        "--a-max-extra",
        type=float,
        default=12.0,
        help="拟合参数 A 上界 = ROI 内 max|y| + 本值 (像素)；过大会允许“近似常数”退化解，默认 12",
    )
    parser.add_argument(
        "--deconv-roi-pad",
        type=float,
        default=0.0,
        help="在椭圆外接正方形半边上额外增加的反卷积 ROI 半宽 (像素)，减轻小块 FFT 边界/核裁剪误差，默认 0",
    )
    parser.add_argument(
        "--deconv-min-half",
        type=float,
        default=0.0,
        help="反卷积 ROI 半边长下限 (像素)；0 表示不强制。可设为 ≥ kernel 半径以包住完整核",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="散点图 PNG；默认 outputs/histogram/y0_vs_A_near_major_axis.png",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="每帧标记图目录；默认 outputs/histogram/near_major_axis_deconv_detect",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不保存识别标记图",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录中无 .npy: {data_dir}")

    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有有效索引")

    tol = float(args.major_axis_tol)
    n_sigma_roi = float(args.n_sigma_roi)

    viz_dir: Path | None = None
    if not args.no_viz:
        viz_dir = args.viz_dir
        if viz_dir is None:
            viz_dir = OUT_HISTOGRAM / "near_major_axis_deconv_detect"
        viz_dir = viz_dir.resolve()
        viz_dir.mkdir(parents=True, exist_ok=True)

    all_y0: list[float] = []
    all_A: list[float] = []
    n_frames_ok = 0
    n_frames_no_boundary = 0

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions, boundary = detect_ions(image)
        if boundary is None:
            n_frames_no_boundary += 1
            continue
        n_frames_ok += 1

        if viz_dir is not None:
            n_near = sum(
                1 for ion in ions
                if abs(float(ion["y0"]) - float(boundary[1])) <= tol
            )
            vis_path = viz_dir / f"detect_near_axis_deconv_{idx:04d}.png"
            visualize(
                image,
                ions,
                n_sigma=2.0,
                title=f"[{idx:04d}] {target.name}  |  {n_near}/{len(ions)} near axis (|y-cy|≤{tol:g})",
                output_path=vis_path,
                boundary=boundary,
                show_fit_quality=True,
                show=False,
                near_major_axis_tol=tol,
            )
            print(
                f"[frame {idx:04d}] ions={len(ions)}, |y-cy|≤{tol:g}: {n_near}  -> {vis_path}"
            )

        y0s, As_ = process_frame(
            image, ions, boundary,
            tol=tol,
            n_sigma_roi=n_sigma_roi,
            kernel_radius=int(args.kernel_radius),
            wiener_reg=float(args.wiener_reg),
            a_max_extra=float(args.a_max_extra),
            deconv_roi_pad=float(args.deconv_roi_pad),
            deconv_min_half=float(args.deconv_min_half),
        )
        all_y0.extend(y0s)
        all_A.extend(As_)

    y0_arr = np.asarray(all_y0, dtype=np.float64)
    A_arr = np.asarray(all_A, dtype=np.float64)

    out_png = args.out
    if out_png is None:
        OUT_HISTOGRAM.mkdir(parents=True, exist_ok=True)
        out_png = OUT_HISTOGRAM / "y0_vs_A_near_major_axis.png"
    else:
        out_png = out_png.resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    suptitle = (
        f"y0 vs A (|y0−cy|≤{tol:g} px; deconv kernel exp(−(x²+y²)/2); "
        f"N={y0_arr.size} from {n_frames_ok} frames"
    )
    if n_frames_no_boundary:
        suptitle += f"; {n_frames_no_boundary} skipped (no boundary)"
    fig.suptitle(suptitle, fontsize=10)

    if y0_arr.size > 0:
        ax.scatter(y0_arr, A_arr, s=22, alpha=0.75, c="steelblue", edgecolors="k", linewidths=0.25)
        slope, intercept = np.polyfit(y0_arr, A_arr, 1)
        xs = np.array([float(y0_arr.min()), float(y0_arr.max())])
        ax.plot(xs, slope * xs + intercept, "r-", lw=1.6, label=f"Linear fit: A = {slope:.4g}·y0 + {intercept:.4g}")
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "无有效拟合点", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("识别中心 y0 (pixel)")
    ax.set_ylabel("拟合参数 A (pixel)")
    ax.set_title("A 来自 f(y)=c0/√(A²−y²) 对反卷积 ROI 的拟合")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}  (N={y0_arr.size})")


if __name__ == "__main__":
    main()
