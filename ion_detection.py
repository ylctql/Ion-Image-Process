"""
离子晶格图像中单离子的检测与椭圆拟合。

算法流程:
  1. 各向异性高斯平滑估计背景并减去
  2. 匹配滤波器增强离子信号, 抑制噪声
  3. 各向异性局部极大值检测定位离子候选中心
  4. 对每个候选用 2D 旋转高斯进行最小二乘拟合
  5. 质量过滤, 输出中心坐标与椭圆参数
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, binary_erosion
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import time

# ──────────────────────────── 2D Gaussian Model ────────────────────────────

def _gauss2d(coords, amp, x0, y0, sx, sy, theta, offset):
    """旋转二维高斯函数。
    sx: 沿旋转角 theta 方向的 sigma
    sy: 沿垂直于 theta 方向的 sigma
    """
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return offset + amp * np.exp(-0.5 * (xr**2 / sx**2 + yr**2 / sy**2))

# ──────────────────────────── Core Detection ───────────────────────────────

def _build_matched_kernel(sigma_x=1.2, sigma_y=1.8, half_size=3):
    """构造与离子 PSF 匹配的高斯核，用于增强信噪比。"""
    ky, kx = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    kernel = np.exp(-0.5 * (kx**2 / sigma_x**2 + ky**2 / sigma_y**2))
    kernel /= kernel.sum()
    return kernel


def detect_ions(image, bg_sigma=(10, 30), peak_size=(5, 9),
                rel_threshold=0.025, fit_hw=(3, 4),
                sigma_range=(0.3, 3.5), use_matched_filter=True,
                refine=True):
    """
    检测离子并拟合椭圆参数。

    Parameters
    ----------
    image : ndarray (H, W), float
    bg_sigma : float or tuple – 背景估计的高斯 sigma (y, x)
    peak_size : int or tuple  – 局部极大值检测窗口尺寸 (y, x)
    rel_threshold : float – 信号阈值 (相对于最大信号)
    fit_hw : int or (int, int) – 高斯拟合的半窗口 (hw_y, hw_x)
    sigma_range : tuple – 允许的 sigma 范围 (min, max)
    use_matched_filter : bool – 是否用匹配滤波器增强峰检测
    refine : bool – 是否启用两阶段精修 (计数模式开启; 形变测量可关闭)

    Returns
    -------
    ions : list[dict]  每个离子的参数:
        x0, y0       – 中心坐标 (像素, 亚像素精度)
        sigma_minor  – 短轴 sigma (像素)
        sigma_major  – 长轴 sigma (像素)
        theta_deg    – 长轴相对 x 轴的旋转角 (度)
        amplitude    – 高斯峰值强度
    """
    img = image.astype(np.float64)
    h, w = img.shape
    s_lo, s_hi = sigma_range

    if isinstance(fit_hw, (list, tuple)):
        hw_y, hw_x = fit_hw
    else:
        hw_y = hw_x = fit_hw

    # 各向异性背景减除
    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg

    # 匹配滤波器：与离子 PSF 卷积提高信噪比
    if use_matched_filter:
        kernel = _build_matched_kernel()
        detect_map = fftconvolve(signal, kernel, mode="same")
    else:
        detect_map = signal

    # 各向异性局部极大值检测
    local_max = maximum_filter(detect_map, size=peak_size)
    thresh = rel_threshold * detect_map.max()
    peak_mask = (detect_map == local_max) & (detect_map > thresh)
    peak_yx = np.argwhere(peak_mask)

    # 在拟合前用晶格边界椭圆过滤噪声候选, 节省拟合时间
    boundary = _estimate_crystal_boundary(signal)
    if boundary is not None:
        peak_yx = _apply_boundary_filter(peak_yx, *boundary)

    ions = _fit_all_peaks(img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
                          h, w, refine=refine)
    ions.sort(key=lambda d: (d["y0"], d["x0"]))
    return ions, boundary


def _estimate_crystal_boundary(signal, smooth_sigma=8,
                               mask_threshold_pct=0.15, margin=1.05):
    """从信号图估计离子晶格的椭圆边界。

    对背景减除后的信号做平滑, 取 15% 阈值确定晶格区域,
    从区域边界拟合椭圆, 加 5% 余量作为候选过滤边界。

    Returns
    -------
    (cx, cy, a, b) 或 None
        cx, cy – 椭圆中心 (像素)
        a      – x 方向半轴
        b      – y 方向半轴
    """
    smoothed = gaussian_filter(np.clip(signal, 0, None), sigma=smooth_sigma)
    mask = smoothed > mask_threshold_pct * smoothed.max()
    boundary = mask & ~binary_erosion(mask, iterations=2)
    by, bx = np.where(boundary)

    if len(bx) < 10:
        return None

    cx = (bx.max() + bx.min()) / 2.0
    cy = (by.max() + by.min()) / 2.0
    a = (bx.max() - bx.min()) / 2.0 * margin
    b = (by.max() - by.min()) / 2.0 * margin
    return cx, cy, a, b


def _apply_boundary_filter(peak_yx, cx, cy, a, b):
    """保留椭圆边界内的候选峰, 移除外部噪声。"""
    px = peak_yx[:, 1].astype(np.float64)
    py = peak_yx[:, 0].astype(np.float64)
    inside = ((px - cx) / a)**2 + ((py - cy) / b)**2 <= 1.0
    return peak_yx[inside]


def _fit_all_peaks(img, signal, peak_yx, hw_y, hw_x, s_lo, s_hi,
                   h, w, refine=True):
    """对所有候选峰做高斯拟合, 可选两阶段精修。"""
    # 权重模板: y 方向更锐利 (0.45), 抑制紧邻离子
    wy, wx = 2 * hw_y + 1, 2 * hw_x + 1
    gy_t, gx_t = np.mgrid[0:wy, 0:wx]
    cy_t, cx_t = hw_y, hw_x
    weight_template = np.exp(-0.5 * ((gx_t - cx_t)**2 / (hw_x * 0.7)**2
                                   + (gy_t - cy_t)**2 / (hw_y * 0.45)**2))

    ions = _do_fit_pass(img, peak_yx, hw_y, hw_x, s_lo, s_hi,
                        h, w, weight_template, sigma_init=(1.2, 1.8))

    if not refine or len(ions) < 20:
        return ions

    # ── 两阶段精修 ──
    # 从稳定拟合中计算参考 sigma
    majors = np.array([d["sigma_major"] for d in ions])
    minors = np.array([d["sigma_minor"] for d in ions])
    stable = (majors < s_hi * 0.9) & (minors < s_hi * 0.9)
    if stable.sum() < 10:
        return ions

    ref_minor = float(np.median(minors[stable]))
    ref_major = float(np.median(majors[stable]))

    # 标记需要重新拟合的离子 (sigma 偏离中位值太远)
    needs_refit = (majors > ref_major * 1.8) | (minors > ref_minor * 2.0)
    if needs_refit.sum() == 0:
        return ions

    # 对需重新拟合的离子, 用参考 sigma 约束上界
    refit_indices = np.where(needs_refit)[0]
    refit_yx = np.array([[ions[i]["_py"], ions[i]["_px"]] for i in refit_indices])

    tight_s_hi_minor = ref_minor * 1.6
    tight_s_hi_major = ref_major * 1.6

    refitted = _do_fit_pass(img, refit_yx, hw_y, hw_x, s_lo,
                            max(tight_s_hi_minor, tight_s_hi_major),
                            h, w, weight_template,
                            sigma_init=(ref_minor, ref_major))

    # 用重拟合结果替换
    refit_map = {}
    for ion in refitted:
        key = (ion["_py"], ion["_px"])
        refit_map[key] = ion

    result = []
    for i, ion in enumerate(ions):
        if needs_refit[i]:
            key = (ion["_py"], ion["_px"])
            if key in refit_map:
                result.append(refit_map[key])
            else:
                result.append(ion)
        else:
            result.append(ion)

    return result


def _do_fit_pass(img, peak_yx, hw_y, hw_x, s_lo, s_hi,
                 h, w, weight_template, sigma_init=(1.2, 1.8)):
    """单次高斯拟合遍历。"""
    ions = []
    si_x, si_y = sigma_init

    for py, px in peak_yx:
        py, px = int(py), int(px)
        y1, y2 = max(0, py - hw_y), min(h, py + hw_y + 1)
        x1, x2 = max(0, px - hw_x), min(w, px + hw_x + 1)
        patch = img[y1:y2, x1:x2]
        ph, pw_ = patch.shape

        yy, xx = np.mgrid[0:ph, 0:pw_]
        ly, lx = py - y1, px - x1
        amp0 = float(patch[ly, lx] - patch.min())
        if amp0 < 1:
            continue

        wt_y1 = hw_y - ly
        wt_x1 = hw_x - lx
        weights = weight_template[wt_y1:wt_y1 + ph, wt_x1:wt_x1 + pw_]
        fit_sigma = 1.0 / np.sqrt(np.clip(weights, 0.01, None))

        try:
            p0 = [amp0, lx, ly, si_x, si_y, 0.0, float(patch.min())]
            lo = [0,       0,       0,       s_lo, s_lo, -np.pi, 0]
            hi = [amp0*4,  pw_ - 1, ph - 1,  s_hi, s_hi,  np.pi, float(patch.max())]

            popt, _ = curve_fit(
                _gauss2d, (xx.ravel(), yy.ravel()), patch.ravel(),
                p0=p0, bounds=(lo, hi), maxfev=2000,
                sigma=fit_sigma.ravel(),
            )
        except (RuntimeError, ValueError):
            continue

        amp, fx, fy, sx, sy, theta, offset = popt

        if sx > sy:
            sx, sy = sy, sx
            theta += np.pi / 2

        theta_deg = np.degrees(theta)
        theta_deg = ((theta_deg + 90) % 180) - 90

        gx, gy = x1 + fx, y1 + fy

        if abs(gx - px) > hw_x or abs(gy - py) > hw_y:
            continue
        if not (s_lo <= sx <= s_hi and s_lo <= sy <= s_hi):
            continue

        ions.append({
            "x0": gx, "y0": gy,
            "sigma_minor": sx, "sigma_major": sy,
            "theta_deg": theta_deg,
            "amplitude": amp,
            "_py": py, "_px": px,
        })

    return ions

# ──────────────────────────── Visualization ────────────────────────────────

def visualize(image, ions, n_sigma=2.0, title="", output_path=None,
              show_zoom=True, zoom_center=None, zoom_size=80,
              boundary=None):
    """
    在图像上标注拟合椭圆。

    n_sigma 控制椭圆的半轴长度 (n_sigma * sigma)。
    boundary: (cx, cy, a, b) 晶格边界椭圆参数, 若提供则绘制在图上。
    可选显示一个局部放大区域。
    """
    nrows = 6 if show_zoom else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(20, 5 + (3 * 5 if show_zoom else 0)),
                             gridspec_kw={"height_ratios": [5, 2, 3, 3, 3, 2]
                                          if show_zoom else [1]})
    if not show_zoom:
        axes = [axes]

    # ── 全局图 ──
    ax = axes[0]
    ax.imshow(image, cmap="gray", aspect="auto", vmin=np.percentile(image, 1),
              vmax=np.percentile(image, 99.5))
    for ion in ions:
        ell = Ellipse(
            xy=(ion["x0"], ion["y0"]),
            width=2 * n_sigma * ion["sigma_minor"],
            height=2 * n_sigma * ion["sigma_major"],
            angle=ion["theta_deg"],
            edgecolor="red", facecolor="none", linewidth=0.4, alpha=0.8,
        )
        ax.add_patch(ell)
    if boundary is not None:
        bcx, bcy, ba, bb = boundary
        bnd_ell = Ellipse(
            xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
            edgecolor="cyan", facecolor="none",
            linewidth=1.2, linestyle="--", alpha=0.9,
        )
        ax.add_patch(bnd_ell)
    ax.set_title(f"{title}   [{len(ions)} ions, ellipse = {n_sigma}σ]", fontsize=13)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # ── 放大图 (5 个区域: 上边缘, 左, 中心, 右, 下边缘) ──
    if show_zoom:
        regions = [
            ("Top edge",    (500, 35),  100, 20),
            ("Left",        (200, 75),   60, 30),
            ("Center",      (500, 85),   60, 30),
            ("Right",       (800, 75),   60, 30),
            ("Bottom edge", (500, 130), 100, 20),
        ]
        for i, (label, (rcx, rcy), rzs_x, rzs_y) in enumerate(regions):
            ax2 = axes[1 + i]
            x1z = max(0, rcx - rzs_x)
            x2z = min(image.shape[1], rcx + rzs_x)
            y1z = max(0, rcy - rzs_y)
            y2z = min(image.shape[0], rcy + rzs_y)

            ax2.imshow(image, cmap="gray", aspect="equal",
                       vmin=np.percentile(image, 1),
                       vmax=np.percentile(image, 99.5))
            for ion in ions:
                if x1z <= ion["x0"] <= x2z and y1z <= ion["y0"] <= y2z:
                    ell = Ellipse(
                        xy=(ion["x0"], ion["y0"]),
                        width=2 * n_sigma * ion["sigma_minor"],
                        height=2 * n_sigma * ion["sigma_major"],
                        angle=ion["theta_deg"],
                        edgecolor="lime", facecolor="none", linewidth=1.2,
                    )
                    ax2.add_patch(ell)
                    ax2.plot(ion["x0"], ion["y0"], "r.", markersize=2)
            if boundary is not None:
                bcx, bcy, ba, bb = boundary
                bnd_ell = Ellipse(
                    xy=(bcx, bcy), width=2 * ba, height=2 * bb, angle=0,
                    edgecolor="cyan", facecolor="none",
                    linewidth=1.5, linestyle="--", alpha=0.9,
                )
                ax2.add_patch(bnd_ell)
            ax2.set_xlim(x1z, x2z)
            ax2.set_ylim(y2z, y1z)
            ax2.set_title(f"Zoom: {label}  x∈[{x1z},{x2z}]  y∈[{y1z},{y2z}]",
                          fontsize=11)
            ax2.set_xlabel("x (pixel)")
            ax2.set_ylabel("y (pixel)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[已保存] {output_path}")
    plt.close(fig)

# ──────────────────────────── Statistics ────────────────────────────────────

def print_summary(ions):
    if not ions:
        print("未检测到离子。")
        return
    minors = np.array([d["sigma_minor"] for d in ions])
    majors = np.array([d["sigma_major"] for d in ions])
    amps   = np.array([d["amplitude"]   for d in ions])
    ratios = majors / minors

    print(f"\n检测结果: {len(ions)} 个离子")
    print(f"  σ_minor  (短轴): mean={minors.mean():.2f} ± {minors.std():.2f}  "
          f"range=[{minors.min():.2f}, {minors.max():.2f}]")
    print(f"  σ_major  (长轴): mean={majors.mean():.2f} ± {majors.std():.2f}  "
          f"range=[{majors.min():.2f}, {majors.max():.2f}]")
    print(f"  长短轴比 (major/minor): mean={ratios.mean():.2f} ± {ratios.std():.2f}")
    print(f"  振幅: mean={amps.mean():.1f} ± {amps.std():.1f}  "
          f"range=[{amps.min():.1f}, {amps.max():.1f}]")

    xs = np.array([d["x0"] for d in ions])
    ys = np.array([d["y0"] for d in ions])
    print(f"  中心范围:  x ∈ [{xs.min():.1f}, {xs.max():.1f}],  "
          f"y ∈ [{ys.min():.1f}, {ys.max():.1f}]")


def _parse_slice_token(token):
    token = token.strip()
    if not token:
        raise ValueError("Empty index token.")

    if ":" not in token:
        return int(token)

    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice token: {token}")

    def _to_int_or_none(s):
        s = s.strip()
        return None if s == "" else int(s)

    start = _to_int_or_none(parts[0]) if len(parts) >= 1 else None
    stop = _to_int_or_none(parts[1]) if len(parts) >= 2 else None
    step = _to_int_or_none(parts[2]) if len(parts) >= 3 else None

    if step == 0:
        raise ValueError(f"Slice step cannot be 0: {token}")
    return slice(start, stop, step)


def _resolve_indices(spec_args, total):
    """解析索引规范，支持并集，如 ['::3,0:10', '25', '-1']。"""
    if not spec_args:
        spec_args = ["0"]

    selected = set()
    all_indices = np.arange(total)

    for raw in spec_args:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue

            obj = _parse_slice_token(token)
            if isinstance(obj, slice):
                selected.update(all_indices[obj].tolist())
            else:
                idx = obj
                if idx < 0:
                    idx += total
                if 0 <= idx < total:
                    selected.add(idx)
                else:
                    print(f"[skip] index {obj} out of range [0, {total - 1}]")

    return sorted(selected)

# ──────────────────────────── Main ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

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
        help="离子中心保存目录，默认是项目根目录下的 IonPos。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "20260305_1727"
    out_dir  = project_root / "visualization_output"
    out_dir.mkdir(exist_ok=True)
    pos_dir = args.pos_dir or (project_root / "IonPos")
    if args.save_pos:
        pos_dir.mkdir(exist_ok=True)

    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    print(f"共找到 {len(files)} 个 npy 文件")
    selected_indices = _resolve_indices(args.indices, len(files))
    if not selected_indices:
        raise ValueError("没有可处理的索引。请检查输入的 indices。")

    for idx in selected_indices:
        target = files[idx]
        print(f"\n[{idx:04d}] 加载: {target.name}")
        image = np.load(target)

        print("正在检测离子...")
        t0 = time.perf_counter()
        ions, boundary = detect_ions(image)
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
