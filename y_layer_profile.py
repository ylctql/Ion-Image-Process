"""
沿固定 x 条带对每行求和，得到 y 向 1D 轮廓，用于观察离子晶格在 y 方向的分层/条纹。

- x 向默认取像素列 [400, 600]（半开写法 400:601，与 numpy 一致）。
- ``--y-range`` 支持逗号分隔的多段 numpy 切片并取并集，例如 ``0:10``、``0:10,20:30``。
  省略时默认使用椭圆 boundary 在竖直方向的包络（[cy-b, cy+b] 对应的整数行并裁剪到图像高度）。
- 边界估计与 ``detect_ions`` / ``edge_strip_profile`` 相同：在 ``image - 高斯背景`` 上调用
  ``estimate_crystal_boundary``。

用法示例:
  python y_layer_profile.py 0
  python y_layer_profile.py 0 --y-range 0:10,20:30
  python y_layer_profile.py 0 --x-range 400:601 --preprocess bgsub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_paths import OUT_Y_LAYER_PROFILE, PROJECT_ROOT

from ion_detect.boundary import estimate_crystal_boundary
from ion_detect.cli_helpers import parse_slice_token
from ion_detect.cli_helpers import resolve_indices


def _default_ellipse_y_interval(h: int, boundary: tuple[float, float, float, float]) -> tuple[int, int]:
    """竖直方向包络：与轴对齐椭圆相接的整数行区间 [y_lo, y_hi)（半开）。"""
    _cx, cy, _a, b = (float(boundary[0]), float(boundary[1]), float(boundary[2]), float(boundary[3]))
    y_lo = max(0, int(np.floor(cy - b)))
    y_hi = min(h, int(np.ceil(cy + b)) + 1)
    if y_lo >= y_hi:
        return 0, h
    return y_lo, y_hi


def parse_y_range_union(spec: str | None, h: int, boundary: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """解析 y-range：None/空串 -> 椭圆默认可用行；否则逗号分隔切片，取并集为若干 [lo, hi)。"""
    if spec is None or not str(spec).strip():
        y0, y1 = _default_ellipse_y_interval(h, boundary)
        return [(y0, y1)]

    intervals: list[tuple[int, int]] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        obj = parse_slice_token(token)
        if isinstance(obj, slice):
            rows = np.arange(h)[obj]
            if rows.size == 0:
                continue
            intervals.append((int(rows[0]), int(rows[-1]) + 1))
        else:
            i = int(obj)
            if i < 0:
                i += h
            if 0 <= i < h:
                intervals.append((i, i + 1))

    if not intervals:
        raise ValueError(f"y-range 未选中任何行: {spec!r} (H={h})")
    return _merge_half_open_intervals(intervals)


def _merge_half_open_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    iv = sorted((max(0, lo), max(0, hi)) for lo, hi in intervals if lo < hi)
    if not iv:
        return []
    out: list[tuple[int, int]] = []
    a, b = iv[0]
    for c, d in iv[1:]:
        if c <= b:
            b = max(b, d)
        else:
            out.append((a, b))
            a, b = c, d
    out.append((a, b))
    return out


def row_mask_from_intervals(h: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(h, dtype=bool)
    for lo, hi in intervals:
        lo = max(0, lo)
        hi = min(h, hi)
        if lo < hi:
            m[lo:hi] = True
    return m


def parse_x_slice(spec: str, w: int) -> tuple[int, int]:
    obj = parse_slice_token(spec.strip())
    if not isinstance(obj, slice):
        raise ValueError(f"x-range 须为 start:stop 切片: {spec!r}")
    cols = np.arange(w)[obj]
    if cols.size == 0:
        raise ValueError(f"x-range 未选中任何列: {spec!r} (W={w})")
    return int(cols[0]), int(cols[-1]) + 1


def integrate_along_x_per_row(
    img: np.ndarray,
    x0: int,
    x1: int,
    row_mask: np.ndarray,
) -> np.ndarray:
    """对 row_mask 为 True 的行，在 [x0, x1) 上对 x 求和；其余行为 NaN。"""
    h = int(img.shape[0])
    prof = np.full(h, np.nan, dtype=np.float64)
    if x0 >= x1:
        return prof
    m = np.asarray(row_mask, dtype=bool)
    if not np.any(m):
        return prof
    strip = np.asarray(img, dtype=np.float64)[:, x0:x1]
    prof[m] = np.sum(strip[m, :], axis=1)
    return prof


def _boundary_from_image_like_pipeline(image: np.ndarray, bg_sigma=(10, 30)):
    img = image.astype(np.float64)
    bg = gaussian_filter(img, sigma=bg_sigma)
    signal = img - bg
    return estimate_crystal_boundary(signal), signal


def _strip_map_for_preprocess(image: np.ndarray, signal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return image.astype(np.float64)
    if mode == "bgsub":
        return np.asarray(signal, dtype=np.float64)
    raise ValueError(f"unknown preprocess: {mode}")


def _ascii_title(idx: int, npy_filename: str) -> str:
    stem = Path(npy_filename).stem
    safe = stem.encode("ascii", "replace").decode("ascii")
    return f"frame {idx:04d} ({safe})"


def _warn_noninteractive_backend() -> None:
    be = matplotlib.get_backend().lower()
    if be == "agg" or "inline" in be:
        print(
            "[提示] 当前 Matplotlib 后端为 "
            f"{matplotlib.get_backend()}（无可交互窗口）。若需弹窗查看，请在终端设置 GUI 后端，例如: "
            "$env:MPLBACKEND='TkAgg'; python y_layer_profile.py 0 --show"
        )


def plot_y_layer_dashboard(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    profile: np.ndarray,
    x0: int,
    x1: int,
    y_intervals: list[tuple[int, int]],
    out_path: Path | None,
    title: str,
    *,
    preprocess: str,
    y_range_desc: str,
    show: bool,
) -> None:
    cx, cy, a, b = (float(boundary[i]) for i in range(4))
    h, w = image.shape
    lo = float(np.percentile(image, 1))
    hi = float(np.percentile(image, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    ax0.imshow(image, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    ax0.add_patch(
        Rectangle(
            (x0, 0), x1 - x0, h,
            linewidth=0, edgecolor="none", facecolor="cyan", alpha=0.12,
        )
    )
    h_img = int(image.shape[0])
    for y0b, y1b in y_intervals:
        y0b = max(0, min(h_img, int(y0b)))
        y1b = max(0, min(h_img, int(y1b)))
        if y0b < y1b:
            ax0.add_patch(
                Rectangle(
                    (x0, y0b), x1 - x0, y1b - y0b,
                    linewidth=0, edgecolor="none", facecolor="orange", alpha=0.18,
                )
            )
    ell = Ellipse(
        xy=(cx, cy), width=2 * a, height=2 * b, angle=0,
        edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--", alpha=0.85,
    )
    ax0.add_patch(ell)
    ax0.plot([x0, x1, x1, x0, x0], [0, 0, h - 1, h - 1, 0], color="lime", linewidth=1.2, alpha=0.9)
    ax0.set_xlim(0, w - 1)
    ax0.set_ylim(h - 1, 0)
    ax0.set_title(
        f"{title}\npreprocess={preprocess}, x=[{x0}, {x1}), y-range={y_range_desc}\n"
        "cyan dashed = ellipse boundary; lime box = x strip; orange = y-range ∩ strip"
    )
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")

    y_axis = np.arange(h, dtype=np.float64)
    ax1.plot(y_axis, profile, color="tab:blue", linewidth=1.0, drawstyle="default")
    ax1.axvline(cy - b, color="gray", linestyle=":", linewidth=1.0, alpha=0.8, label="cy-b")
    ax1.axvline(cy + b, color="gray", linestyle=":", linewidth=1.0, alpha=0.8, label="cy+b")
    ax1.set_xlabel("y (row px)")
    ax1.set_ylabel("sum over x strip")
    ax1.set_title("Row-wise sum within x strip (NaN outside y-range)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(0, h - 1)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"[已保存] {out_path}")
    if not show:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="y 向分层：x 条带内按行积分并绘图")
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引（numpy 风格，与 ion_detect 相同）",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "20260305_1727", help="npy 目录")
    parser.add_argument("--out", type=Path, default=OUT_Y_LAYER_PROFILE, help="输出 PNG 目录")
    parser.add_argument(
        "--x-range",
        type=str,
        default="400:601",
        help="x 向半开切片（默认 400:601 即列 400..600）",
    )
    parser.add_argument(
        "--y-range",
        type=str,
        default="",
        help='y 向半开切片，逗号分隔多段并集，如 "0:10" 或 "0:10,20:30"；默认椭圆 boundary 竖直包络',
    )
    parser.add_argument(
        "--preprocess",
        choices=("raw", "bgsub"),
        default="raw",
        help="积分所用二维图：raw 或 高斯背景减除",
    )
    parser.add_argument("--show", action="store_true", help="显示交互窗口")
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录 {data_dir} 下没有 npy 文件")
    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有可处理的索引")

    y_range_opt = args.y_range.strip() or None

    if args.show:
        plt.ioff()
        _warn_noninteractive_backend()

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        boundary, signal = _boundary_from_image_like_pipeline(image)
        if boundary is None:
            print(f"[{idx:04d}] {target.name}: 无法估计 boundary，跳过。")
            continue

        h, w = image.shape[:2]
        try:
            x0, x1 = parse_x_slice(args.x_range, w)
        except ValueError as e:
            print(f"[{idx:04d}] {target.name}: {e}")
            continue

        try:
            intervals = parse_y_range_union(y_range_opt, h, boundary)
        except ValueError as e:
            print(f"[{idx:04d}] {target.name}: {e}")
            continue
        row_mask = row_mask_from_intervals(h, intervals)
        if not np.any(row_mask):
            print(f"[{idx:04d}] {target.name}: y-range 为空，跳过。")
            continue

        if y_range_opt is None:
            y_range_desc = f"ellipse [{intervals[0][0]}, {intervals[0][1]})"
        else:
            y_range_desc = y_range_opt

        strip_map = _strip_map_for_preprocess(image, signal, args.preprocess)
        profile = integrate_along_x_per_row(strip_map, x0, x1, row_mask)

        print(
            f"\n[{idx:04d}] {target.name}  boundary=({boundary[0]:.1f}, {boundary[1]:.1f}, "
            f"a={boundary[2]:.1f}, b={boundary[3]:.1f})  x=[{x0},{x1})  y intervals={intervals}"
        )

        out_png = args.out / f"y_layer_profile_{idx:04d}.png"
        plot_y_layer_dashboard(
            image,
            boundary,
            profile,
            x0,
            x1,
            intervals,
            out_png,
            _ascii_title(idx, target.name),
            preprocess=args.preprocess,
            y_range_desc=y_range_desc,
            show=args.show,
        )

    if args.show:
        plt.show(block=True)


if __name__ == "__main__":
    main()
