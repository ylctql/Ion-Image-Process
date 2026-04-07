"""
对多帧离子图调用 :func:`ion_detect.pipeline.detect_ions`，对每帧所有检出离子的
``_sigma_x`` / ``_sigma_y``（轴对齐拟合下的 x/y 方向 σ，像素）分别绘制直方图。

输出：
- ``outputs/histogram/sx/{frame_index:04d}.png``
- ``outputs/histogram/sy/{frame_index:04d}.png``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.cli_helpers import resolve_indices
from ion_detect.pipeline import detect_ions
from output_paths import DEFAULT_DATA_DIR, OUT_HISTOGRAM


def _hist_one(
    arr: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int,
) -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    fig.suptitle(title, fontsize=11)
    b = max(5, int(bins))

    if arr.size > 0:
        ax.hist(arr, bins=b, color="steelblue", edgecolor="black", linewidth=0.35, alpha=0.85)
        mu = float(np.mean(arr))
        ax.axvline(mu, color="darkviolet", ls="-", lw=1.5, label=f"mean μ={mu:.4f}")
        ax.axvline(
            float(np.median(arr)), color="crimson", ls="--", lw=1.2,
            label=f"median={np.median(arr):.4f}",
        )
        if arr.size > 1:
            sd_txt = f"{float(np.std(arr, ddof=1)):.4f}"
        else:
            sd_txt = "—"
        st = (
            f"mean μ = {mu:.4f}\n"
            f"std s = {sd_txt}\n"
            f"median = {float(np.median(arr)):.4f}\n"
            f"N = {arr.size}"
        )
        ax.text(
            0.98, 0.97, st, transform=ax.transAxes, va="top", ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.62),
        )
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "N = 0 (no valid σ)", transform=ax.transAxes,
            ha="center", va="center", fontsize=11,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="每帧 detect_ions 后 σx/σy 直方图 → histogram/sx、histogram/sy",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        default=["0"],
        help="帧索引（与 python -m ion_detect 相同，支持切片并集）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f".npy 数据目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="直方图 bin 数量",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_HISTOGRAM,
        help=f"sx/sy 子目录的父路径 (默认: {OUT_HISTOGRAM})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"目录中无 .npy: {data_dir}")

    selected = resolve_indices(args.indices, len(files))
    if not selected:
        raise SystemExit("没有有效索引")

    out_root = args.out_root.resolve()
    dir_sx = out_root / "sx"
    dir_sy = out_root / "sy"

    for idx in selected:
        target = files[idx]
        image = np.load(target)
        ions, _boundary = detect_ions(image)
        sx_list: list[float] = []
        sy_list: list[float] = []
        for ion in ions:
            if "_sigma_x" not in ion or "_sigma_y" not in ion:
                continue
            sx_list.append(float(ion["_sigma_x"]))
            sy_list.append(float(ion["_sigma_y"]))
        sx = np.asarray(sx_list, dtype=np.float64)
        sy = np.asarray(sy_list, dtype=np.float64)

        stem = f"{idx:04d}"
        name = target.name
        title_x = f"[{stem}] {name}  —  sigma_x (detect_ions), N={sx.size}"
        title_y = f"[{stem}] {name}  —  sigma_y (detect_ions), N={sy.size}"
        path_x = dir_sx / f"{stem}.png"
        path_y = dir_sy / f"{stem}.png"
        _hist_one(sx, title_x, "sigma_x (pixel)", path_x, args.bins)
        _hist_one(sy, title_y, "sigma_y (pixel)", path_y, args.bins)
        print(f"[{stem}] ions={len(ions)}, with σx/σy={sx.size} -> {path_x} , {path_y}")


if __name__ == "__main__":
    main()
