"""
统计并可视化 20260305_1727 目录中的 npy 矩阵数据。

目录包含 997 个 (171, 1024) float32 矩阵，
文件名格式: YYYYMMDD_HHMMSS.npy（表示采集时间戳）。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "20260305_1727"
OUTPUT_DIR = PROJECT_ROOT / "visualization_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_timestamp(filename: str) -> datetime:
    stem = Path(filename).stem
    return datetime.strptime(stem, "%Y%m%d_%H%M%S")


def load_all_metadata():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
    timestamps = [parse_timestamp(f) for f in files]
    paths = [DATA_DIR / f for f in files]
    return files, timestamps, paths


def print_statistics(files, timestamps, paths):
    print("=" * 60)
    print(f"  目录: {DATA_DIR}")
    print(f"  文件总数: {len(files)}")
    print(f"  文件类型: 全部为 .npy")
    print("=" * 60)

    sample = np.load(paths[0])
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"  矩阵形状: {sample.shape}")
    print(f"  数据类型: {sample.dtype}")
    print(f"  单文件大小: {paths[0].stat().st_size / 1024:.1f} KB")
    print(f"  总数据量: {total_bytes / (1024**2):.1f} MB")
    print(f"  时间范围: {timestamps[0]} → {timestamps[-1]}")
    duration = timestamps[-1] - timestamps[0]
    print(f"  总时长: {duration}")
    if len(timestamps) > 1:
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds()
                      for i in range(len(timestamps) - 1)]
        print(f"  平均采集间隔: {np.mean(intervals):.1f} 秒")
        print(f"  间隔范围: {np.min(intervals):.1f} ~ {np.max(intervals):.1f} 秒")

    global_min, global_max, global_mean = np.inf, -np.inf, 0.0
    for p in paths:
        arr = np.load(p)
        global_min = min(global_min, arr.min())
        global_max = max(global_max, arr.max())
        global_mean += arr.mean()
    global_mean /= len(paths)
    print(f"  全局最小值: {global_min:.2f}")
    print(f"  全局最大值: {global_max:.2f}")
    print(f"  全局均值: {global_mean:.2f}")
    print("=" * 60)
    return global_min, global_max


def plot_sample_heatmaps(files, timestamps, paths, vmin, vmax):
    """选取均匀分布的 9 张图片做热力图展示"""
    n = len(files)
    indices = np.linspace(0, n - 1, 9, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle("Sample Heatmaps (9 evenly spaced frames)", fontsize=16, y=0.98)

    norm = Normalize(vmin=vmin, vmax=vmax)
    for ax, idx in zip(axes.flat, indices):
        data = np.load(paths[idx])
        im = ax.imshow(data, aspect="auto", cmap="viridis", norm=norm)
        ts = timestamps[idx].strftime("%H:%M:%S")
        ax.set_title(f"#{idx}  {ts}", fontsize=10)
        ax.set_xlabel("Pixel (1024)")
        ax.set_ylabel("Row (171)")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Intensity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "sample_heatmaps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")


def plot_mean_image(paths):
    """计算所有帧的均值图像"""
    acc = np.zeros((171, 1024), dtype=np.float64)
    for p in paths:
        acc += np.load(p).astype(np.float64)
    mean_img = (acc / len(paths)).astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mean_img, aspect="auto", cmap="inferno")
    ax.set_title(f"Mean Image (averaged over {len(paths)} frames)", fontsize=14)
    ax.set_xlabel("Pixel (1024)")
    ax.set_ylabel("Row (171)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Intensity")
    fig.tight_layout()
    out = OUTPUT_DIR / "mean_image.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")
    return mean_img


def plot_std_image(paths, mean_img):
    """计算所有帧的标准差图像，反映时间波动"""
    acc_sq = np.zeros((171, 1024), dtype=np.float64)
    for p in paths:
        diff = np.load(p).astype(np.float64) - mean_img
        acc_sq += diff ** 2
    std_img = np.sqrt(acc_sq / len(paths)).astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(std_img, aspect="auto", cmap="hot")
    ax.set_title(f"Std Deviation Image ({len(paths)} frames)", fontsize=14)
    ax.set_xlabel("Pixel (1024)")
    ax.set_ylabel("Row (171)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Std Dev")
    fig.tight_layout()
    out = OUTPUT_DIR / "std_image.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")


def plot_temporal_evolution(timestamps, paths):
    """绘制全帧均值 & 最大值的时间演化曲线"""
    means, maxs, mins = [], [], []
    for p in paths:
        arr = np.load(p)
        means.append(arr.mean())
        maxs.append(arr.max())
        mins.append(arr.min())

    t_sec = [(ts - timestamps[0]).total_seconds() / 60.0 for ts in timestamps]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(t_sec, means, color="steelblue", linewidth=0.8, label="Mean")
    ax1.fill_between(t_sec, mins, maxs, alpha=0.15, color="steelblue", label="Min–Max range")
    ax1.set_xlabel("Time (minutes since start)", fontsize=12)
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_title("Temporal Evolution of Frame Statistics", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "temporal_evolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")


def plot_row_column_profiles(paths, timestamps):
    """绘制平均行/列剖面（用首/中/末帧对比）"""
    indices = [0, len(paths) // 2, len(paths) - 1]
    labels = ["First", "Middle", "Last"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for idx, label in zip(indices, labels):
        data = np.load(paths[idx])
        ts = timestamps[idx].strftime("%H:%M:%S")
        ax1.plot(data.mean(axis=0), linewidth=0.8, label=f"{label} ({ts})")
        ax2.plot(data.mean(axis=1), linewidth=0.8, label=f"{label} ({ts})")

    ax1.set_title("Column-wise Mean Profile", fontsize=13)
    ax1.set_xlabel("Column index (0–1023)")
    ax1.set_ylabel("Mean intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Row-wise Mean Profile", fontsize=13)
    ax2.set_xlabel("Row index (0–170)")
    ax2.set_ylabel("Mean intensity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "row_column_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")


def plot_histogram(paths):
    """对首/中/末帧绘制像素值直方图"""
    indices = [0, len(paths) // 2, len(paths) - 1]
    labels = ["First", "Middle", "Last"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in zip(indices, labels):
        data = np.load(paths[idx]).ravel()
        ax.hist(data, bins=200, alpha=0.5, label=label, density=True)
    ax.set_title("Pixel Intensity Distribution", fontsize=14)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "intensity_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[已保存] {out}")


if __name__ == "__main__":
    print("正在加载文件列表...")
    files, timestamps, paths = load_all_metadata()
    vmin, vmax = print_statistics(files, timestamps, paths)

    print("\n正在生成可视化图表...")
    plot_sample_heatmaps(files, timestamps, paths, vmin, vmax)
    mean_img = plot_mean_image(paths)
    plot_std_image(paths, mean_img)
    plot_temporal_evolution(timestamps, paths)
    plot_row_column_profiles(paths, timestamps)
    plot_histogram(paths)

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
    print("完成！")
