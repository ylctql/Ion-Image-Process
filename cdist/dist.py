from itertools import combinations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from output_paths import OUT_ION_POS, default_cdist_hist_png
import numpy as np
from scipy.spatial.distance import cdist


def normalized_config_distance(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    center_centroids: bool = True,
) -> float:
    """
    Compute normalized lattice distance from config A to config B.

    默认将两套坐标各自减去质心再算距离，使整体平移不改变度量；设 ``center_centroids=False`` 则使用原始像素坐标。
    """
    if coords_a.ndim != 2 or coords_b.ndim != 2 or coords_a.shape[1] != 2 or coords_b.shape[1] != 2:
        raise ValueError("Each configuration must have shape (N, 2).")
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        raise ValueError("Configuration with zero ions is not supported.")

    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if center_centroids:
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)

    dmat = cdist(a, b)
    row_min = dmat.min(axis=1)
    return float(row_min.sum() / row_min.size)


def load_config_files(pos_dir: Path, count: int | None = None) -> list[Path]:
    files = sorted([p for p in pos_dir.iterdir() if p.suffix.lower() == ".npy"])
    if not files:
        raise FileNotFoundError(f"No .npy coordinate files found under: {pos_dir}")
    if count is None:
        return files
    if count <= 1:
        raise ValueError("count must be >= 2 to compute pairwise distances.")
    return files[: min(count, len(files))]


def compute_all_pair_distances(
    config_files: list[Path],
    *,
    center_centroids: bool = True,
) -> list[float]:
    distances = []
    for f_a, f_b in combinations(config_files, 2):
        coords_a = np.asarray(np.load(f_a), dtype=np.float64)
        coords_b = np.asarray(np.load(f_b), dtype=np.float64)
        dist = normalized_config_distance(
            coords_a, coords_b, center_centroids=center_centroids
        )
        distances.append(dist)
    return distances


def compute_all_pair_distances_from_coords(
    coords_list: list[np.ndarray],
    *,
    center_centroids: bool = True,
) -> list[float]:
    """与 ``compute_all_pair_distances`` 相同度量，输入为已加载的 ``(N, 2)`` 坐标数组列表。"""
    distances: list[float] = []
    for a, b in combinations(coords_list, 2):
        distances.append(
            normalized_config_distance(
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
                center_centroids=center_centroids,
            )
        )
    return distances


def plot_histogram(
    distances: list[float],
    config_count: int,
    output_path: Path,
    bins: int = 50,
    show: bool = False,
    *,
    centroid_aligned: bool = True,
) -> None:
    if not distances:
        raise ValueError("No pair distances were computed.")

    arr = np.asarray(distances, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(f"Configuration Distance Histogram (configs={config_count}, pairs={arr.size})")
    ax.set_xlabel(
        "Normalized distance (pixel), centroids aligned"
        if centroid_aligned
        else "Normalized distance (pixel), raw coordinates"
    )
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[saved] histogram: {output_path}")
    print(
        f"[stats] min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise lattice distances with cdist and plot histogram."
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=OUT_ION_POS,
        help="Directory that stores ion center coordinate .npy files (default: outputs/IonPos).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of coordinate files to use (default: all files).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window after saving.",
    )
    parser.add_argument(
        "--no-center-centroids",
        action="store_true",
        help="不在各构型内减质心（恢复对整体平移敏感的旧度量）。",
    )
    args = parser.parse_args()

    pos_dir = args.pos_dir
    if not pos_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {pos_dir}")

    config_files = load_config_files(pos_dir, args.count)
    if len(config_files) < 2:
        raise ValueError("Need at least 2 coordinate files to compute pairwise distances.")

    print(f"[info] using {len(config_files)} files from {pos_dir}")
    distances = compute_all_pair_distances(
        config_files, center_centroids=not args.no_center_centroids
    )
    print(f"[info] computed {len(distances)} pairwise distances")

    output_path = default_cdist_hist_png(len(config_files))
    plot_histogram(
        distances,
        config_count=len(config_files),
        output_path=output_path,
        bins=args.bins,
        show=args.show,
        centroid_aligned=not args.no_center_centroids,
    )


if __name__ == "__main__":
    main()
