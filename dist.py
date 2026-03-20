from itertools import combinations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def normalized_config_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute normalized lattice distance from config A to config B."""
    if coords_a.ndim != 2 or coords_b.ndim != 2 or coords_a.shape[1] != 2 or coords_b.shape[1] != 2:
        raise ValueError("Each configuration must have shape (N, 2).")
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        raise ValueError("Configuration with zero ions is not supported.")

    dmat = cdist(coords_a, coords_b)
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


def compute_all_pair_distances(config_files: list[Path]) -> list[float]:
    distances = []
    for f_a, f_b in combinations(config_files, 2):
        coords_a = np.asarray(np.load(f_a), dtype=np.float64)
        coords_b = np.asarray(np.load(f_b), dtype=np.float64)
        dist = normalized_config_distance(coords_a, coords_b)
        distances.append(dist)
    return distances


def plot_histogram(
    distances: list[float],
    config_count: int,
    output_path: Path,
    bins: int = 50,
    show: bool = False,
) -> None:
    if not distances:
        raise ValueError("No pair distances were computed.")

    arr = np.asarray(distances, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(f"Configuration Distance Histogram (configs={config_count}, pairs={arr.size})")
    ax.set_xlabel("Normalized configuration distance (pixel)")
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
        default=Path(__file__).resolve().parent / "IonPos",
        help="Directory that stores ion center coordinate .npy files.",
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
        "--output",
        type=Path,
        default=None,
        help="Output path of histogram image (default: histogram/cdist_hist_{configs}.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window after saving.",
    )
    args = parser.parse_args()

    pos_dir = args.pos_dir
    if not pos_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {pos_dir}")

    config_files = load_config_files(pos_dir, args.count)
    if len(config_files) < 2:
        raise ValueError("Need at least 2 coordinate files to compute pairwise distances.")

    print(f"[info] using {len(config_files)} files from {pos_dir}")
    distances = compute_all_pair_distances(config_files)
    print(f"[info] computed {len(distances)} pairwise distances")

    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent
            / "histogram"
            / f"cdist_hist_{len(config_files)}.png"
        )

    plot_histogram(
        distances,
        config_count=len(config_files),
        output_path=output_path,
        bins=args.bins,
        show=args.show,
    )


if __name__ == "__main__":
    main()
