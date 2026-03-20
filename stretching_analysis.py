"""
Analyze ion stretching along y direction for first 100 frames.

Method:
  1. Detect ions in each frame with `detect_ions`.
  2. Use crystal center y (from boundary ellipse of each frame) as origin.
  3. Fit ratio = sigma_minor / sigma_major vs y_rel with even polynomial:
         ratio(y) = a0 + a2*y^2 + a4*y^4
  4. Plot scatter + fit + binned statistics.
"""

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

from ion_detection import detect_ions


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "20260305_1727"
OUTPUT_DIR = PROJECT_ROOT / "visualization_output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_FRAMES = 100
OUT_FIG = OUTPUT_DIR / "stretching_analysis.png"

# 形变测量模式: 放宽 sigma 上限、增大拟合窗口、关闭两阶段精修,
# 避免边缘离子的 sigma_major 被截断导致 ratio 失真。
DETECT_KWARGS = dict(
    sigma_range=(0.3, 8.0),   # 允许 sigma_major 达 8 px (默认 3.5)
    fit_hw=(5, 5),             # 11x11 窗口覆盖拉伸离子全貌 (默认 7x9)
    refine=False,              # 跳过二次精修中的 sigma 收紧
)


def even_poly_fit(y_rel: np.ndarray, ratio: np.ndarray):
    """Least-squares fit with even polynomial terms [1, y^2, y^4]."""
    y2 = y_rel ** 2
    X = np.column_stack([np.ones_like(y_rel), y2, y2 ** 2])
    coef, *_ = np.linalg.lstsq(X, ratio, rcond=None)
    return coef  # a0, a2, a4


def even_poly_eval(y: np.ndarray, coef: np.ndarray):
    y2 = y ** 2
    return coef[0] + coef[1] * y2 + coef[2] * y2 ** 2


def bin_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 60):
    """Compute mean/std/count of y grouped by x bins."""
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.digitize(x, edges) - 1
    centers, means, stds, counts = [], [], [], []
    for i in range(n_bins):
        m = idx == i
        if m.sum() < 10:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2.0)
        means.append(y[m].mean())
        stds.append(y[m].std())
        counts.append(m.sum())
    return np.asarray(centers), np.asarray(means), np.asarray(stds), np.asarray(counts)


def main():
    files = sorted(DATA_DIR.glob("*.npy"))[:N_FRAMES]
    if not files:
        raise RuntimeError(f"No .npy files found in {DATA_DIR}")

    all_y_rel = []
    all_ratio = []
    all_theta = []
    cy_values = []

    t0 = time.perf_counter()
    print(f"Processing {len(files)} frames from {DATA_DIR} ...")
    for i, f in enumerate(files, start=1):
        image = np.load(f)
        ions, boundary = detect_ions(image, **DETECT_KWARGS)

        if boundary is not None:
            _, cy_ref, _, _ = boundary
        else:
            cy_ref = image.shape[0] / 2.0
        cy_values.append(cy_ref)

        y0 = np.asarray([d["y0"] for d in ions], dtype=np.float64)
        minor = np.asarray([d["sigma_minor"] for d in ions], dtype=np.float64)
        major = np.asarray([d["sigma_major"] for d in ions], dtype=np.float64)
        theta = np.asarray([d["theta_deg"] for d in ions], dtype=np.float64)

        ratio = minor / np.maximum(major, 1e-8)
        y_rel = y0 - cy_ref

        all_y_rel.append(y_rel)
        all_ratio.append(ratio)
        all_theta.append(theta)

        if i == 1 or i % 10 == 0 or i == len(files):
            elapsed = time.perf_counter() - t0
            print(
                f"[{i:03d}/{len(files)}] {f.name} | ions={len(ions)} | "
                f"cy={cy_ref:.3f} | elapsed={elapsed:.1f}s"
            )

    cy_values = np.asarray(cy_values)
    print(f"\nCrystal center y stats: mean={cy_values.mean():.3f}, "
          f"std={cy_values.std():.3f}, range=[{cy_values.min():.3f}, {cy_values.max():.3f}]")

    y_rel = np.concatenate(all_y_rel)
    ratio = np.concatenate(all_ratio)
    theta = np.concatenate(all_theta)

    coef = even_poly_fit(y_rel, ratio)
    y_grid = np.linspace(y_rel.min(), y_rel.max(), 800)
    ratio_fit = even_poly_eval(y_grid, coef)

    centers, means, stds, counts = bin_stats(y_rel, ratio, n_bins=60)

    # Fit quality (R^2)
    pred = even_poly_eval(y_rel, coef)
    ss_res = np.sum((ratio - pred) ** 2)
    ss_tot = np.sum((ratio - ratio.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print("\nFit result (ratio = a0 + a2*y^2 + a4*y^4):")
    print(f"  a0 = {coef[0]:.8f}")
    print(f"  a2 = {coef[1]:.8e}")
    print(f"  a4 = {coef[2]:.8e}")
    print(f"  R^2 = {r2:.6f}")
    print(f"  Total ions used = {len(ratio)}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Plot 1: scatter + fit + binned mean
    ax = axes[0]
    ax.scatter(y_rel, ratio, s=2, alpha=0.08, color="steelblue", label="All ions")
    ax.plot(y_grid, ratio_fit, color="crimson", linewidth=2.0, label="Even poly fit")
    if len(centers) > 0:
        ax.plot(centers, means, color="limegreen", linewidth=1.8, label="Binned mean")
    ax.set_ylabel("sigma_minor / sigma_major")
    ax.set_title(
        "Stretching vs y_rel (first 100 frames)\n"
        f"Fit: a0={coef[0]:.4f}, a2={coef[1]:.2e}, a4={coef[2]:.2e}, R^2={r2:.4f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Plot 2: binned statistics with error bars
    ax = axes[1]
    if len(centers) > 0:
        ax.errorbar(
            centers, means, yerr=stds, fmt="o-", markersize=3, linewidth=1,
            color="darkorange", ecolor="gray", elinewidth=0.8, capsize=2,
            label="Bin mean ± std"
        )
    ax.set_ylabel("ratio (mean ± std)")
    ax.set_title("Binned statistics along y_rel")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Plot 3: theta vs y_rel
    ax = axes[2]
    ax.scatter(y_rel, theta, s=2, alpha=0.08, color="purple")
    ax.set_xlabel("y_rel (pixel)  [origin at crystal center]")
    ax.set_ylabel("theta_deg")
    ax.set_title("Ellipse angle vs y_rel")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
