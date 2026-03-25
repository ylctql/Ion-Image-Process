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
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ion_detection import detect_ions
from output_paths import OUT_AMP_Y_FIT, OUT_STRETCH_ANALYSIS, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "20260305_1727"

# 形变测量模式: 放宽 sigma 上限、增大拟合窗口、关闭两阶段精修,
# 避免边缘离子的 sigma_major 被截断导致 ratio 失真。
DETECT_KWARGS = dict(
    sigma_range=(0.3, 8.0),   # 允许 sigma_major 达 8 px (默认 3.5)
    fit_hw=(5, 5),             # 11x11 窗口覆盖拉伸离子全貌 (默认 7x9)
    refine=False,              # 跳过二次精修中的 sigma 收紧
)
QUADRATIC_Y_LIMIT = 40.0


def _gaussian_model(x: np.ndarray, c0: float, amp: float, mu: float, sigma: float):
    return c0 + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_model(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    if method == "quadratic":
        return np.polyfit(x, y, 2)
    if method == "quartic":
        return np.polyfit(x, y, 4)
    if method == "gaussian":
        c0 = float(np.percentile(y, 10))
        amp = float(np.max(y) - c0)
        mu = float(x[np.argmax(y)])
        sigma = float(max(float(np.std(x)), 1.0))
        p0 = np.array([c0, amp, mu, sigma], dtype=np.float64)
        lo = np.array([-np.inf, -np.inf, float(np.min(x)), 1e-6], dtype=np.float64)
        hi = np.array([np.inf, np.inf, float(np.max(x)), np.inf], dtype=np.float64)
        popt, _ = curve_fit(_gaussian_model, x, y, p0=p0, bounds=(lo, hi), maxfev=10000)
        return popt
    raise ValueError(f"Unsupported fit method: {method}")


def eval_model(x: np.ndarray, params: np.ndarray, method: str) -> np.ndarray:
    if method in ("quadratic", "quartic"):
        return np.polyval(params, x)
    if method == "gaussian":
        return _gaussian_model(x, *params)
    raise ValueError(f"Unsupported fit method: {method}")


def select_fit_data(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    quadratic_y_limit: float = QUADRATIC_Y_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fitting subset by method."""
    if method == "quadratic":
        mask = np.abs(x) <= quadratic_y_limit
        if int(mask.sum()) < 3:
            raise ValueError(
                f"Not enough points for quadratic fit within |y_rel| <= {quadratic_y_limit}."
            )
        return x[mask], y[mask]
    return x, y


def fit_r2(x: np.ndarray, y: np.ndarray, params: np.ndarray, method: str) -> float:
    pred = eval_model(x, params, method)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def format_params(params: np.ndarray, method: str) -> str:
    if method in ("quadratic", "quartic"):
        terms = [f"p{i}={v:.3e}" for i, v in enumerate(params)]
        return ", ".join(terms)
    return f"c0={params[0]:.3e}, amp={params[1]:.3e}, mu={params[2]:.3e}, sigma={params[3]:.3e}"


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


def main(
    n_frames: int = 100,
    ratio_fit_method: str = "quadratic",
    amp_fit_method: str = "quadratic",
):
    OUT_STRETCH_ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_AMP_Y_FIT.mkdir(parents=True, exist_ok=True)
    out_fig = OUT_STRETCH_ANALYSIS / (
        f"stretching_analysis_{n_frames}_ratio-{ratio_fit_method}_amp-{amp_fit_method}.png"
    )
    amp_coef_file = OUT_AMP_Y_FIT / f"amp_vs_y_coef_{n_frames}_{amp_fit_method}.npy"

    files = sorted(DATA_DIR.glob("*.npy"))[:n_frames]
    if not files:
        raise RuntimeError(f"No .npy files found in {DATA_DIR}")

    all_y_rel = []
    all_ratio = []
    all_theta = []
    all_amp = []
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
        amp = np.asarray([d["amplitude"] for d in ions], dtype=np.float64)

        ratio = minor / np.maximum(major, 1e-8)
        y_rel = y0 - cy_ref

        all_y_rel.append(y_rel)
        all_ratio.append(ratio)
        all_theta.append(theta)
        all_amp.append(amp)

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
    amp = np.concatenate(all_amp)

    y_rel_fit_ratio, ratio_fit_data = select_fit_data(y_rel, ratio, ratio_fit_method)
    coef = fit_model(y_rel_fit_ratio, ratio_fit_data, ratio_fit_method)
    y_grid = np.linspace(y_rel.min(), y_rel.max(), 800)
    ratio_fit = eval_model(y_grid, coef, ratio_fit_method)

    centers, means, stds, counts = bin_stats(y_rel, ratio, n_bins=60)

    r2 = fit_r2(y_rel_fit_ratio, ratio_fit_data, coef, ratio_fit_method)

    print(f"\nFit result (ratio, method={ratio_fit_method}):")
    print(f"  {format_params(coef, ratio_fit_method)}")
    print(f"  Fit points used = {len(ratio_fit_data)}")
    if ratio_fit_method == "quadratic":
        print(f"  Fit y-range constraint: |y_rel| <= {QUADRATIC_Y_LIMIT:g}")
    print(f"  R^2 = {r2:.6f}")
    print(f"  Total ions used = {len(ratio)}")

    # ── Amplitude vs y_rel ──
    y_rel_fit_amp, amp_fit_data = select_fit_data(y_rel, amp, amp_fit_method)
    coef_amp = fit_model(y_rel_fit_amp, amp_fit_data, amp_fit_method)
    amp_fit = eval_model(y_grid, coef_amp, amp_fit_method)
    centers_a, means_a, stds_a, counts_a = bin_stats(y_rel, amp, n_bins=60)

    r2_amp = fit_r2(y_rel_fit_amp, amp_fit_data, coef_amp, amp_fit_method)

    print(f"\nFit result (amplitude, method={amp_fit_method}):")
    print(f"  {format_params(coef_amp, amp_fit_method)}")
    print(f"  Fit points used = {len(amp_fit_data)}")
    if amp_fit_method == "quadratic":
        print(f"  Fit y-range constraint: |y_rel| <= {QUADRATIC_Y_LIMIT:g}")
    print(f"  R^2 = {r2_amp:.6f}")

    np.save(amp_coef_file, coef_amp)
    print(f"  Saved amplitude coefficients: {amp_coef_file}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    # Plot 1: scatter + fit + binned mean
    ax = axes[0]
    ax.scatter(y_rel, ratio, s=2, alpha=0.08, color="steelblue", label="All ions")
    ax.plot(y_grid, ratio_fit, color="crimson", linewidth=2.0, label=f"{ratio_fit_method} fit")
    if len(centers) > 0:
        ax.plot(centers, means, color="limegreen", linewidth=1.8, label="Binned mean")
    ax.set_ylabel("sigma_minor / sigma_major")
    ax.set_title(
        f"Stretching vs y_rel (first {len(files)} frames)\n"
        f"Method={ratio_fit_method}, R^2={r2:.4f}"
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
    ax.set_ylabel("theta_deg")
    ax.set_title("Ellipse angle vs y_rel")
    ax.grid(alpha=0.3)

    # Plot 4: amplitude vs y_rel
    ax = axes[3]
    ax.scatter(y_rel, amp, s=2, alpha=0.08, color="teal", label="All ions")
    ax.plot(y_grid, amp_fit, color="crimson", linewidth=2.0, label=f"{amp_fit_method} fit")
    if len(centers_a) > 0:
        ax.plot(centers_a, means_a, color="limegreen", linewidth=1.8, label="Binned mean")
    ax.set_xlabel("y_rel (pixel)  [origin at crystal center]")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        "Amplitude vs y_rel\n"
        f"Method={amp_fit_method}, R^2={r2_amp:.4f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"\nSaved: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ion stretching and amplitude vs y.")
    parser.add_argument(
        "-n", "--n-frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100).",
    )
    parser.add_argument(
        "--ratio-fit",
        type=str,
        choices=("quadratic", "quartic", "gaussian"),
        default="quadratic",
        help="Fit method for ratio vs y_rel.",
    )
    parser.add_argument(
        "--amp-fit",
        type=str,
        choices=("quadratic", "quartic", "gaussian"),
        default="quadratic",
        help="Fit method for amplitude vs y_rel.",
    )
    args = parser.parse_args()
    main(
        n_frames=args.n_frames,
        ratio_fit_method=args.ratio_fit,
        amp_fit_method=args.amp_fit,
    )
