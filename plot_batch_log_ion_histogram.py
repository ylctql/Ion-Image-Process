"""Parse batch_merge_second_layer_slab log lines (离子数=...) and plot ion-count histogram (English labels)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# batch_merge_second_layer_slab per-frame log line contains: 离子数=1234
ION_COUNT_RE = re.compile(r"离子数=(\d+)")


def parse_ion_counts(log_path: Path) -> np.ndarray:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    counts = [int(m.group(1)) for m in ION_COUNT_RE.finditer(text)]
    if not counts:
        raise SystemExit(f"No '离子数=' entries found in {log_path}")
    return np.asarray(counts, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram of per-frame ion counts from batch run log.")
    parser.add_argument("log_file", type=Path, help="Path to batch_run.log (or similar)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: same directory as log, ion_count_histogram.png)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Histogram bin width in ion-count units (default: 1)",
    )
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    counts = parse_ion_counts(log_path)
    out = args.out
    if out is None:
        out = log_path.parent / "ion_count_histogram_from_log.png"
    else:
        out = out.resolve()

    n = int(counts.size)
    mu = float(np.mean(counts))
    sigma = float(np.std(counts))
    vmin, vmax = int(counts.min()), int(counts.max())
    w = float(args.bin_width)
    if w <= 0:
        raise SystemExit("--bin-width must be positive")
    # Edges ... [k*w, (k+1)*w) so each bin has width w; for w=1 and integer counts use integer-aligned edges
    lo = np.floor(vmin / w) * w
    hi = np.ceil((vmax + 1) / w) * w
    bin_edges = np.arange(lo, hi + w * 0.5, w, dtype=np.float64)

    freqs, _ = np.histogram(counts, bins=bin_edges)
    i_peak = int(np.argmax(freqs))
    peak_ion_left = float(bin_edges[i_peak])
    peak_ion_center = peak_ion_left + w / 2.0
    peak_freq = int(freqs[i_peak])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=bin_edges, color="steelblue", edgecolor="black", linewidth=0.35, alpha=0.88)
    ax.axvline(mu, color="crimson", ls="--", lw=1.5, label=f"mean = {mu:.2f}")
    ax.axvline(
        peak_ion_center,
        color="darkgreen",
        ls="-",
        lw=1.8,
        label=f"tallest bin: ion count = {peak_ion_left:g} ({peak_freq} frames)",
    )
    ax.set_xlabel("Ion count (per frame)")
    ax.set_ylabel("Number of frames")
    ax.set_title(f"Histogram of merged ion counts (bin width = {w:g})")
    ax.legend(loc="upper right")

    stats = (
        f"N = {n}\n"
        f"min = {vmin}\n"
        f"max = {vmax}\n"
        f"mean = {mu:.2f}\n"
        f"std = {sigma:.2f}\n"
        f"peak ion count = {peak_ion_left:g}\n"
        f"  (tallest bin, {peak_freq} frames)"
    )
    ax.text(
        0.02,
        0.98,
        stats,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="0.4"),
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {out} | N={n}, mean={mu:.2f}, std={sigma:.2f}, "
        f"peak ion count={peak_ion_left:g} (freq={peak_freq})",
    )


if __name__ == "__main__":
    main()
