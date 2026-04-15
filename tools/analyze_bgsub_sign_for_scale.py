"""
临时分析：多帧 bgsub 后椭圆内像素符号分布，并对比「仅用正值分位」与「全像素分位」作为尺度时的差异。
运行：python analyze_bgsub_sign_for_scale.py [--data-dir PATH] [--frames 0,100,200,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ion_detect.blob_preprocess import (  # noqa: E402
    ellipse_interior_mask,
    subtract_gaussian_background,
)
from ion_detect.boundary import estimate_crystal_boundary  # noqa: E402


def _load_frame(path: Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 3:
        x = x.mean(axis=0)
    return np.asarray(x, dtype=np.float64)


def analyze_frame(z: np.ndarray, boundary: tuple[float, float, float, float] | None, pct: float):
    if boundary is not None:
        m = ellipse_interior_mask(z.shape, boundary)
    else:
        m = np.isfinite(z)
    vals = z[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    n = vals.size
    n_neg = int(np.sum(vals < 0))
    n_zero = int(np.sum(vals == 0))
    n_pos = int(np.sum(vals > 0))
    pos = vals[vals > 0]

    def safe_pct(a: np.ndarray, p: float) -> float:
        if a.size == 0:
            return float("nan")
        return float(np.percentile(a, p))

    p = float(np.clip(pct, 1.0, 100.0))
    scale_pos = safe_pct(pos, p) if pos.size > 0 else float("nan")
    scale_all = safe_pct(vals, p)
    scale_abs = safe_pct(np.abs(vals), p)

    # 当前实现：无正值时用 |z| 分位
    scale_impl = scale_pos if pos.size > 0 and np.isfinite(scale_pos) else scale_abs
    eps = 1e-12
    if not np.isfinite(scale_impl) or scale_impl <= eps:
        scale_impl = max(float(np.nanmax(np.abs(vals))), eps)

    med = float(np.median(vals))
    mean = float(np.mean(vals))

    ratio_pos_all = scale_pos / scale_all if scale_all and np.isfinite(scale_pos) else float("nan")

    return {
        "n": n,
        "frac_neg": n_neg / n,
        "frac_zero": n_zero / n,
        "frac_pos": n_pos / n,
        "median": med,
        "mean": mean,
        f"P{p:g}_pos": scale_pos,
        f"P{p:g}_all": scale_all,
        f"P{p:g}_abs": scale_abs,
        "scale_as_implemented": scale_impl,
        "ratio_P_pos_over_P_all": ratio_pos_all,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT.parent / "20260305_1727",
    )
    ap.add_argument(
        "--frames",
        type=str,
        default="0,50,100,200,400,600,800,900",
        help="逗号分隔帧索引",
    )
    ap.add_argument("--pct", type=float, default=95.0, help="与 thr-norm-pct 一致的分位 (1–100)")
    args = ap.parse_args()

    files = sorted(f for f in args.data_dir.iterdir() if f.suffix == ".npy")
    if not files:
        raise SystemExit(f"无 .npy: {args.data_dir}")

    indices = [int(s.strip()) for s in args.frames.split(",") if s.strip()]
    pct = args.pct

    print(f"数据目录: {args.data_dir.resolve()} (共 {len(files)} 帧)")
    print(f"分析帧索引: {indices}")
    print(f"分位数 P{pct:g}；bgsub 与 blob 默认一致 (bg_sigma=(10,30))")
    print("-" * 88)

    rows = []
    for idx in indices:
        if idx < 0 or idx >= len(files):
            print(f"[{idx}] 跳过：越界")
            continue
        path = files[idx]
        img = _load_frame(path)
        signal = subtract_gaussian_background(img, bg_sigma=(10, 30))
        boundary = estimate_crystal_boundary(signal)
        stats = analyze_frame(signal, boundary, pct)
        if stats is None:
            print(f"[{idx}] {path.name}: 无有效像素")
            continue
        rows.append(stats)
        b_ok = boundary is not None
        print(f"\n[{idx:4d}] {path.name}  boundary={'OK' if b_ok else 'None'}")
        print(
            f"  椭圆(或全图)内: n={stats['n']:,}  "
            f"负/零/正占比: {100*stats['frac_neg']:.1f}% / {100*stats['frac_zero']:.2f}% / {100*stats['frac_pos']:.1f}%",
        )
        print(f"  median(all)={stats['median']:.6g}  mean(all)={stats['mean']:.6g}")
        print(
            f"  P{pct:g}(仅正值)={stats[f'P{pct:g}_pos']:.6g}  "
            f"P{pct:g}(全体有限)={stats[f'P{pct:g}_all']:.6g}  "
            f"P{pct:g}(|z|)={stats[f'P{pct:g}_abs']:.6g}",
        )
        print(
            f"  比值 P_pos/P_all = {stats['ratio_P_pos_over_P_all']:.4f}  "
            f"(>1 表示仅用正值分位得到的尺度更大)",
        )

    if not rows:
        return

    mean_frac_neg = float(np.mean([r["frac_neg"] for r in rows]))
    mean_ratio = float(
        np.nanmean([r["ratio_P_pos_over_P_all"] for r in rows]),
    )
    print("\n" + "=" * 88)
    print("汇总（抽样帧平均）:")
    print(f"  平均负值像素占比: {100*mean_frac_neg:.1f}%")
    print(f"  平均 P_pos/P_all: {mean_ratio:.4f}")
    print("\n结论提示（结合本批数据，非通用定理）:")
    if mean_frac_neg < 0.35 and mean_ratio < 1.15:
        print(
            "  - 负值占比不高，且 P_pos 与 P_all 接近：仅用正值作尺度与「全体分位」差别不大，"
            "当前策略通常足够合理。",
        )
    elif mean_frac_neg >= 0.35:
        print(
            "  - 负值占比较高：全体分位 P_all 会显著受负值拖曳/重塑上尾；是否纳入负值取决于你要的语义 "
            "（上尾亮度 vs 相对零的对称离差）。可对比 P_all、P_pos、P(|z|) 三者再决定。",
        )
    if mean_ratio > 1.2:
        print(
            "  - P_pos 明显大于 P_all：说明在「全体」分布下高分位更低，仅用正值分位会得到更「大」的 scale、"
            "归一化更「温和」(z/scale 更小)。若改用全体 P95 作 scale，同等 T 下阈值会更严。",
        )


if __name__ == "__main__":
    main()
