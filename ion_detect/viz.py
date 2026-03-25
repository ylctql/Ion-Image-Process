"""可视化与结果摘要。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def visualize(image, ions, n_sigma=2.0, title="", output_path=None,
              show_zoom=True, zoom_center=None, zoom_size=80,
              boundary=None):
    """
    在图像上标注拟合椭圆。

    n_sigma 控制椭圆的半轴长度 (n_sigma * sigma)。
    boundary: (cx, cy, a, b) 晶格边界椭圆参数, 若提供则绘制在图上。
    可选显示一个局部放大区域。
    zoom_center / zoom_size 保留为兼容参数, 当前未使用 (放大区域为内置固定列表)。
    """
    _ = (zoom_center, zoom_size)
    nrows = 6 if show_zoom else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(20, 5 + (3 * 5 if show_zoom else 0)),
                             gridspec_kw={"height_ratios": [5, 2, 3, 3, 3, 2]
                                          if show_zoom else [1]})
    if not show_zoom:
        axes = [axes]

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
    ax.set_title(
        f"{title}   [{len(ions)} ions, ellipse = {n_sigma} sigma]",
        fontsize=13,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

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
            ax2.set_title(
                f"Zoom: {label}  x=[{x1z},{x2z}]  y=[{y1z},{y2z}]",
                fontsize=11,
            )
            ax2.set_xlabel("x (pixel)")
            ax2.set_ylabel("y (pixel)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"[已保存] {output_path}")
    plt.close(fig)


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
