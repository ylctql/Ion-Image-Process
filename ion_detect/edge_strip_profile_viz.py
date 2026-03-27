"""外缘条带轮廓：汇总图与按峰列的交互 gallery（依赖 Matplotlib）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from matplotlib.widgets import RadioButtons, Slider, TextBox

from .edge_strip import outer_y_edge_strip_masks
from .edge_strip_profile_analysis import (
    column_y_profile_in_strip,
    fitted_xy_for_auxiliary_strip_peaks,
    gaussian_1d_profile,
    masked_strip_profiles_for_plot,
    strip_profile_peak_xs,
    strip_profile_peaks_ixy,
    two_gaussian_1d_profile,
    y_center_from_y_profile_fit,
    y_center_of_mass_from_profile,
    y_marked_com_nearest_local_peak,
)


def draw_edge_strip_outlines(ax: Axes, meta: dict, image_shape: tuple[int, int]) -> None:
    h, w = image_shape
    cx, cy, a, b = meta["cx"], meta["cy"], meta["a"], meta["b"]
    x_half = meta["x_half"]
    y_below, y_above = meta["y_below"], meta["y_above"]
    y_top_v, y_bot_v = meta["y_top_vertex"], meta["y_bot_vertex"]
    lc = "orange"
    lw = 1.2
    xt = [
        cx - x_half, cx + x_half, cx + x_half, cx - x_half, cx - x_half,
    ]
    yt = [y_top_v, y_top_v, y_below, y_below, y_top_v]
    ax.plot(xt, yt, color=lc, linewidth=lw, linestyle="-", alpha=0.9)
    xb = [
        cx - x_half, cx + x_half, cx + x_half, cx - x_half, cx - x_half,
    ]
    yb = [y_above, y_above, y_bot_v, y_bot_v, y_above]
    ax.plot(xb, yb, color=lc, linewidth=lw, linestyle="-", alpha=0.9)
    ax.axhline(y_below, color="darkmagenta", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.axhline(y_above, color="darkmagenta", linestyle=":", linewidth=1.0, alpha=0.75)
    ell = Ellipse(
        xy=(cx, cy), width=2 * a, height=2 * b, angle=0,
        edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--", alpha=0.85,
    )
    ax.add_patch(ell)
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)


def plot_edge_strip_dashboard(
    image: np.ndarray,
    boundary: tuple[float, float, float, float],
    result: dict,
    out_path: Path | None,
    title: str,
    preprocess: str,
    *,
    show: bool,
    peak_dist: float,
    left_panel: np.ndarray | None = None,
    strip_map: np.ndarray | None = None,
    plot_center: str | None = None,
    clip_ellipse: bool = True,
    y_fit_frac: float | None = None,
    add_neighbor_x: bool = False,
    double_peak_fit: bool = False,
    prominence_min: float | None = None,
) -> None:
    """``left_panel``: imshow 顶栏；None 时用原始 ``image``；peel 模式仅在指定 ``--plot-peel`` 时传入残差图。

    ``plot_center``：``None`` 不标点；``"fit"`` 为峰拟合/ prominence 链路与 gallery 一致；``"com"`` 为亮度加权质心；
    ``"com_fit"`` 为 COM 后取沿 y 邻域局部峰中距 COM 最近者作为标点。非 ``None`` 时需传入 ``strip_map``。
    ``prominence_min`` 仅在 ``plot_center=="fit"`` 下生效；``"com"`` / ``"com_fit"`` 时忽略。
    """
    panel0 = image if left_panel is None else left_panel
    meta = result["meta"]
    h, w = panel0.shape
    x = result["x"]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.33, wspace=0.2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    lo = float(np.percentile(panel0, 1))
    hi = float(np.percentile(panel0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(panel0)), float(np.nanmax(panel0))
    ax0.imshow(panel0, cmap="gray", aspect="equal", vmin=lo, vmax=hi)
    draw_edge_strip_outlines(ax0, meta, (h, w))
    fe = f"{float(meta['y_edge_frac']):g}"
    clip = "clip_ellipse" if meta["clip_ellipse"] else "full_rect"
    cm = meta.get("col_metric", "mean")
    agg = {"sum": "col_sum", "mean": "col_mean/mask", "max": "col_max/mask"}.get(cm, cm)
    pd = float(peak_dist)
    dash_hint = (
        f"strip peaks: spacing>{pd:g} px (prominence tie-break)"
        if pd > 0.0
        else "strip peaks: all local maxima"
    )
    center_hint = ""
    if plot_center is not None:
        if plot_center == "com":
            _fit_lab = "intensity COM (strip col profile)"
        elif plot_center == "com_fit":
            _fit_lab = "nearest y-local max to COM (1D neighbors)"
        elif prominence_min is not None:
            _fit_lab = f"prom top-2 (min={prominence_min:g})"
        elif double_peak_fit:
            _fit_lab = "2-Gauss midpoint"
        else:
            _fit_lab = "1-Gauss mu"
        center_hint = f"\ndots: (x_aux, y_center) {_fit_lab} on strip map"
    ax0.set_title(
        f"{title}\nF={fe}, preprocess={preprocess}, {clip}, {agg}\n"
        f"(dashes: {dash_hint}; tomato=top, lime=bottom){center_hint}"
    )
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")

    _yl = {
        "sum": ("col sum (strip)", "col sum", "sum ~"),
        "mean": ("col mean (strip)", "col mean", "mean ~"),
        "max": ("col max (strip)", "col max", "max ~"),
    }
    y_metric, ylbl_short, peak_lbl = _yl.get(cm, (cm, cm, "peak ~"))

    top_plot, bot_plot, tc, bc = masked_strip_profiles_for_plot(result)

    y_below_m = float(meta["y_below"])
    y_above_m = float(meta["y_above"])
    y_top_v = float(meta["y_top_vertex"])
    y_bot_v = float(meta["y_bot_vertex"])
    for xp in strip_profile_peak_xs(x, top_plot, tc, peak_dist):
        ax0.plot(
            [xp, xp], [y_top_v, y_below_m],
            color="tomato", linestyle="--", linewidth=1.0, alpha=0.92, zorder=5,
        )
    for xp in strip_profile_peak_xs(x, bot_plot, bc, peak_dist):
        ax0.plot(
            [xp, xp], [y_above_m, y_bot_v],
            color="lime", linestyle="--", linewidth=1.0, alpha=0.92, zorder=5,
        )

    if plot_center is not None:
        if strip_map is None:
            raise ValueError(
                "plot_edge_strip_dashboard(..., plot_center=...) requires strip_map when plot_center is set",
            )
        top_xy, bot_xy = fitted_xy_for_auxiliary_strip_peaks(
            strip_map,
            result,
            boundary,
            peak_dist=peak_dist,
            clip_ellipse=clip_ellipse,
            y_fit_frac=y_fit_frac,
            add_neighbor_x=add_neighbor_x,
            double_peak_fit=double_peak_fit,
            prominence_min=prominence_min,
            center_mode=plot_center,
        )
        if top_xy:
            txs, tys = zip(*top_xy)
            ax0.scatter(
                txs,
                tys,
                s=22,
                marker="o",
                c="tomato",
                edgecolors="tomato",
                linewidths=0,
                zorder=6,
            )
        if bot_xy:
            bxs, bys = zip(*bot_xy)
            ax0.scatter(
                bxs,
                bys,
                s=22,
                marker="o",
                c="lime",
                edgecolors="lime",
                linewidths=0,
                zorder=6,
            )

    ax1.plot(x, top_plot, color="tab:blue", linewidth=1.0)
    ax1.axvline(result["top_peak_x"], color="tab:red", linestyle="--", alpha=0.85)
    ax1.scatter([result["top_peak_x"]], [result["top_peak_value"]], color="tab:red", s=36, zorder=3)
    ax1.set_title(
        f"Top strip: {y_metric} vs x\npeak x ~ {result['top_peak_x']:.2f}, "
        f"{peak_lbl} {result['top_peak_value']:.4g}"
    )
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel(ylbl_short)

    ax2.plot(x, bot_plot, color="tab:green", linewidth=1.0)
    ax2.axvline(result["bot_peak_x"], color="tab:red", linestyle="--", alpha=0.85)
    ax2.scatter([result["bot_peak_x"]], [result["bot_peak_value"]], color="tab:red", s=36, zorder=3)
    ax2.set_title(
        f"Bottom strip: {y_metric} vs x\npeak x ~ {result['bot_peak_x']:.2f}, "
        f"{peak_lbl} {result['bot_peak_value']:.4g}"
    )
    ax2.set_xlabel("x (px)")
    ax2.set_ylabel(ylbl_short)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"[已保存] {out_path}")
    if not show:
        plt.close(fig)


def show_peak_column_gallery(
    strip_map: np.ndarray,
    result: dict,
    *,
    peak_dist: float,
    title: str,
    preprocess: str,
    boundary: tuple[float, float, float, float],
    clip_ellipse: bool,
    y_fit_frac: float | None = None,
    add_neighbor_x: bool = False,
    double_peak_fit: bool = False,
    prominence_min: float | None = None,
    center_mode: str = "fit",
) -> None:
    """交互 gallery：每条辅助峰列上的 y-profile 与 y 中心估计。

    峰位置仍由 ``result``（``--y-edge-frac``）对应的条带 1D 轮廓决定；若给定
    ``y_fit_frac``，则沿列采样与拟合时使用按该 F 另行构造的更宽（或不同）条带掩膜，
    与 ``outer_y_edge_strip_masks`` / ``--y-edge-frac`` 同一参数含义。
    ``add_neighbor_x`` 为真时，profile 每点为该列与左右邻列强度之和（见 ``column_y_profile_in_strip``）。
    ``center_mode=="com"`` 时竖线标注亮度质心；``"com_fit"`` 标注最近局部峰（虚线可辅以 COM 点线）。
    否则 ``prominence_min`` 非 None 时以此选 y 向 top-2 局部峰（见 ``y_center_from_profile_prominence_top2``），
    ``double_peak_fit`` 控制双/单高斯曲线。
    """
    top_plot, bot_plot, tc, bc = masked_strip_profiles_for_plot(result)
    x = result["x"]
    top_peaks = strip_profile_peaks_ixy(x, top_plot, tc, peak_dist)
    bot_peaks = strip_profile_peaks_ixy(x, bot_plot, bc, peak_dist)
    meta = result["meta"]
    peak_f = float(meta["y_edge_frac"])
    if y_fit_frac is None or abs(float(y_fit_frac) - peak_f) <= 1e-9:
        top_m_fit = result["top_mask"]
        bot_m_fit = result["bot_mask"]
        f_fit_used = peak_f
    else:
        top_m_fit, bot_m_fit, meta_fit = outer_y_edge_strip_masks(
            boundary, float(y_fit_frac), strip_map.shape, clip_ellipse=clip_ellipse,
        )
        f_fit_used = float(meta_fit["y_edge_frac"])
    fe_strip = f"{peak_f:g}"
    fe_fit = f"{f_fit_used:g}"
    cm = meta.get("col_metric", "mean")
    clip = "clip_ellipse" if meta["clip_ellipse"] else "full_rect"

    which = ["top"]
    n_top, n_bot = len(top_peaks), len(bot_peaks)
    nmax = max(n_top, n_bot, 1)
    slider_hi = max(0, nmax - 1)

    fig = plt.figure(figsize=(11, 6.5))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and getattr(mgr, "set_window_title", None):
        mgr.set_window_title("strip peak column profiles")
    # 归一化坐标 [left, bottom, width, height]：整体略偏右下以趋近窗口中心；底行滑块与 Peak # 文本框拉开间距避免重叠
    ax = fig.add_axes((0.12, 0.22, 0.62, 0.58))
    ax_slider = fig.add_axes((0.12, 0.07, 0.34, 0.042))
    ax_text = fig.add_axes((0.51, 0.07, 0.24, 0.042))
    ax_radio = fig.add_axes((0.78, 0.38, 0.18, 0.20))

    slider = Slider(ax_slider, "Idx", 0, slider_hi, valinit=0, valstep=1)
    peak_text = TextBox(ax_text, "Peak # ", initial="1")
    radio = RadioButtons(ax_radio, ("Top strip", "Bot strip"), active=0)
    setattr(fig, "peak_column_gallery_widgets", (slider, radio, peak_text))

    def clamp_slider_to_current_strip(*, silent: bool = False) -> None:
        """将滑块夹到当前条带峰数量范围内；silent=True 时不触发 Slider 回调（由调用方 redraw）。"""
        peaks, _, _ = current()
        if peaks:
            hi = len(peaks) - 1
            k = int(np.clip(round(float(slider.val)), 0, hi))
            if silent:
                slider.eventson = False
                slider.set_val(k)
                slider.eventson = True
            else:
                slider.set_val(k)
        else:
            if silent:
                slider.eventson = False
                slider.set_val(0)
                slider.eventson = True
            else:
                slider.set_val(0)

    def current() -> tuple[list[tuple[int, float, float]], np.ndarray, str]:
        if which[0] == "top":
            return top_peaks, top_m_fit, "top"
        return bot_peaks, bot_m_fit, "bottom"

    neigh = "+x_neigh" if add_neighbor_x else "col_only"
    if center_mode == "com":
        peak_fit_mode = "COM"
    elif center_mode == "com_fit":
        peak_fit_mode = "COM+nearest_peak"
    elif prominence_min is not None:
        peak_fit_mode = f"prom(>={prominence_min:g})"
    elif double_peak_fit:
        peak_fit_mode = "2-Gauss"
    else:
        peak_fit_mode = "1-Gauss"
    ylab_prof = (
        "I[y,x-1]+I[y,x]+I[y,x+1] (strip map)" if add_neighbor_x else "intensity (strip map)"
    )

    def redraw(_: float | None = None) -> None:
        peaks, mask, sname = current()
        extra = (
            f"F_strip={fe_strip}, F_fit={fe_fit}, preprocess={preprocess}, {clip}, "
            f"col_metric={cm}, peak_dist={peak_dist:g}, profile={neigh}, yfit={peak_fit_mode}\n"
            "Keys: Left/Right peak | Up/Down strip | Enter in Peak # (1-based)"
        )
        if not peaks:
            ax.clear()
            ax.set_title(f"{title}\n{extra}\n{sname} strip: no auxiliary peaks")
            ax.set_xlabel("y (row px)")
            ax.set_ylabel(ylab_prof)
            peak_text.set_val("--")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            return
        k = int(np.clip(round(float(slider.val)), 0, len(peaks) - 1))
        peak_text.set_val(str(k + 1))
        col_ix, xp, strip_prof_y = peaks[k]
        y_row, vals = column_y_profile_in_strip(
            strip_map, mask, col_ix, add_neighbor_x=add_neighbor_x,
        )
        if center_mode == "com":
            y_mid = y_center_of_mass_from_profile(y_row, vals)
            popt = None
            fit_info = ""
            y_pr = None
            y_com_ref: float | None = y_mid
        elif center_mode == "com_fit":
            y_mid, y_com_ref = y_marked_com_nearest_local_peak(y_row, vals)
            popt = None
            fit_info = ""
            y_pr = None
        else:
            y_mid, popt, fit_info, y_pr = y_center_from_y_profile_fit(
                y_row,
                vals,
                double_peak_fit=double_peak_fit,
                prominence_min=prominence_min,
            )
            y_com_ref = None

        ax.clear()
        ax.plot(y_row, vals, "o", color="tab:blue", markersize=5, label="data")
        fit_txt = "fit failed"
        if center_mode == "com":
            if y_mid is not None:
                ax.axvline(y_mid, color="tab:red", linestyle="--", linewidth=2.0, label="COM")
                fit_txt = f"COM y={y_mid:.3f}"
            else:
                fit_txt = "COM failed (no positive mass)"
        elif center_mode == "com_fit":
            if y_mid is not None and y_com_ref is not None:
                ax.axvline(y_mid, color="tab:red", linestyle="--", linewidth=2.0, label="center")
                if abs(y_mid - y_com_ref) > 1e-4:
                    ax.axvline(
                        y_com_ref, color="tab:orange", linestyle=":", linewidth=1.5,
                        alpha=0.95, label="COM",
                    )
                    fit_txt = (
                        f"marked y={y_mid:.3f} (nearest local max to COM={y_com_ref:.3f})"
                    )
                else:
                    fit_txt = f"y={y_mid:.3f} (COM or no local max; same)"
            elif y_mid is None:
                fit_txt = "com_fit failed (no positive mass)"
        elif popt is not None and y_row.size >= 2:
            y_f = np.linspace(float(np.min(y_row)), float(np.max(y_row)), 200)
            if popt.size == 7:
                ax.plot(
                    y_f,
                    two_gaussian_1d_profile(y_f, *popt),
                    "-",
                    color="tab:red",
                    linewidth=2.0,
                    label="2-Gauss fit",
                )
                c0, a1, m1, s1, a2, m2, s2 = (float(popt[i]) for i in range(7))
                fit_txt = (
                    f"2-Gauss: c0={c0:.5g}, a1,m1,s1={a1:.5g},{m1:.3f},{s1:.3f}, "
                    f"a2,m2,s2={a2:.5g},{m2:.3f},{s2:.3f}; mid_y={y_mid:.3f}"
                )
            else:
                ax.plot(
                    y_f,
                    gaussian_1d_profile(y_f, *popt),
                    "-",
                    color="tab:red",
                    linewidth=2.0,
                    label="Gaussian fit",
                )
                c0, amp, mu, sig = (float(popt[i]) for i in range(4))
                fit_txt = (
                    f"1-Gauss: c0={c0:.5g}, amp={amp:.5g}, mu={mu:.3f}, sigma={sig:.3f}"
                )
                if fit_info == "single_fallback":
                    fit_txt += " (2-Gauss failed, fallback)"
        elif y_mid is not None and fit_info.startswith("prominence:"):
            if y_pr is not None:
                ya, yb = y_pr
                if abs(ya - yb) > 1e-6:
                    ax.axvline(ya, color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.9)
                    ax.axvline(yb, color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.9)
            ax.axvline(y_mid, color="tab:red", linestyle="--", linewidth=2.0, label="center")
            fit_txt = f"{fit_info}; mid_y={y_mid:.3f}"
        elif y_mid is None:
            fit_txt = f"failed: {fit_info}" if fit_info else "fit failed"

        ax.set_xlabel("y (row px)")
        ax.set_ylabel(ylab_prof)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"{title}\n{extra}\n{sname} strip | peak {k + 1}/{len(peaks)} | col x={xp:.2f} | "
            f"strip-profile@peak={strip_prof_y:.5g}\n{fit_txt}"
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def on_radio(label: str | None) -> None:
        if not label:
            return
        which[0] = "top" if label.startswith("Top") else "bot"
        clamp_slider_to_current_strip(silent=True)
        redraw()

    def on_peak_text(text: str) -> None:
        peaks, _, _ = current()
        if not peaks:
            peak_text.set_val("--")
            return
        t = text.strip()
        try:
            v = int(t)
        except ValueError:
            k0 = int(np.clip(round(float(slider.val)), 0, len(peaks) - 1))
            peak_text.set_val(str(k0 + 1))
            return
        k = int(np.clip(v - 1, 0, len(peaks) - 1))
        prev = int(np.clip(round(float(slider.val)), 0, len(peaks) - 1))
        slider.set_val(k)
        if k == prev:
            redraw()

    def on_key(event: object) -> None:
        raw = getattr(event, "key", None)
        if raw is None:
            return
        key = str(raw)
        peaks, _, _ = current()
        if not peaks:
            return
        hi = len(peaks) - 1
        k = int(np.clip(round(float(slider.val)), 0, hi))
        if key in ("left", "arrow_left"):
            k = max(0, k - 1)
            slider.set_val(k)
            return
        if key in ("right", "arrow_right"):
            k = min(hi, k + 1)
            slider.set_val(k)
            return
        if key in ("up", "arrow_up"):
            if which[0] != "top":
                which[0] = "top"
                radio.set_active(0)
                clamp_slider_to_current_strip(silent=True)
                redraw()
            return
        if key in ("down", "arrow_down"):
            if which[0] != "bot":
                which[0] = "bot"
                radio.set_active(1)
                clamp_slider_to_current_strip(silent=True)
                redraw()
            return

    peak_text.on_submit(on_peak_text)
    radio.on_clicked(on_radio)
    slider.on_changed(redraw)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
