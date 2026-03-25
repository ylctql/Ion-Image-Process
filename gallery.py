from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons

from ion_detection import detect_ions
from output_paths import OUT_AMP_Y_FIT, PROJECT_ROOT


class NpyGalleryApp:
    def __init__(self, data_dir: Path, project_root: Path | None = None):
        self.data_dir = Path(data_dir)
        self._project_root = Path(project_root) if project_root is not None else self.data_dir.parent
        self.files = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() == ".npy"])
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in directory: {self.data_dir}")

        self.idx = 0
        # (frame_index, detect_config_signature) -> (ions, boundary)
        self.detect_cache = {}
        self._updating_controls = False

        self._default_amp_coef = OUT_AMP_Y_FIT / "amp_vs_y_coef_10.npy"

        # Wide figure: detection params on top bar, image full width in the middle.
        self.fig = plt.figure(figsize=(16, 9))

        # Shift content right (less empty margin on the right of the window).
        x0 = 0.068
        w_main = 0.864

        # --- Top: detection options (ion_detection CLI) ---
        self.ax_chk = self.fig.add_axes((x0, 0.898, w_main, 0.078))
        self.chk_params = CheckButtons(
            self.ax_chk,
            ["theta=0", "Y thresh comp", "Matched filt.", "Two-pass ref."],
            [False, False, True, True],
        )

        # Narrow TextBox axes + explicit gaps so labels do not collide with the next box.
        h_tb = 0.042
        y_tb = 0.828
        gap = 0.028
        gap_before_mode = 0.042  # extra space before "mode" only
        x = x0
        w_rel, w_comp, w_j, w_mode = 0.105, 0.090, 0.076, 0.088

        self.ax_tb_rel = self.fig.add_axes((x, y_tb, w_rel, h_tb))
        self.tb_rel_threshold = TextBox(self.ax_tb_rel, "rel ", initial="0.025")
        x += w_rel + gap

        self.ax_tb_comp = self.fig.add_axes((x, y_tb, w_comp, h_tb))
        self.tb_comp_floor = TextBox(self.ax_tb_comp, "c_fl ", initial="0.2")
        x += w_comp + gap

        self.ax_tb_jy = self.fig.add_axes((x, y_tb, w_j, h_tb))
        self.tb_joint_y = TextBox(self.ax_tb_jy, "jY ", initial="")
        x += w_j + gap

        self.ax_tb_jx = self.fig.add_axes((x, y_tb, w_j, h_tb))
        self.tb_joint_x = TextBox(self.ax_tb_jx, "jX ", initial="")
        x += w_j + gap + gap_before_mode

        self.ax_tb_mode = self.fig.add_axes((x, y_tb, w_mode, h_tb))
        self.tb_amp_mode = TextBox(self.ax_tb_mode, "mode ", initial="even")

        # --- Center: image only (nearly full width) ---
        self.ax_img = self.fig.add_axes((x0, 0.175, w_main, 0.638))

        # --- Bottom: navigation (short slider, buttons shifted right with clear gap) ---
        y_sl = 0.108
        h_ctl = 0.044
        w_slider = 0.50
        gap_sl = 0.045  # space between slider and Go / Prev column
        btn_w = 0.086
        col_gap = 0.020

        self.ax_slider = self.fig.add_axes((x0, y_sl, w_slider, h_ctl))
        self.slider = Slider(
            self.ax_slider,
            "Index",
            0,
            len(self.files) - 1,
            valinit=0,
            valstep=1,
        )

        x_go = x0 + w_slider + gap_sl
        x_det = x_go + btn_w + col_gap

        self.ax_text = self.fig.add_axes((x_go, y_sl, btn_w, h_ctl))
        self.textbox = TextBox(self.ax_text, "Go", initial="0")

        self.ax_detect = self.fig.add_axes((x_det, y_sl, btn_w, h_ctl))
        self.btn_detect = Button(self.ax_detect, "Detect")

        y_btn = 0.052
        self.ax_prev = self.fig.add_axes((x_go, y_btn, btn_w, h_ctl))
        self.btn_prev = Button(self.ax_prev, "Prev")

        self.ax_next = self.fig.add_axes((x_det, y_btn, btn_w, h_ctl))
        self.btn_next = Button(self.ax_next, "Next")

        self.ax_status = self.fig.add_axes((x0, 0.012, w_main, 0.032))
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.0, 0.5, "", fontsize=9, va="center")

        self._connect_events()
        self._render_current(show_detection=True)

    def _connect_events(self):
        self.slider.on_changed(self._on_slider_change)
        self.textbox.on_submit(self._on_text_submit)
        self.btn_detect.on_clicked(self._on_detect_click)
        self.btn_prev.on_clicked(lambda _evt: self._jump(self.idx - 1))
        self.btn_next.on_clicked(lambda _evt: self._jump(self.idx + 1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.chk_params.on_clicked(lambda _lbl: self._on_detect_params_changed())
        for tb in (
            self.tb_rel_threshold,
            self.tb_comp_floor,
            self.tb_joint_y,
            self.tb_joint_x,
            self.tb_amp_mode,
        ):
            tb.on_submit(lambda _t: self._on_detect_params_changed())

    def _on_detect_params_changed(self):
        self.detect_cache.clear()
        self._set_status(
            "Detection options changed — click Detect again (or switch frame, then Detect)."
        )
        self._render_current(show_detection=False)

    @staticmethod
    def _parse_optional_float(s: str) -> float | None:
        s = (s or "").strip()
        if not s:
            return None
        return float(s)

    def _detect_config_signature(self) -> tuple:
        st = self.chk_params.get_status()
        jy = self._parse_optional_float(self.tb_joint_y.text)
        jx = self._parse_optional_float(self.tb_joint_x.text)
        mode = (self.tb_amp_mode.text or "even").strip().lower()
        try:
            rel = float((self.tb_rel_threshold.text or "0.025").strip())
        except ValueError:
            rel = float("nan")
        try:
            cf = float((self.tb_comp_floor.text or "0.2").strip())
        except ValueError:
            cf = float("nan")
        return (
            tuple(st),
            rel,
            cf,
            jy if jy is None else round(jy, 6),
            jx if jx is None else round(jx, 6),
            mode,
        )

    def _build_detect_kwargs(self) -> dict:
        st = self.chk_params.get_status()
        fix_theta_zero, use_y_comp, use_matched, refine = st
        try:
            rel = float((self.tb_rel_threshold.text or "0.025").strip())
            comp_floor = float((self.tb_comp_floor.text or "0.2").strip())
        except ValueError:
            raise ValueError("rel_thr and comp_floor must be valid numbers") from None
        joint_y = self._parse_optional_float(self.tb_joint_y.text)
        joint_x = self._parse_optional_float(self.tb_joint_x.text)
        mode = (self.tb_amp_mode.text or "even").strip().lower()
        if mode not in ("even", "poly2"):
            raise ValueError(f"amp_mode must be 'even' or 'poly2', got: {mode!r}")

        kw: dict = {
            "rel_threshold": rel,
            "comp_floor": comp_floor,
            "fix_theta_zero": fix_theta_zero,
            "use_y_threshold_comp": use_y_comp,
            "use_matched_filter": use_matched,
            "refine": refine,
            "joint_pair_y_gap": joint_y,
            "joint_pair_x_gap": joint_x,
            "amp_y_coef_mode": mode,
        }
        if use_y_comp:
            kw["amp_y_coef_path"] = self._default_amp_coef
        return kw

    def _cache_key(self) -> tuple[int, tuple]:
        return (self.idx, self._detect_config_signature())

    def _set_status(self, msg: str):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    @staticmethod
    def _normalize_image(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr.mean(axis=0)
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _load_image(self, idx: int) -> np.ndarray:
        arr = np.load(self.files[idx])
        return self._normalize_image(arr).astype(np.float64)

    def _draw_overlays(self, ions, boundary):
        for ion in ions:
            ell = Ellipse(
                xy=(ion["x0"], ion["y0"]),
                width=2.0 * ion["sigma_minor"],
                height=2.0 * ion["sigma_major"],
                angle=ion["theta_deg"],
                edgecolor="red",
                facecolor="none",
                linewidth=0.8,
                alpha=0.9,
            )
            self.ax_img.add_patch(ell)

        if boundary is not None:
            cx, cy, a, b = boundary
            bnd = Ellipse(
                xy=(cx, cy),
                width=2 * a,
                height=2 * b,
                angle=0,
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
            self.ax_img.add_patch(bnd)

    def _render_current(self, show_detection: bool):
        img = self._load_image(self.idx)
        fname = self.files[self.idx].name

        self.ax_img.clear()
        self.ax_img.imshow(
            img,
            cmap="gray",
            aspect="equal",
            vmin=float(np.percentile(img, 1)),
            vmax=float(np.percentile(img, 99.5)),
        )

        detect_info = ""
        key = self._cache_key()
        if show_detection and key in self.detect_cache:
            ions, boundary = self.detect_cache[key]
            self._draw_overlays(ions, boundary)
            detect_info = f" | detected ions: {len(ions)}"

        self.ax_img.set_title(f"[{self.idx}/{len(self.files)-1}] {fname}{detect_info}")
        self.ax_img.set_xlabel("x (pixel)")
        self.ax_img.set_ylabel("y (pixel)")

        if not self._updating_controls:
            self._updating_controls = True
            self.slider.set_val(self.idx)
            self.textbox.set_val(str(self.idx))
            self._updating_controls = False

        self.fig.canvas.draw_idle()

    def _jump(self, new_idx: int):
        new_idx = max(0, min(len(self.files) - 1, int(new_idx)))
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self._render_current(show_detection=True)

    def _on_slider_change(self, val):
        if self._updating_controls:
            return
        self._jump(int(val))

    def _on_text_submit(self, text: str):
        text = text.strip()
        if not text:
            self._set_status("Please enter an integer index.")
            return
        try:
            target = int(text)
        except ValueError:
            self._set_status(f"Invalid input: {text}")
            self._updating_controls = True
            self.textbox.set_val(str(self.idx))
            self._updating_controls = False
            return

        if target < 0 or target >= len(self.files):
            self._set_status(f"Index out of range: 0 ~ {len(self.files)-1}")
            self._updating_controls = True
            self.textbox.set_val(str(self.idx))
            self._updating_controls = False
            return

        self._set_status("")
        self._jump(target)

    def _on_detect_click(self, _event):
        fname = self.files[self.idx].name
        try:
            kw = self._build_detect_kwargs()
        except ValueError as e:
            self._set_status(str(e))
            return

        self._set_status(f"Detecting: [{self.idx}] {fname} ...")
        self.fig.canvas.draw_idle()

        t0 = time.perf_counter()
        img = self._load_image(self.idx)
        try:
            ions, boundary = detect_ions(img, **kw)
        except Exception as e:
            self._set_status(f"Detection failed: {type(e).__name__}: {e}")
            self.fig.canvas.draw_idle()
            return

        elapsed = time.perf_counter() - t0
        key = self._cache_key()
        self.detect_cache[key] = (ions, boundary)

        self._set_status(f"Done: [{self.idx}] {fname}, ions={len(ions)}, {elapsed:.2f}s")
        self._render_current(show_detection=True)

    def _on_key(self, event):
        if event.key in ("right", "down", "pagedown", "n"):
            self._jump(self.idx + 1)
        elif event.key in ("left", "up", "pageup", "p"):
            self._jump(self.idx - 1)
        elif event.key == "home":
            self._jump(0)
        elif event.key == "end":
            self._jump(len(self.files) - 1)

    def run(self):
        plt.show()


def _resolve_data_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "20260305_1727",
        project_root / "data" / "20260305_1727",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        "Data directory not found. Put `20260305_1727` under the project root, "
        "or update candidate paths in `_resolve_data_dir()`."
    )


def main():
    project_root = Path(__file__).resolve().parent
    data_dir = _resolve_data_dir(project_root)
    app = NpyGalleryApp(data_dir, project_root=project_root)
    app.run()


if __name__ == "__main__":
    main()
