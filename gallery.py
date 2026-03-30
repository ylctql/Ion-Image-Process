from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons

from ion_detection import detect_ions


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

        # Wide figure: detection params on top bar, image full width in the middle.
        self.fig = plt.figure(figsize=(16, 9))

        # Shift content right (less empty margin on the right of the window).
        x0 = 0.068
        w_main = 0.864

        # --- Top: detection options ---
        self.ax_chk = self.fig.add_axes((x0, 0.898, w_main, 0.078))
        self.chk_params = CheckButtons(
            self.ax_chk,
            ["Two-pass ref."],
            [True],
        )

        h_tb = 0.042
        y_tb = 0.828
        self.ax_tb_rel = self.fig.add_axes((x0, y_tb, 0.12, h_tb))
        self.tb_rel_threshold = TextBox(self.ax_tb_rel, "rel ", initial="0.025")
        self.tb_rel_threshold.on_submit(lambda _t: self._on_detect_params_changed())

        # --- Center: image only (nearly full width) ---
        self.ax_img = self.fig.add_axes((x0, 0.175, w_main, 0.638))

        # --- Bottom: navigation ---
        y_sl = 0.108
        h_ctl = 0.044
        w_slider = 0.50
        gap_sl = 0.045
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

    def _on_detect_params_changed(self):
        self.detect_cache.clear()
        self._set_status(
            "Detection options changed — click Detect again (or switch frame, then Detect)."
        )
        self._render_current(show_detection=False)

    def _detect_config_signature(self) -> tuple:
        st = self.chk_params.get_status()
        try:
            rel = float((self.tb_rel_threshold.text or "0.025").strip())
        except ValueError:
            rel = float("nan")
        return (tuple(st), round(rel, 8) if np.isfinite(rel) else rel)

    def _build_detect_kwargs(self) -> dict:
        st = self.chk_params.get_status()
        refine = st[0]
        try:
            rel = float((self.tb_rel_threshold.text or "0.025").strip())
        except ValueError as e:
            raise ValueError("rel must be a valid number") from e

        return {
            "rel_threshold": rel,
            "refine": refine,
        }

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
                width=2 * 2.0 * ion["sigma_minor"],
                height=2 * 2.0 * ion["sigma_major"],
                angle=ion["theta_deg"],
                edgecolor="lime",
                facecolor="none",
                linewidth=1.2,
            )
            self.ax_img.add_patch(ell)
        if boundary is not None:
            bcx, bcy, ba, bb = boundary
            ell_b = Ellipse(
                xy=(bcx, bcy),
                width=2 * ba,
                height=2 * bb,
                angle=0,
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.0,
                linestyle="--",
            )
            self.ax_img.add_patch(ell_b)

    def _render_current(self, show_detection: bool):
        self.ax_img.clear()
        img = self._load_image(self.idx)
        self.ax_img.imshow(img, cmap="gray", aspect="equal")
        self.ax_img.set_title(self.files[self.idx].name, fontsize=11)

        if show_detection:
            key = self._cache_key()
            hit = self.detect_cache.get(key)
            if hit is not None:
                ions, boundary = hit
                self._draw_overlays(ions, boundary)

        self.ax_img.set_xlabel("x (pixel)")
        self.ax_img.set_ylabel("y (pixel)")
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        if self._updating_controls:
            return
        new_idx = int(round(float(val)))
        self._jump(new_idx)

    def _jump(self, target: int):
        target = int(np.clip(target, 0, len(self.files) - 1))
        if target == self.idx:
            return
        self.idx = target
        self._updating_controls = True
        self.slider.set_val(self.idx)
        self.textbox.set_val(str(self.idx))
        self._updating_controls = False
        self._set_status("")
        self._render_current(show_detection=True)

    def _on_text_submit(self, text: str):
        try:
            target = int((text or "").strip())
        except ValueError:
            self._set_status(f"Invalid integer: {text!r}")
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
