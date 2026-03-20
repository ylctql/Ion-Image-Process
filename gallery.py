from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, TextBox, Button

from ion_detection import detect_ions


class NpyGalleryApp:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.files = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() == ".npy"])
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in directory: {self.data_dir}")

        self.idx = 0
        self.detect_cache = {}
        self._updating_controls = False

        self.fig = plt.figure(figsize=(14, 8))
        self.ax_img = self.fig.add_axes((0.05, 0.24, 0.90, 0.70))

        self.ax_slider = self.fig.add_axes((0.10, 0.14, 0.62, 0.04))
        self.slider = Slider(
            self.ax_slider,
            "Index",
            0,
            len(self.files) - 1,
            valinit=0,
            valstep=1,
        )

        self.ax_text = self.fig.add_axes((0.75, 0.07, 0.10, 0.04))
        self.textbox = TextBox(self.ax_text, "Go", initial="0")

        self.ax_detect = self.fig.add_axes((0.87, 0.07, 0.08, 0.04))
        self.btn_detect = Button(self.ax_detect, "Detect")

        self.ax_prev = self.fig.add_axes((0.75, 0.02, 0.09, 0.04))
        self.btn_prev = Button(self.ax_prev, "Prev")

        self.ax_next = self.fig.add_axes((0.86, 0.02, 0.09, 0.04))
        self.btn_next = Button(self.ax_next, "Next")

        self.ax_status = self.fig.add_axes((0.05, 0.05, 0.65, 0.08))
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.0, 0.5, "", fontsize=11, va="center")

        self._connect_events()
        self._render_current(show_detection=True)

    def _connect_events(self):
        self.slider.on_changed(self._on_slider_change)
        self.textbox.on_submit(self._on_text_submit)
        self.btn_detect.on_clicked(self._on_detect_click)
        self.btn_prev.on_clicked(lambda _evt: self._jump(self.idx - 1))
        self.btn_next.on_clicked(lambda _evt: self._jump(self.idx + 1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

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
            aspect="auto",
            vmin=float(np.percentile(img, 1)),
            vmax=float(np.percentile(img, 99.5)),
        )

        detect_info = ""
        if show_detection and self.idx in self.detect_cache:
            ions, boundary = self.detect_cache[self.idx]
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
        self._set_status(f"Detecting: [{self.idx}] {fname} ...")
        self.fig.canvas.draw_idle()

        t0 = time.perf_counter()
        img = self._load_image(self.idx)
        ions, boundary = detect_ions(img)
        elapsed = time.perf_counter() - t0
        self.detect_cache[self.idx] = (ions, boundary)

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
    app = NpyGalleryApp(data_dir)
    app.run()


if __name__ == "__main__":
    main()
