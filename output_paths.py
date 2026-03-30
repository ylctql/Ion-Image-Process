"""项目中数据目录与输出目录的默认路径（集中在此文件）。"""
from __future__ import annotations

import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# 多数脚本默认的 .npy 数据目录（输入）
DEFAULT_DATA_DIR = PROJECT_ROOT / "20260305_1727"

OUT_ION_DETECT_IMGS = OUTPUTS_ROOT / "ion_detect_imgs"
OUT_ION_POS = OUTPUTS_ROOT / "IonPos"
OUT_AMP_Y_FIT = OUTPUTS_ROOT / "amp_y_fit"
OUT_STRETCH_ANALYSIS = OUTPUTS_ROOT / "stretch_analysis"
OUT_HISTOGRAM = OUTPUTS_ROOT / "histogram"
OUT_DATASET_VIZ = OUTPUTS_ROOT / "dataset_viz"  # project_info.py
OUT_NPY_SELECTED = OUTPUTS_ROOT / "npy_plots"  # vis_selected_npy.py
OUT_BGSUB_IMGS = OUTPUTS_ROOT / "bgsub_imgs"
OUT_BGSUB_BIN_IMGS = OUTPUTS_ROOT / "bgsub_binarize_imgs"
OUT_EDGE_STRIP = OUTPUTS_ROOT / "edge_strip_profiles"
OUT_Y_LAYER_PROFILE = OUTPUTS_ROOT / "y_layer_profiles"
OUT_ION_CENTERS_MERGED = OUTPUTS_ROOT / "ion_centers_merged"
OUT_SECOND_LAYER_PEAKS = OUTPUTS_ROOT / "second_layer_peaks"
OUT_BLOB_CONNECTED = OUTPUTS_ROOT / "blob_connected"


def new_batch_merge_slab_run_dir() -> Path:
    """
    ``batch_merge_second_layer_slab`` 单次运行的输出根目录（``outputs/batch_merge_slab_<时间戳>``），
    避免互相覆盖。
    """
    d = OUTPUTS_ROOT / f"batch_merge_slab_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_log_ion_hist_png(log_stem: str) -> Path:
    """由 ``plot_batch_log_ion_histogram`` 写入的默认 PNG。"""
    return OUT_HISTOGRAM / f"log_ion_hist_{log_stem}.png"


def default_cdist_hist_png(config_count: int) -> Path:
    """``dist.py`` 直方图默认路径。"""
    return OUT_HISTOGRAM / f"cdist_hist_{config_count}.png"


def default_vis_selected_montage_png() -> Path:
    """``vis_selected_npy --one-figure`` 时的默认单文件输出。"""
    return OUT_NPY_SELECTED / "selected_montage.png"
