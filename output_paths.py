"""项目输出目录约定：统一放在项目根下的 ``outputs/`` 子目录。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

OUT_ION_DETECT_IMGS = OUTPUTS_ROOT / "ion_detect_imgs"
OUT_ION_POS = OUTPUTS_ROOT / "IonPos"
OUT_AMP_Y_FIT = OUTPUTS_ROOT / "amp_y_fit"
OUT_STRETCH_ANALYSIS = OUTPUTS_ROOT / "stretch_analysis"
OUT_HISTOGRAM = OUTPUTS_ROOT / "histogram"
OUT_DATASET_VIZ = OUTPUTS_ROOT / "dataset_viz"  # project_info.py
OUT_NPY_SELECTED = OUTPUTS_ROOT / "npy_plots"  # vis_selected_npy.py
OUT_RESIDUAL_IMGS = OUTPUTS_ROOT / "residual_imgs"  # peak-peel 残差图
OUT_EDGE_STRIP = OUTPUTS_ROOT / "edge_strip_profiles"  # edge_strip_profile.py 列轮廓 PNG
OUT_Y_LAYER_PROFILE = OUTPUTS_ROOT / "y_layer_profiles"  # y_layer_profile.py 沿 x 条带行积分
OUT_ION_CENTERS_MERGED = OUTPUTS_ROOT / "ion_centers_merged"  # merge_ion_centers.py 合并中心 PNG
OUT_SECOND_LAYER_PEAKS = OUTPUTS_ROOT / "second_layer_peaks"  # second_layer_ion_peaks.py 第二 y 层 x 峰 + COM
