# IonImage

离子晶格图像处理与分析工具集。本项目包含**多种互不替代的定位思路**：以 **`ion_detect`** 包为核心的 **2D 高斯拟合检测**、可选的 **阈值二值化 + 连通域矩形**、以及针对上下缘漏检的 **外缘条带剖面 + 中心合并** 与 **第二层 y 行 / slab** 等。根目录 `ion_detection.py` 为兼容入口（`detect_ions` 与命令行）。

**文档结构（按用途）**

| 章节 | 内容 |
|------|------|
| [1. 环境依赖](#1-环境依赖) | Python 与 pip |
| [2. 数据与目录约定](#2-数据与目录约定) | 输入目录、`output_paths`、各 `outputs/` 子文件夹 |
| [3. 识别与定位方法](#3-识别与定位方法) | 高斯流水线、blob、条带合并、第二层 / 批量 slab |
| [4. 诊断与一维剖面](#4-诊断与一维剖面) | 外缘条带列剖面、固定 x 条带的 y 向分层 |
| [5. 交互浏览与统计分析](#5-交互浏览与统计分析) | Gallery、形变拟合、构型距离、数据集概览、导出 PNG、补充直方图脚本 |
| [6. ion_detect 包内模块分工](#6-ion_detect-包内模块分工) | 便于二次开发的模块表 |
| [7. 推荐工作流](#7-推荐工作流) | 从检测到合并的典型顺序 |
| [8. 调参建议](#8-调参建议) | 常见参数与注意点 |
| [9. 项目文件索引](#9-项目文件索引) | 仓库内主要文件说明 |

---

## 1. 环境依赖

建议 Python 3.10+。

```bash
pip install numpy scipy matplotlib
```

---

## 2. 数据与目录约定

- **输入**：默认 `20260305_1727/*.npy`，每个文件一帧（3D 由各脚本自行解释）。
- **路径约定**：根目录 [`output_paths.py`](output_paths.py) 集中定义 `DEFAULT_DATA_DIR` 与各 `OUT_*`；命令行**不提供**修改输出根目录的选项。

**`outputs/` 下常用子目录**

| 子目录 | 用途 |
|--------|------|
| `outputs/ion_detect_imgs/` | `python -m ion_detect` 默认椭圆叠加 PNG |
| `outputs/bgsub_imgs/` | 减高斯背景（`--save-bgsub-img`） |
| `outputs/bgsub_binarize_imgs/` | bgsub 与二值 mask（`--bgsub-binarize-threshold`） |
| `outputs/blob_connected/` | `python -m ion_detect.blob_cli` 双栏 PNG |
| `outputs/blob/` | `blob_cli` 的 `--log`（`merge_split.log`）、`--hist`（`hist_merge_split.png`） |
| `outputs/pixel_hist/` | `blob_cli` 的 `--plot-pixel-hist` |
| `outputs/IonPos/` | 离子中心 `N×2`（`--save-pos`） |
| `outputs/amp_y_fit/` | y 向振幅拟合系数 |
| `outputs/stretch_analysis/` | `stretching_analysis.py` |
| `outputs/histogram/` | `dist.py`、`plot_batch_log_ion_histogram.py`、`hist_*.py` 等 |
| `outputs/dataset_viz/` | `project_info.py` |
| `outputs/npy_plots/` | `vis_selected_npy.py` |
| `outputs/edge_strip_profiles/` | `edge_strip_profile.py` |
| `outputs/y_layer_profiles/` | `y_layer_profile.py` |
| `outputs/ion_centers_merged/` | `merge_ion_centers.py` |
| `outputs/second_layer_peaks/` | `second_layer_ion_peaks.py` |
| `outputs/batch_merge_slab_<时间戳>/` | `batch_merge_second_layer_slab.py`（含 `batch_run.log`） |

---

## 3. 识别与定位方法

| 方法 | 典型入口 | 要点 |
|------|----------|------|
| 高斯拟合 | `python -m ion_detect`、`detect_ions` | 每峰 2D 高斯 + 椭圆；主流程 |
| Blob | `python -m ion_detect.blob_cli` | 阈值二值化 → 连通域 → 轴对齐矩形 |
| 条带 + 合并 | `merge_ion_centers.py` | 外缘条带辅助峰/COM 与 detect 按规则合并 |
| 第二层 / slab | `merge --second-layer-slab`、`second_layer_ion_peaks.py`、批量脚本 | y 直方图选行 + x 剖面 + COM |

### 3.1 高斯拟合单离子检测（`detect_ions`）

**流程**：减背景 → 匹配滤波 → 局部极大与相对阈值 → 2D 高斯拟合（可选 `refine`）。默认 **θ=0 轴对齐**（`fix_theta_zero=True`，仅代码可改）。

**入口**：推荐 `python -m ion_detect`；等价 `python .\ion_detection.py`。

**代码**

```python
from ion_detect import detect_ions, visualize, print_summary
from ion_detect import visualize_bgsub, visualize_bgsub_binarized, bgsub_binarize
# 兼容：from ion_detection import detect_ions, ...
```

**命令行**

```bash
python -m ion_detect
python -m ion_detect 0 5 -1 "::3,0:10"
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| 位置参数 `indices` | `0` | numpy 风格切片并集 |
| `--save-pos` | 关 | 写入 `outputs/IonPos/` |
| `--save-bgsub-img` | 关 | `outputs/bgsub_imgs/` |
| `--bgsub-binarize-threshold T` | 未设置 | `outputs/bgsub_binarize_imgs/`（仅可视化导出，非 blob） |
| `--bgsub-binarize-strict` | 关 | 前景 `>` 而非 `>=` |
| `--show` | 关 | 弹窗 |

**`detect_ions` 默认**（见 `ion_detect/pipeline.py`，CLI 未暴露）：`rel_threshold=0.025`，`bg_sigma=(10, 30)`，`peak_size=(5, 9)`，`fit_hw=(3, 4)`，`sigma_range=(0.3, 3.5)`，`refine=True`，`fix_theta_zero=True`。首轮 bgsub：`return_bgsub=True`。

**保存中心**

```bash
python -m ion_detect "::5" --save-pos
python -m ion_detect 0:20 --save-pos
```

每帧 `N×2` 的 `[x0, y0]`。交互调参见 [5.2 节](#52-gallerypy)。

---

### 3.2 阈值二值化 + 连通域 + 轴对齐矩形（blob）

**流程**：减背景（默认开，可用 `--no-bgsub` 关）→ 可选匹配滤波 → 浮点图二值化（默认 **`--thr-norm p95`**：椭圆 ROI 内正值子集的 P 分位为 `scale`，阈值 `T` 作用在 **`z/scale`** 上；可用 `--thr-norm none` 改为在 denoised 图上直接用 `T`）→ `label` 连通域 → 轴对齐最小外接矩形 → 可选外缘小矩形合并 → **默认**对矩形做 **y 向 split**（`--max-ysize`）与 **x 向 refine** 得到多离子位置，再按 **`--ion-dist`** 合并近邻。无单峰高斯拟合。

**实现**：`ion_detect/binarize.py`、`blob_preprocess.py`、`blob_components.py`、`blob_workflow.py`、`blob_viz.py`、`blob_ion_positions.py`。

```bash
# 与下列显式参数等价：--split --refine-x --x-profile-threshold 0.4 --y-edge-frac 0.35
#   --thr-norm p95 --thr-norm-pct 1 --threshold 40
python -m ion_detect.blob_cli 0
python -m ion_detect.blob_cli "::5" --threshold 30 --matched-filter
python -m ion_detect.blob_cli 0 --no-split --no-refine-x --thr-norm none --threshold 50
```

| 选项 | 默认 | 说明 |
|------|------|------|
| `--threshold T` | `40` | 二值化阈值；`--thr-norm` 为 `p95` / `p95_all` 时作用在 **`z/scale`**（前景仍为 **≥T** 的比较语义，见工作流实现） |
| `--thr-norm` | `p95` | `none`：在 denoised 图上直接用 `T`；`p95`：椭圆 ROI 内**仅正值**的 P 分位作 `scale`；`p95_all`：ROI 内**全体有限像素的有符号** P 分位作 `scale`（不是对绝对值取分位） |
| `--thr-norm-pct P` | `1` | 与 `p95` / `p95_all` 配合，分位数 **P∈[1,100]** |
| `--split` / `--no-split` | **开** | y 向条带分割后求平衡位置；`--no-split` 仅用矩形中心 |
| `--max-ysize Y` | `9` | 与 split 配合：仅当矩形 y 跨度更大时才细分 |
| `--refine-x` / `--no-refine-x` | **开** | 子带内按列剖面细化 x、可多离子；`--no-refine-x` 关闭 |
| `--x-profile-threshold P` | `0.4` | x 列掩膜阈值（与条带内 y 向二值占有率比较；`--x-profile-rel-to-max` 时改为相对 `max(col)`） |
| `--x-profile-rel-to-max` | 关 | 列掩膜 `col_mean > P * max(col_mean)` |
| `--y-edge-frac` | `0.35` | 外缘条带参数 F（小矩形合并等，与 `outer_y_edge_strip_masks` 一致） |
| `--ion-dist D` | `5` | 全部位置求出后，欧氏距离 `< D` 合并；**≤0** 关闭 |
| `--no-merge-small-rects` | 关 | 关闭椭圆 y 外缘带内薄小矩形与 AABB 合并 |
| `--min-edge-ysize` | `5` | 参与上述合并的矩形 y 向边长上限条件 |
| `--no-merge-band-clip-ellipse` | 关 | 条带掩膜不按椭圆裁剪 |
| `--no-pre-merge-drop` | 关 | merge 前不剔除两轴跨度均 ≤1 的矩形 |
| `--log` | 关 | 追加 TSV 至 `outputs/blob/merge_split.log` |
| `--hist` | 关 | 各帧最终离子数直方图 → `outputs/blob/hist_merge_split.png` |
| `--plot-pixel-hist` | 关 | 椭圆内亮度直方图（标 `T`）→ `outputs/pixel_hist/` |
| `--no-bgsub` | 关 | 不减高斯背景 |
| `--matched-filter` | 关 | 与 `detect_ions` 相同匹配滤波 |
| `--connectivity` | `4` | `4` 或 `8` |
| `--min-area-pixels N` | `1` | 忽略更小连通域 |
| `--show` | 关 | 弹窗：先打开与 `viz` 附录类似的**空域 brightness 分布图**（`RdBu_r`/横向色标，叠 boundary 与矩形），再显示双栏 workflow |
| `--data-dir` | `DEFAULT_DATA_DIR` | 数据目录 |

输出：`outputs/blob_connected/` 双栏 PNG：**上**为实际参与阈值的浮点图（线性 brightness 色标，文框标明 bgsub / matched filter 是否开启及显示范围），**下**为二值图与矩形；**估计得到 boundary 时，上下栏均绘制**晶格椭圆（cyan 虚线）。图上洋红等标记为 **split / refine / `ion-dist` 之后**的最终离子位置（默认启用 split 与 refine）。**区别**：`python -m ion_detect` 的 `--bgsub-binarize-threshold` 只导出 bgsub 与 mask PNG，**不**跑连通域与矩形。

---

### 3.3 外缘条带与混合合并（`merge_ion_centers.py`）

在 **`detect_ions`** 与 **4.1 节** 同一套外缘 y 条带几何之间合并，单 PNG 出图（边界椭圆、上下缘线、`edge-x-range` 竖线、各中心）。

**规则**

- **椭圆内** + **上下外缘条带** + **x ∈ [`--edge-x-range`]**（默认 `250 750`）：保留条带 **辅助峰 + 列向 COM**，**丢弃**该域内 `ion_detect` 中心。
- **其余**：仅 `detect_ions`；条带在 x 段外的峰不参与。
- **`--ion-dist`**（默认 `4`）：detect 与 strip 距离 **≤** 阈值反复并最近一对，中心取平均 → `fused_mean`（洋红）。**≤0** 关闭；过大易误并相邻格点。

```bash
python merge_ion_centers.py 0
python merge_ion_centers.py 0 --edge-x-range 250 750 --y-edge-frac 0.25 ^
  --y-fit-frac 0.35 --peak-dist 5 --ion-dist 4
python merge_ion_centers.py ::10
```

| 选项 | 默认 |
|------|------|
| `--edge-x-range` | `250 750` |
| `--y-edge-frac` | `0.25` |
| `--y-fit-frac` | `0.35` |
| `--peak-dist` | `5` |
| `--col-metric` | `mean` |
| `--strip-center-mode` | `com` |
| `--ion-dist` | `4` |
| `--preprocess` | `raw`（条带聚合图；`detect` 始终原图） |
| `--data-dir` | `DEFAULT_DATA_DIR`；PNG → `outputs/ion_centers_merged/` |

**`--second-layer-slab`**：由 y 直方图多峰定水平条带，条带内改用第二层行寻峰（`second_layer_core`）。多帧单次调用 = **合并所有帧中心**的画直方图；**每帧独立**请用 `batch_merge_second_layer_slab.py`。

| 选项 | 默认 |
|------|------|
| `--profile-x-range` | 未传时常 `300–600` |
| `--second-layer-hist-prominence` | `5`（`second_layer_ion_peaks` 默认 `10`） |
| `--second-layer-prof-prominence-frac` | `0.08` |
| `--second-layer-prof-peak-distance` | `4` |
| `--second-layer-y-halfwin` | `3` |
| `--second-layer-com-neighbor-cols` | `1` |
| `--second-layer-line-first/second/third` | `1` / `2` / `3` |
| `--second-layer-y-cut-pad` | `1` |

---

### 3.4 第二层诊断与批量 slab

**`second_layer_ion_peaks.py`**：与 `merge_ion_centers` 相同流程得合并中心后，**全帧中心合并** → y 直方图按 `--line-id` 选行 → x 剖面 → COM。默认 `outputs/second_layer_peaks/`。

```bash
python second_layer_ion_peaks.py 0 --line-id 2
python second_layer_ion_peaks.py ::10 --hist-prominence 10
```

**`batch_merge_second_layer_slab.py`**：**每帧单独**直方图与 slab；输出 `outputs/batch_merge_slab_<时间戳>/`。

```bash
python batch_merge_second_layer_slab.py
python batch_merge_second_layer_slab.py 800 --data-dir D:/data/one_run
```

**`plot_batch_log_ion_histogram.py`**：解析 `离子数=…` 日志行 → 直方图（`outputs/histogram/…`）。

```bash
python plot_batch_log_ion_histogram.py path/to/batch_run.log --bin-width 2
```

---

## 4. 诊断与一维剖面

用于理解上下缘、分层与预处理；**不替代** 3.1–3.4 的主定位定义。

### 4.1 外缘条带列剖面（`edge_strip_profile.py`）

几何与 **`|y-cy|/b ≥ 1-F`** 截线一致；上下轴对齐条带内按列 `sum/mean/max`。`peel` / `peel_bgsub` 须首轮 `detect_ions` 成功。

**实现**：`ion_detect.edge_strip_profile_analysis`、`edge_strip_profile_viz`；根脚本为 CLI。

**叠图线型**：红虚线+点 = 剖面全局最大（细化）；番茄/绿虚线 = 多峰示意（`--peak-dist` + prominence）。

```bash
python edge_strip_profile.py 0
python edge_strip_profile.py 0 --preprocess bgsub --y-edge-frac 0.25
python edge_strip_profile.py 0 --preprocess peel --plot-peel
python edge_strip_profile.py 0 --peak-col-gallery --y-fit-frac 0.35
python edge_strip_profile.py ::20
```

| 选项 | 默认 |
|------|------|
| `--y-edge-frac` | `0.25` |
| `--preprocess` | `raw` |
| `--col-metric` | `mean` |
| `--peak-dist` | `5` |

聚合与掩膜：`ion_detect.edge_strip.outer_y_edge_column_profiles`、`outer_y_edge_strip_masks`。详见 `python edge_strip_profile.py -h`。

### 4.2 y 向分层（`y_layer_profile.py`）

固定 **x** 条带内按行对 x 求和，**y** 为横轴。`--preprocess`：`raw` / `bgsub` 仅。

```bash
python y_layer_profile.py 0
python y_layer_profile.py 0 --x-range 400:601 --preprocess bgsub --show
```

输出：`outputs/y_layer_profiles/`。

---

## 5. 交互浏览与统计分析

### 5.1 脚本与识别路径对照

| 脚本 | 依赖的主要识别 / 数据来源 |
|------|---------------------------|
| `gallery.py` | `detect_ions` |
| `stretching_analysis.py` | `detect_ions`（σ、振幅 vs y） |
| `hist_sigma_xy_per_frame.py` | `detect_ions`，每帧 σx/σy 直方图 → `outputs/histogram/sx|sy/` |
| `hist_sigma_near_boundary_major_axis.py` | `detect_ions`，近长轴离子 σ 直方图 |
| `hist_y0_A_deconv_near_major_axis.py` | `detect_ions` + ROI 反卷积与 `A(y)` 拟合 |
| `merge_ion_centers.py` | `detect_ions` + 外缘条带 |
| `second_layer_*.py`、`batch_merge_second_layer_slab.py` | 合并或第二层几何 |
| `dist.py` | 已保存的 `outputs/IonPos/*.npy` |

### 5.2 gallery.py

交互浏览 `.npy` 并叠加 `detect_ions`（英文 UI）。默认 `rel=0.025`，**Two-pass ref.** 默认开。

```bash
python .\gallery.py
```

改参数后点 **Detect**；结果按 `(帧, 参数)` 缓存。

### 5.3 `stretching_analysis.py`

`y_rel` 与 `sigma_minor/sigma_major`、`amplitude` 的统计拟合。

```bash
python .\stretching_analysis.py
python .\stretching_analysis.py -n 10
python .\stretching_analysis.py -n 100 --ratio-fit quartic --amp-fit gaussian
```

**默认**：`-n` 为 `100`；`--ratio-fit` / `--amp-fit` 均为 `quadratic`（可选 `quartic`、`gaussian`）。`quadratic` 时仅用 `|y_rel| <= 40` 做点。输出例：`outputs/stretch_analysis/…`、`outputs/amp_y_fit/…`。

### 5.4 `dist.py`

读 `outputs/IonPos/*.npy`，构型间标准化距离（对 `cdist` 行取最小再平均）与直方图。

```bash
python .\dist.py
python .\dist.py --count 100
```

默认 `--pos-dir`=`outputs/IonPos`；PNG → `outputs/histogram/cdist_hist_{构型数}.png`。`--bins` 默认 `50`。

### 5.5 `vis_selected_npy.py`

按 stem 从数据目录导出灰度 PNG 或预览；默认 `outputs/npy_plots/`。

```bash
python .\vis_selected_npy.py 20260305_005542 20260305_010000
python .\vis_selected_npy.py --one-figure --cmap inferno 20260305_005542
```

详见 `python vis_selected_npy.py -h`。

### 5.6 `project_info.py`

扫描默认数据目录，摘要 + `outputs/dataset_viz/` 示例图。

```bash
python .\project_info.py
```

### 5.7 其他

- **`statistic.ipynb`**：数据集统计示例（按需）。
- **`hist_sigma_xy_per_frame.py`**、**`hist_sigma_near_boundary_major_axis.py`**、**`hist_y0_A_deconv_near_major_axis.py`**：基于 `detect_ions` 的专项统计图（见各文件 docstring）。

---

## 6. ion_detect 包内模块分工

| 模块 | 职责 |
|------|------|
| `ion_detect.pipeline` | `detect_ions` 端到端 |
| `ion_detect.gaussian` | 2D 高斯与剥离叠加 |
| `ion_detect.boundary` | 晶格椭圆边界 |
| `ion_detect.preprocess` | 匹配滤波、局部极大候选 |
| `ion_detect.fitting` | 单峰拟合与精修 |
| `ion_detect.peel` | `y_edge_band_thresholds` |
| `ion_detect.edge_strip` | 外缘条带按列聚合 |
| `ion_detect.edge_strip_profile_analysis` | 条带列剖面分析 |
| `ion_detect.edge_strip_profile_viz` | 条带总览与交互 |
| `ion_detect.viz` | `visualize`、`print_summary`、bgsub/二值可视化 |
| `ion_detect.cli_helpers` | 索引解析 |
| `ion_detect.binarize` | `bgsub_binarize` / `bgsub_binarize_u8` |
| `ion_detect.blob_*` | 预处理、二值、连通域、工作流、可视化 |
| `ion_detect/__main__.py` | `python -m ion_detect` |

根目录 **`second_layer_core.py`**：y 直方图 / 剖面 / COM / slab，供 merge 与批量脚本共用。

---

## 7. 推荐工作流

1. **检测并导出坐标**：`python -m ion_detect "::5" --save-pos`
2. **单帧交互调参**：`python .\gallery.py`
3. **y 向形变统计**：`python .\stretching_analysis.py -n 100`
4. **构型距离**：`python .\dist.py --count 100`
5. **（可选）剖面**：`edge_strip_profile.py`、`y_layer_profile.py`
6. **（可选）合并外缘**：`merge_ion_centers.py`（含 `--second-layer-slab`）
7. **（可选）第二层 / 批量 slab**：`second_layer_ion_peaks.py`、`batch_merge_second_layer_slab.py`

---

## 8. 调参建议

- **漏检 / 假峰**：`gallery` 调 `rel`，或代码 `detect_ions(..., rel_threshold=...)`；可调 `fit_hw`、`peak_size`（代码）。
- **噪点多**：提高 `rel_threshold`。
- **旋转椭圆 PSF**：`detect_ions(..., fix_theta_zero=False)`。
- **形变分析**：增大 `-n`，对比 `quadratic` / `quartic` / `gaussian`。
- **合并中心**：略减 `--peak-dist` 可补外缘峰；`--ion-dist` 过大易误并；`--edge-x-range` 覆盖可靠列、避开圆角误判。
- **第二层**：`merge` 与 `second_layer_ion_peaks` 的 `hist-prominence` 默认不同（5 vs 10），对比时统一；勿混淆「单次 merge 多帧直方图」与「batch 每帧独立」。
- **Blob（`blob_cli`）**：默认已启用 `p95` 阈值尺度（`--thr-norm-pct` 默认 `1`）、`--threshold 40`、`--split` / `--refine-x` 与较大的 `--y-edge-frac`（`0.35`）。若需接近早期「仅连通域矩形、原始浮点阈值」行为，可试 `--no-split --no-refine-x --thr-norm none` 并显式设 `--threshold`。

---

## 9. 项目文件索引

| 文件 | 说明 |
|------|------|
| `output_paths.py` | 默认数据目录与各 `OUT_*` |
| `ion_detect/` | 检测核心包；`python -m ion_detect`、`python -m ion_detect.blob_cli` |
| `ion_detection.py` | 兼容再导出与 `python ion_detection.py` |
| `gallery.py` | 交互浏览 |
| `stretching_analysis.py` | y 向形变 / 振幅拟合 |
| `dist.py` | 构型距离直方图 |
| `vis_selected_npy.py` | 指定帧 PNG |
| `edge_strip_profile.py` | 外缘条带列剖面 CLI |
| `y_layer_profile.py` | 固定 x 条带 y 向积分 |
| `merge_ion_centers.py` | detect + 条带合并与可选 slab |
| `second_layer_core.py` | 第二层几何内核 |
| `second_layer_ion_peaks.py` | 第二层独立诊断 |
| `batch_merge_second_layer_slab.py` | 批量每帧 slab |
| `plot_batch_log_ion_histogram.py` | 批日志离子数直方图 |
| `hist_sigma_xy_per_frame.py` | 每帧 σx/σy 直方图 |
| `hist_sigma_near_boundary_major_axis.py` | 近长轴 σ 直方图 |
| `hist_y0_A_deconv_near_major_axis.py` | 反卷积 + A(y) 拟合 |
| `project_info.py` | 数据集概览 |
| `statistic.ipynb` | Notebook 示例 |
| `.gitignore` | 忽略规则 |
