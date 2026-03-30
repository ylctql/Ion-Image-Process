# IonImage

离子晶格图像处理与分析工具集，包含：

- 单帧离子检测与椭圆拟合：核心实现为 **`ion_detect` 包**；根目录 `ion_detection.py` 为兼容入口（`detect_ions` / 命令行）
- 交互式浏览与手动触发检测（`gallery.py`）
- 形变与亮度随 `y` 方向统计拟合（`stretching_analysis.py`）
- 固定 **x** 条带内按行积分，观察 **y** 向分层/条纹（`y_layer_profile.py`）
- 中心区域 `ion_detect` 与上下缘条带 COM **合并离子中心**（`merge_ion_centers.py`）；可选 **第二层 y 直方图峰行** 条带内替换（`--second-layer-slab`，算法见 `second_layer_core.py`）
- **第二层诊断**：合并中心 y 直方图选峰 → 三行 x 剖面寻峰 → 列向 COM（`second_layer_ion_peaks.py`，默认输出 `outputs/second_layer_peaks/`）
- **批量**每帧独立执行与 `merge_ion_centers --second-layer-slab` 等价的 slab 流程，并汇总图（`batch_merge_second_layer_slab.py`）；从运行日志解析每帧离子数并画直方图（`plot_batch_log_ion_histogram.py`）
- 多构型距离统计与直方图（`dist.py`）
- 数据集整体统计与示例图（`project_info.py`）
- 指定若干帧导出矩阵 PNG（`vis_selected_npy.py`）
- 可选：减背景图导出与阈值二值化 PNG（`python -m ion_detect` 的 `--save-bgsub-img` / `--bgsub-binarize-threshold`）；Notebook 统计示例（`statistic.ipynb`）

检测流程概览：**减背景 → 匹配滤波 → 局部极大与阈值 → 2D 高斯拟合**（可选两阶段 `refine` 收紧过大 σ；命令行未单独暴露，可在代码中调用 `detect_ions` 修改 `rel_threshold`、`fit_hw` 等）。

---

## 1. 环境依赖

建议 Python 3.10+。

安装依赖：

```bash
pip install numpy scipy matplotlib
```

---

## 2. 数据与目录约定

默认目录（相对项目根目录）：

- 输入图像：`20260305_1727/*.npy`
- **生成结果统一在 `outputs/` 下**（**所有默认保存路径与默认数据目录**均集中在根目录 `output_paths.py`；命令行不再提供修改输出目录的选项，便于约定一致）：
  - `outputs/ion_detect_imgs/`：离子椭圆叠加 PNG（`python -m ion_detect` 默认）
  - `outputs/bgsub_imgs/`：减高斯背景后的 signal 图（`--save-bgsub-img`）
  - `outputs/bgsub_binarize_imgs/`：bgsub 与二值 mask 成对 PNG（`--bgsub-binarize-threshold`）
  - `outputs/IonPos/`：离子中心 `N×2` 的 `.npy`
  - `outputs/amp_y_fit/`：y 向振幅拟合系数等（如 `amp_vs_y_coef_10.npy`）
  - `outputs/stretch_analysis/`：形变分析图
  - `outputs/histogram/`：构型距离直方图
  - `outputs/dataset_viz/`：`project_info.py` 生成的数据集统计与热力图等
  - `outputs/npy_plots/`：`vis_selected_npy.py` 默认输出
  - `outputs/edge_strip_profiles/`：`edge_strip_profile.py` 默认 PNG（常量 `OUT_EDGE_STRIP`）
  - `outputs/y_layer_profiles/`：`y_layer_profile.py` 默认 PNG（常量 `OUT_Y_LAYER_PROFILE`）
  - `outputs/ion_centers_merged/`：`merge_ion_centers.py` 默认 PNG（常量 `OUT_ION_CENTERS_MERGED`）
  - `outputs/second_layer_peaks/`：`second_layer_ion_peaks.py` 默认（直方图、剖面图、叠图；常量 `OUT_SECOND_LAYER_PEAKS`）
  - `outputs/batch_merge_slab_<时间戳>/`：`batch_merge_second_layer_slab.py` 每次运行的输出根目录（由 `output_paths.new_batch_merge_slab_run_dir()` 创建；同目录下写 `batch_run.log`）
  - `outputs/blob_connected/`：`python -m ion_detect.blob_cli` 双栏 PNG（`OUT_BLOB_CONNECTED`）

> `20260305_1727` 中每个 `npy` 被视为一帧图像（若为 3D 会按脚本逻辑处理）。

---

## 3. 核心脚本

## 离子检测（`ion_detect` / `ion_detection.py`）

对指定帧做离子检测，输出椭圆叠加图；默认 **θ=0 轴对齐拟合**（`fix_theta_zero=True`，仅在代码中可改为 `False`）。支持索引切片并集、保存中心与 bgsub / 二值化导出等。

### 代码中调用

推荐直接使用包入口，便于与模块结构一致：

```python
from ion_detect import detect_ions, visualize, print_summary
# 可选：bgsub / 二值化可视化
from ion_detect import visualize_bgsub, visualize_bgsub_binarized, bgsub_binarize
```

历史脚本也可继续使用：

```python
from ion_detection import detect_ions, visualize_bgsub, bgsub_binarize  # 兼容入口再导出
```

### 3.1 命令行基础用法

以下两种方式等价；**推荐使用** `python -m ion_detect`，避免依赖当前工作目录对单文件的解析。

```bash
# 默认处理第 0 帧
python -m ion_detect
# 或
python .\ion_detection.py

# 多个索引（整数、负数、numpy 风格切片并集）
python -m ion_detect 0 5 -1 "::3,0:10"
```

### 3.1.1 `python -m ion_detect` 选项与默认值

| 选项 | 默认值 | 说明 |
|------|--------|------|
| 位置参数 `indices` | `0` | 可多段并集 |
| `--save-pos` | 关闭 | 离子中心写入 `outputs/IonPos/`（与 `output_paths.OUT_ION_POS` 一致，不可通过 CLI 改目录） |
| `--save-bgsub-img` | 关闭 | 保存与首轮检测一致的减背景 signal PNG 至 `outputs/bgsub_imgs/` |
| `--bgsub-binarize-threshold T` | 未设置 | 指定则对 bgsub 二值化并保存至 `outputs/bgsub_binarize_imgs/`（`*_bgsub.png` 与 mask PNG） |
| `--bgsub-binarize-strict` | 关闭 | 与上一项合用：前景为 `>` 而非 `>=` |
| `--show` | 关闭 | 弹窗显示主图及（若有）bgsub / 二值化；多帧时每帧关窗后继续 |

**未单独暴露给 CLI、但 `detect_ions` 使用的默认**（见 `ion_detect/pipeline.py`）：`rel_threshold=0.025`，`bg_sigma=(10, 30)`，`peak_size=(5, 9)`，`fit_hw=(3, 4)`（半窗口），`sigma_range=(0.3, 3.5)`，`refine=True`，`fix_theta_zero=True`。可在代码中调用 `detect_ions(...)` 修改；若需要首轮减背景图，设 `return_bgsub=True`（返回值 `(..., bgsub)`）。

### 3.2 保存离子中心

```bash
python -m ion_detect "::5" --save-pos
python -m ion_detect 0:20 --save-pos
```

每帧保存为同名 `npy`，内容为 `N x 2` 的 `[x0, y0]`，目录固定为 `outputs/IonPos`。

### 3.3 y 向外缘条带列轮廓（诊断上下缘漏检）

与上表 **`|y-cy|/b ≥ 1-F`** 同一套截线（`y_below = cy - (1-F)*b`，`y_above = cy + (1-F)*b`）。在 **上下两侧各** 取一轴对齐矩形：**y** 从椭圆在 **竖直方向的极点**（`cy ∓ b`，与 `boundary` 的 y 半轴一致）到对应截线；**x** 取该截线与椭圆边界的两交点之间的弦（`|x-cx| ≤ a·√(1 - ((y_cut-cy)/b)²)`）。对条带掩膜内像素按列做 **聚合**（`--col-metric`：`sum` / `mean` / `max`）得到 1D 曲线，脚本会保存曲线图并用抛物线插值细化 **峰值 x**（控制台亦打印）。

**实现结构**：列剖面核心算法在 `ion_detect.edge_strip_profile_analysis`，总览图与「逐列 y 剖面」交互窗口在 `ion_detect.edge_strip_profile_viz`；根目录 **`edge_strip_profile.py`** 仅为命令行入口（用法见其文件头注释）。

独立工具 **不加** 检测 CLI 前缀。`raw` / `bgsub` 时在 **`image -` 高斯背景** 上调用 `estimate_crystal_boundary`（与 `detect_ions` 一致）。**`peel` / `peel_bgsub`** 时先 `detect_ions` 得首轮离子，再用 `原图 − Σ拟合高斯` 得残差（`peel_bgsub` 为残差再减高斯背景）；**须首轮能检出离子**。**`--preprocess`** 只决定送入条带 **按列聚合** 的二维图（与 `--col-metric` 正交）。

**图上两类竖线（与实现一致）**：

- **红虚线 + 红点**（子图 `Top/Bottom strip`）：整条 1D 剖面的 **全局最大**，经三点抛物线细化（`outer_y_edge_column_profiles`），**不受** `--peak-dist` 影响。
- **番茄红 / 绿色虚线**（叠在顶栏灰度图上）：上下条带剖面上的 **多峰示意**；由严格离散局部极大经 **`--peak-dist`** 最小间距筛选；过近的一对去低 **prominence** 峰（`scipy.signal.peak_prominences`；平手取 **y 较大** 者）。

**辅助峰列的 y 向诊断（`--peak-col-gallery` / `--plot-center`）**：在轮廓算法得到的每个辅助峰 **列** 上，沿 **y** 采样条带内像素（可用 `--y-fit-frac` 单独加宽约 `--y-edge-frac` 语义，条带 1D 轮廓与辅助峰 **x** 仍由 `--y-edge-frac` 决定）；列向 profile **默认**为 **x−1,x,x+1 三列强度之和**。中心标点：`--plot-center` 可选 `fit`（默认，单/双高斯或 `--prominence`）、`com`（亮度加权质心）、`com_fit`（质心后取最近严格局部极大）；`--double-peak-fit` 强制双高斯曲线拟合；`--prominence [MIN]` 优先用 prominence 选两峰平均。**`--peak-col-gallery`** 会额外打开可滑动切换峰序号、单选项切换上/下条带的交互窗口，并隐含 **`--show`**（若在 IDE 内联后端无响应，请在系统终端设置 GUI 后端，例如 PowerShell：`$env:MPLBACKEND='TkAgg'; python edge_strip_profile.py 0 --peak-col-gallery`）。

```bash
python edge_strip_profile.py 0
python edge_strip_profile.py 0 5 --y-edge-frac 0.25 --preprocess bgsub
python edge_strip_profile.py 0 --preprocess peel
python edge_strip_profile.py 0 --preprocess peel_bgsub
python edge_strip_profile.py 0 --preprocess peel --plot-peel
python edge_strip_profile.py 0 --col-metric max --peak-dist 3
python edge_strip_profile.py 0 --peak-col-gallery --y-fit-frac 0.35
python edge_strip_profile.py 0 --plot-center
python edge_strip_profile.py 0 --plot-center com --prominence 0.5
python edge_strip_profile.py ::20
# 全部选项：python edge_strip_profile.py -h
```

| 选项 | 含义 | 默认 |
|------|------|------|
| 位置参数 `indices` | 与 `python -m ion_detect` 相同 numpy 风格切片并集 | `0` |
| `--y-edge-frac F` | 外缘条带参数 F，截线 `|y-cy|/b = 1-F` | `0.25` |
| `--preprocess` | `raw`；`bgsub`；`peel`（首轮拟合高斯剥离残差）；`peel_bgsub`（残差再减高斯背景）；`peel*` 须首轮检出离子 | `raw` |
| `--col-metric` | 条带掩膜内按列 `sum` / `mean` / `max`；`mean` 整除该列掩膜像素数；`max` 在无掩膜列为 NaN；绘图时 `mean`/`max` 在无掩膜列不连线 | `mean` |
| `--plot-peel` | 仅 `peel` / `peel_bgsub`：顶栏 `imshow` 用 peel（或 peel+bgsub）图；**默认** 顶栏为原始载入的 `npy` | 关 |
| `--peak-dist D` | 见上文「叠图虚线」：相邻候选峰在 x 上须 **>** `D`（px）；`D≤0` 则标出全部严格局部极大 | `5` |
| `--peak-col-gallery` | 辅助峰列上沿 y 的 profile + 一维高斯拟合，滑动条切换峰；隐含 `--show` | 关 |
| `--plot-center [MODE]` | 顶图标注辅助峰列的 y 中心；省略 `MODE` 为 `fit`；可选 `com`、`com_fit` | 不标注 |
| `--double-peak-fit` | 列 y-profile 用双高斯（与 gallery / plot-center 联用） | 关 |
| `--prominence [MIN]` | prominence 筛选后取最多两峰平均作为中心；单独写时 `MIN=0` | 未启用 |
| `--y-fit-frac Ff` | 列 y 采样/拟合条带宽度（F 语义同 `--y-edge-frac`）；省略则与 `--y-edge-frac` 一致 | 与 F 相同 |
| `--no-clip-ellipse` | 条带不按椭圆再裁剪（整块轴对齐条带矩形参与掩膜/聚合） | 否 |
| `--show` | 处理完后 **交互弹窗**（仍会写入 `outputs/edge_strip_profiles/`） | 关 |
| `--data-dir` | `.npy` 目录 | `output_paths.DEFAULT_DATA_DIR` |

条带几何与按列聚合：`ion_detect.edge_strip.outer_y_edge_column_profiles`（参数 `col_metric`，默认 `mean`；`outer_y_edge_strip_masks` 亦导出在 `ion_detect` 包根）。

### 3.4 合并离子中心（`merge_ion_centers.py`）

在 **`detect_ions`（默认 θ=0 轴对齐）** 与 **3.3 节**（外缘条带）同一套几何之间做规则合并，输出带图例的单张 PNG（灰度图、晶格边界椭圆、上下缘分界线、`edge-x-range` 竖线、各中心点）。

**区域规则（与实现一致）**

- 在 **椭圆内**、**上下外缘 y 条带**（与 `outer_y_edge_strip_masks` 一致，由 `--y-edge-frac` 控制）且 **x** 落在 **`--edge-x-range X0 X1`**（默认 `250 750`）内：仅保留该 x 段上的 **条带辅助峰 + 列向 COM**（由 `fitted_xy_for_auxiliary_strip_peaks`）；**丢弃**落在此域内的 `ion_detect` 中心。
- **其余**（含外缘行但在 x 段之外）：仅保留 `ion_detect`；条带在该 x 段外的峰不参加合并。
- **`--ion-dist`**（默认 `4`，像素）：对 **detect** 与 **strip_top / strip_bot** 成对、欧氏距离 **≤** 该阈值的，反复合并 **最近** 的一对，**中心取坐标算术平均**，`sigma` / `amplitude` 等仍取自 detect；合并后 `source=fused_mean`（图上为洋红色）。**≤0** 关闭此步。**注意**：阈值若大于约半格距，可能误并相邻两个离子，请按数据微调。

条带侧列向 COM **默认**使用 x−1,x,x+1 三列和（与条带诊断工具一致）；`--y-fit-frac` 默认 `0.35`，`--strip-center-mode` 默认 `com`。`--preprocess` 仅作用于送入条带 **按列聚合** 的二维图，`detect_ions` 始终用原图流水线。

```bash
python merge_ion_centers.py 0
python merge_ion_centers.py 0 --edge-x-range 250 750 --y-edge-frac 0.25 ^
  --y-fit-frac 0.35 --peak-dist 5 --ion-dist 4
python merge_ion_centers.py ::10 --out outputs/ion_centers_merged
# 全部选项：python merge_ion_centers.py -h
```

| 选项 | 含义 | 默认 |
|------|------|------|
| 位置参数 `indices` | 与 `python -m ion_detect` 相同 numpy 风格切片并集 | `0` |
| `--edge-x-range X0 X1` | 上下缘「条带优先」的 x 闭区间；外侧外缘行仍只用 detect | `250 750` |
| `--y-edge-frac F` | 外缘条带几何 F（同 3.3 节 `--y-edge-frac`） | `0.25` |
| `--y-fit-frac Ff` | 列 y 向 COM 采样条带宽度（同 3.3 节） | `0.35` |
| `--peak-dist D` | 条带 1D 剖面上辅助峰最小间距（像素） | `5` |
| `--col-metric` | `sum` / `mean` / `max` | `mean` |
| `--strip-center-mode` | `com` / `com_fit` / `fit` | `com` |
| `--ion-dist PX` | detect 与 strip 中心距离 ≤ PX 时合并为均值；≤0 关闭 | `4` |
| `--preprocess` | 送入条带的图：`raw` / `bgsub` / `peel` / `peel_bgsub`（后两者须首轮能检出离子） | `raw` |
| `--no-clip-ellipse` | 条带不按椭圆裁剪 | 否 |
| `--show` | 弹窗显示（仍会写 PNG） | 关 |
| `--data-dir` | `.npy` 目录 | `output_paths.DEFAULT_DATA_DIR`（PNG 固定 `outputs/ion_centers_merged/`） |

**可选 `--second-layer-slab`**：在由「第一 / 第二 / 第三个 y 直方图峰」确定的水平条带内，丢弃原 merge 结果并改用第二层行寻峰（与 `second_layer_core` 一致）；可与 **`--profile-x-range`**（默认未指定时为 `300–600`，与 `merge` 的 `--edge-x-range` 默认 `250–750` 独立）及下列参数联用。多帧一次调用时，y 直方图为**所选帧合并中心**的并集；若需**每帧单独**直方图与 slab，请用 `batch_merge_second_layer_slab.py`。

| 选项 | 含义 | 默认 |
|------|------|------|
| `--second-layer-slab` | 启用条带内 second-layer 替换及与近邻的 `ion-dist` 融合（`source=fused_second_layer`） | 关 |
| `--profile-x-range X0 X1` | 第二层 x 剖面与替换条带的列范围（像素） | `300–600`（未传时） |
| `--second-layer-hist-prominence` | 合并中心 y 直方图 `find_peaks` 的 prominence | `5`（独立脚本 `second_layer_ion_peaks` 默认 `10`） |
| `--second-layer-prof-prominence-frac` | x 剖面 prominence = 该比例 × 剖面最大值 | `0.08` |
| `--second-layer-prof-peak-distance` | x 向峰最小间距（像素） | `4` |
| `--second-layer-y-halfwin` | 列向 COM 的 y 半窗（像素） | `3` |
| `--second-layer-com-neighbor-cols N` | COM 时在峰列两侧各并 N 列 | `1` |
| `--second-layer-line-first` / `-second` / `-third` | y 直方图峰序号（1 起，按峰位 y 排序）；第二与第三峰 bin 中心的中点（减 `--second-layer-y-cut-pad`）为替换条带上界 | `1` / `2` / `3` |
| `--second-layer-y-cut-pad` | 上界在中点基础上再减去的像素（略大则更保守） | `1` |

核心函数：`merge_centers_hybrid`、`fuse_detect_strip_by_distance`（见脚本内文档字符串）；第二层几何与替换见 `second_layer_core.py`。

### 3.5 y 向分层轮廓（`y_layer_profile.py`）

在图像上固定 **x** 向半开区间（默认列 **`400:601`**，即 400…600），对选定 **y** 行集合在条带内 **按行对 x 求和**，得到 **y 为横轴** 的 1D 曲线，便于观察竖直方向分层/条纹。

- **Boundary**：在 `image −` 高斯背景上与 `detect_ions` / `edge_strip_profile` 相同，调用 `estimate_crystal_boundary`。
- **`--y-range`**：逗号分隔的多段 numpy 切片并取并集（如 `0:10,20:30`）；**省略或空串** 时使用椭圆在竖直方向的整数行包络 `[⌊cy−b⌋, ⌈cy+b⌉+1)`。
- **`--preprocess`**：仅 `raw` 或 `bgsub`（无 peel 模式）。

```bash
python y_layer_profile.py 0
python y_layer_profile.py 0 --y-range 0:10,20:30
python y_layer_profile.py 0 --x-range 400:601 --preprocess bgsub
python y_layer_profile.py 0 --show
```

| 选项 | 含义 | 默认 |
|------|------|------|
| 位置参数 `indices` | 与 `ion_detect` 相同 numpy 风格切片并集 | `0` |
| `--x-range` | x 向半开切片字符串 | `400:601` |
| `--y-range` | y 向多段切片并集；空则为椭圆竖直包络 | 空 |
| `--preprocess` | `raw` / `bgsub` | `raw` |
| `--show` | 交互显示 | 关 |
| `--data-dir` | 数据目录 | `output_paths.DEFAULT_DATA_DIR`（PNG 固定 `outputs/y_layer_profiles/`） |

### 3.6 包内模块分工（便于维护与二次开发）

| 模块 | 职责 |
|------|------|
| `ion_detect.pipeline` | `detect_ions` 端到端流程 |
| `ion_detect.gaussian` | 2D 高斯模型与剥离核叠加 |
| `ion_detect.boundary` | 晶格椭圆边界估计与候选过滤 |
| `ion_detect.preprocess` | 匹配滤波、局部极大候选 |
| `ion_detect.fitting` | 单峰拟合与两阶段精修 |
| `ion_detect.peel` | `y_edge_band_thresholds`（外缘条带分界线，供 `edge_strip` / 诊断） |
| `ion_detect.edge_strip` | y 向外缘条带按列聚合（sum/mean/max）与主峰值（与 peel 同一 F 几何） |
| `ion_detect.edge_strip_profile_analysis` | 条带列剖面辅助分析（y 向列 profile、高斯/prominence 等） |
| `ion_detect.edge_strip_profile_viz` | 条带总览图 `plot_edge_strip_dashboard`、逐列交互 `show_peak_column_gallery` |
| `ion_detect.viz` | `visualize`、`print_summary` |
| `ion_detect.cli_helpers` | 命令行索引解析 |
| `ion_detect.binarize` | `bgsub_binarize` / `bgsub_binarize_u8`（与 `detect_ions` 首轮 signal 同量纲） |
| `ion_detect.blob_*` | 连通域 + 轴对齐矩形工作流；CLI：`python -m ion_detect.blob_cli`（输出 `outputs/blob_connected/`） |
| `second_layer_core`（根目录模块） | y 直方图选峰、三行 x 剖面、列 COM、`ions_from_second_layer_row`、条带替换等 |
| `output_paths`（根目录模块） | **集中定义** `DEFAULT_DATA_DIR` 与各 `OUT_*`（含 `blob_connected`、直方图、批量 slab 时间戳目录工厂函数等） |

### 3.7 第二层 y 行诊断（`second_layer_ion_peaks.py`）

先按与 `merge_ion_centers.py` 相同的检测/条带/合并参数，对所选帧逐帧得到合并中心，再**将所有帧的中心合并为一点集**，在 **y 频数直方图** 上按 `--line-id`（默认 2，即第二峰）取 bin，得到行 `y0`；在 **`--profile-x-range`**（默认与 `--edge-x-range` 相同，脚本默认 `300–600`）上对 `I(y0-1)+I(y0)+I(y0+1)` 做 x 剖面，`find_peaks` 得各列 `x`，再在 `y0±halfwin` 与 `x±N` 列上求 **亮度质心** 得精细 `y`。输出默认目录见 `outputs/second_layer_peaks/`：全样本 y 直方图（标出所选峰）、每帧剖面图与离子叠图。详情见文件头说明与 `python second_layer_ion_peaks.py -h`。

```bash
python second_layer_ion_peaks.py 0 --line-id 2
python second_layer_ion_peaks.py ::10 --hist-prominence 10
```

### 3.8 批量 second-layer-slab（`batch_merge_second_layer_slab.py`）

对单目录或 `--batch-root` 下多子目录批量运行：对**每一帧**单独用该帧的合并中心建 y 直方图并完成 slab 替换（与 `merge_ion_centers.py 0:997 --second-layer-slab` **一次**调用时「多帧 y 合并成一张直方图」**不同**）。默认索引 `0:997`、`--data-dir` 为 `output_paths.DEFAULT_DATA_DIR`；输出根目录由 `output_paths.new_batch_merge_slab_run_dir()` 生成（`outputs/batch_merge_slab_<时间戳>/`），同目录写 `batch_run.log`，汇总结果在 `summary/`（含 `batch_summary.json` 与汇总图）。用法见脚本文件头与 `python batch_merge_second_layer_slab.py -h`。

```bash
python batch_merge_second_layer_slab.py
python batch_merge_second_layer_slab.py 800 --data-dir D:/data/one_run
```

### 3.9 批处理日志离子数直方图（`plot_batch_log_ion_histogram.py`）

解析日志中含 `离子数=整数` 的行（与 `batch_merge_second_layer_slab` 输出格式一致），绘制每帧离子数的直方图（坐标轴标签为英文）。

```bash
python plot_batch_log_ion_histogram.py path/to/batch_run.log
python plot_batch_log_ion_histogram.py path/to/batch_run.log --bin-width 2
```

（默认 PNG：`outputs/histogram/log_ion_hist_<日志 stem>.png`，见 `output_paths.default_log_ion_hist_png`。）

---

## `gallery.py`

交互式浏览 `npy` 图像并叠加检测结果；界面文案为英文。

```bash
python .\gallery.py
```

**界面初始默认**：`rel=0.025`；复选框 **`Two-pass ref.`**（`refine`）默认开。检测默认与 `detect_ions` 一致（含 θ=0 轴对齐）。

### 布局与显示

- **顶部**：`rel` 文本框与 **`Two-pass ref.`** 复选框。修改参数并回车后缓存会清空，需再次点击 **`Detect`**。
- **中部**：当前帧灰度图，**`aspect="equal"`**，像素在屏幕上近似为正方形。
- **底部**：帧索引滑条、`Go` 文本框、`Detect` / `Prev` / `Next`，以及状态栏。

检测结果按 **`(帧索引, 参数签名)`** 缓存，便于在同参数下切换帧时复用叠加。

### 操作

- 滑条切换帧
- `Go` 文本框输入索引后回车跳转
- `Prev` / `Next` 按钮翻页
- 键盘：`Left/Right`、`Up/Down`、`PageUp/PageDown`、`Home/End`、`n` / `p`
- **`Detect`**：用当前面板参数对当前帧调用 `detect_ions` 并绘制椭圆与晶格边界

---

## `stretching_analysis.py`

统计 `y_rel`（相对晶格中心 y）与：

- `ratio = sigma_minor / sigma_major`
- `amplitude`

之间关系，并输出拟合图与系数。

```bash
# 默认 100 帧，ratio/amp 都用 quadratic
python .\stretching_analysis.py

# 指定帧数
python .\stretching_analysis.py -n 10

# 独立指定拟合方法
python .\stretching_analysis.py -n 100 --ratio-fit quartic --amp-fit gaussian
```

**参数默认值**：`-n` / `--n-frames` 为 `100`；`--ratio-fit` 与 `--amp-fit` 均为 `quadratic`（可选 `quartic`、`gaussian`）。

可选拟合方法：

- `quadratic`
- `quartic`
- `gaussian`

注意：

- 当方法为 `quadratic` 时，仅使用 `|y_rel| <= 40` 的点参与拟合。

输出文件示例：

- 图：`outputs/stretch_analysis/stretching_analysis_10_ratio-quadratic_amp-quadratic.png`
- amplitude 系数：`outputs/amp_y_fit/amp_vs_y_coef_10_quadratic.npy`

---

## `dist.py`

读取 `outputs/IonPos/*.npy`（默认），计算任意两构型之间的标准化距离并绘制直方图。

距离定义（A 对 B）：

1. `D = cdist(A, B)`
2. 对每行取最小值
3. 求平均：`mean(min(D, axis=1))`

```bash
#   使用默认 outputs/IonPos 下全部文件
python .\dist.py

# 仅用前 100 个构型
python .\dist.py --count 100

```

**参数默认值**：`--pos-dir` 为 `outputs/IonPos`；`--count` 未传则用目录下**全部** `.npy`；`--bins` 为 `50`；直方图 PNG 固定为 `outputs/histogram/cdist_hist_{所用构型数}.png`（`output_paths.default_cdist_hist_png`）；`--show` 默认关闭。

---

## `vis_selected_npy.py`

从数据目录（默认 `20260305_1727`）读取指定的 `.npy` 文件名或 stem，导出灰度 PNG 或弹出窗口预览；默认输出目录为 `outputs/npy_plots/`。

```bash
python .\vis_selected_npy.py 20260305_005542 20260305_010000
python .\vis_selected_npy.py --dir 20260305_1727 20260305_005542
python .\vis_selected_npy.py --one-figure --cmap inferno 20260305_005542
```

**参数默认值**：`--dir` 为 `output_paths.DEFAULT_DATA_DIR`；多文件输出至 `outputs/npy_plots/`，`--one-figure` 时单图为 `outputs/npy_plots/selected_montage.png`（`default_vis_selected_montage_png`）；`--cmap` 为 `viridis`；`--dpi` 为 `150`；`--zoom-axes` 未传等价于 `1,1`（不上采样）；`--zoom-order` 为 `1`（线性）；`--interpolation` 为 `antialiased`；`--one-figure`、`--per-file-scale`、`--show` 默认关（多帧时默认**共用**色标以便对比）。

更多参数见脚本内 `argparse` 帮助：`python .\vis_selected_npy.py -h`。

---

## `project_info.py`

扫描 `20260305_1727`，打印文件数量、形状、时间范围等摘要，并在 `outputs/dataset_viz/` 下保存示例热力图、均值图、行/列剖面、强度直方图等。

```bash
python .\project_info.py
```

---

## 4. 推荐工作流

1. **先检测并导出坐标**  
   `python -m ion_detect "::5" --save-pos`（或 `python .\ion_detection.py ...`）
2. **查看单帧效果**  
   `python .\gallery.py`
3. **做 y 向统计拟合**  
   `python .\stretching_analysis.py -n 100 --ratio-fit quadratic --amp-fit quadratic`
4. **做构型距离统计**  
   `python .\dist.py --count 100`
5. **（可选）y 向边缘列剖面或竖直分层**  
   `python .\edge_strip_profile.py 0` 或 `python .\y_layer_profile.py 0`（**3.3** / **3.5** 节）
6. **（可选）合并外缘条带与 `ion_detect` 中心并出图**  
   `python .\merge_ion_centers.py 0`（见 **3.4** 节；含 `--second-layer-slab`）
7. **（可选）第二层 y 行或批量 slab**  
   `python second_layer_ion_peaks.py 0` 或 `python batch_merge_second_layer_slab.py`（见 **3.7**–**3.9** 节）

---

## 5. 调参建议

- 漏检 / 假峰：在 `gallery` 中调 `rel`，或在代码里改 `detect_ions(..., rel_threshold=...)`；必要时改 `fit_hw`、`peak_size`（仅代码）。
- 噪点过多：提高 `rel_threshold`。
- **需要旋转椭圆 PSF**：在代码中调用 `detect_ions(..., fix_theta_zero=False)`（CLI 不再提供开关）。
- 形变分析时：优先保证样本数量（`-n` 更大），并对比 `quadratic / quartic / gaussian` 的拟合稳定性与 `R^2`。
- **合并中心**（`merge_ion_centers.py`）：外缘条带域内漏检时可略减 `--peak-dist`；`--ion-dist` 过大可能误并相邻格点；`--edge-x-range` 需覆盖可靠条带列、又尽量不含左右圆角误判区。
- **第二层 / slab**：`merge` 与独立脚本的 `--second-layer-hist-prominence` 默认不同（5 vs 10），跨脚本对比时注意统一；多帧直方图行为以「单次 `merge_ion_centers`」与「`batch_merge_second_layer_slab` 每帧独立」为两套语义，勿混用。

---

## 6. 当前项目文件概览

- `output_paths.py`：统一约定 `outputs/` 下各子目录路径，供脚本默认读写
- `ion_detect/`：检测核心包（`pipeline`、`gaussian`、`boundary`、`preprocess`、`fitting`、`peel`、`binarize`、`edge_strip`、`edge_strip_profile_analysis`、`edge_strip_profile_viz`、`viz`、`cli_helpers`；`python -m ion_detect` 入口）
- `ion_detection.py`：兼容层，再导出 `detect_ions`、`visualize`、`visualize_bgsub`、`visualize_bgsub_binarized`、`bgsub_binarize` / `bgsub_binarize_u8`、`print_summary`，并支持 `python ion_detection.py ...` 调用 CLI
- `gallery.py`：交互式可视化浏览
- `stretching_analysis.py`：y 向统计拟合分析
- `dist.py`：构型距离统计
- `vis_selected_npy.py`：指定帧矩阵导出 PNG
- `edge_strip_profile.py`：y 向外缘条带按列聚合与主峰值/多峰示意；分析作图逻辑在 `ion_detect.edge_strip_profile_*`（见 **3.3** 节）
- `statistic.ipynb`：数据集统计相关 Notebook（按需使用）
- `merge_ion_centers.py`：默认 θ=0 的 `detect_ions` 与外缘条带 COM 规则合并、可选 detect–strip 距离融合与 `--second-layer-slab`（见 **3.4** 节）
- `second_layer_core.py`：第二层 y 直方图 / 三行剖面 / COM / slab 替换（供 `merge_ion_centers`、`second_layer_ion_peaks`、`batch_merge_second_layer_slab` 共用）
- `second_layer_ion_peaks.py`：独立第二层诊断与出图（见 **3.7** 节）
- `batch_merge_second_layer_slab.py`：批量每帧 slab（见 **3.8** 节）
- `plot_batch_log_ion_histogram.py`：从批处理日志画离子数直方图（见 **3.9** 节）
- `y_layer_profile.py`：固定 x 条带内按行积分，y 向 1D 轮廓（见 **3.5** 节）
- `project_info.py`：数据集统计与概览图
- `ion_detect/edge_strip_profile_analysis.py`、`ion_detect/edge_strip_profile_viz.py`：条带列剖面分析与可视化（由 `edge_strip_profile.py` 调用）
- `.gitignore`：忽略数据、缓存与本地环境文件

