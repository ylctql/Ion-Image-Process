# IonImage

离子晶格图像处理与分析工具集，包含：

- 单帧离子检测与椭圆拟合：核心实现为 **`ion_detect` 包**；根目录 `ion_detection.py` 为兼容入口（`detect_ions` / 命令行）
- 交互式浏览与手动触发检测（`gallery.py`）
- 形变与亮度随 `y` 方向统计拟合（`stretching_analysis.py`）
- 固定 **x** 条带内按行积分，观察 **y** 向分层/条纹（`y_layer_profile.py`）
- 中心区域 `ion_detect` 与上下缘条带 COM **合并离子中心**（`merge_ion_centers.py`）
- 多构型距离统计与直方图（`dist.py`）
- 数据集整体统计与示例图（`project_info.py`）
- 指定若干帧导出矩阵 PNG（`vis_selected_npy.py`）

检测流程概览：**减背景 →（可选）匹配滤波 → 局部极大与阈值 → 2D 高斯拟合（可选近邻联合双峰）→（可选）峰值剥离第二轮**。剥离轮可限制在晶格椭圆 **y 向边缘带**，并单独提高阈值或振幅门槛，以减轻重叠区漏检与中间区域假小峰。

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
- **生成结果统一在 `outputs/` 下**（路径常量见根目录 `output_paths.py`，便于各脚本与 `.gitignore` 一致）：
  - `outputs/ion_detect_imgs/`：离子椭圆叠加 PNG（`python -m ion_detect` 默认）
  - `outputs/IonPos/`：离子中心 `N×2` 的 `.npy`
  - `outputs/amp_y_fit/`：y 向振幅拟合系数等（如 `amp_vs_y_coef_10.npy`）
  - `outputs/stretch_analysis/`：形变分析图
  - `outputs/histogram/`：构型距离直方图
  - `outputs/dataset_viz/`：`project_info.py` 生成的数据集统计与热力图等
  - `outputs/npy_plots/`：`vis_selected_npy.py` 默认输出
  - `outputs/edge_strip_profiles/`：`edge_strip_profile.py` 默认 PNG（常量 `OUT_EDGE_STRIP`）
  - `outputs/y_layer_profiles/`：`y_layer_profile.py` 默认 PNG（常量 `OUT_Y_LAYER_PROFILE`）
  - `outputs/ion_centers_merged/`：`merge_ion_centers.py` 默认 PNG（常量 `OUT_ION_CENTERS_MERGED`）

> `20260305_1727` 中每个 `npy` 被视为一帧图像（若为 3D 会按脚本逻辑处理）。

---

## 3. 核心脚本

## 离子检测（`ion_detect` / `ion_detection.py`）

对指定帧做离子检测，输出椭圆叠加图；支持索引切片并集、y 向阈值补偿、联合双峰拟合与峰值剥离等选项。

### 代码中调用

推荐直接使用包入口，便于与模块结构一致：

```python
from ion_detect import detect_ions, visualize, print_summary
```

历史脚本也可继续使用：

```python
from ion_detection import detect_ions  # 等价于从 ion_detect 再导出
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
| `--save-pos` | 关闭 | |
| `--pos-dir` | `outputs/IonPos` | 未传时使用 |
| `--use-y-thresh-comp` | 关闭 | |
| `--amp-coef-path` | `outputs/amp_y_fit/amp_vs_y_coef_10.npy` | 未传时使用 |
| `--amp-coef-mode` | `even` | 可选 `poly2` |
| `--comp-floor` | `0.2` | |
| `--fix-theta-zero` | 关闭 | |
| `--no-matched-filter` | 关闭 | 默认**启用**匹配滤波 |
| `--joint-pair-y-gap` | 未设置 | 不设则不启用联合双峰 |
| `--joint-pair-x-gap` | 未设置 | 仅在设置 `jY` 时生效；未设则检测内为 `max(4, hw_x)`，`hw_x` 来自 `fit_hw` |
| `--peak-peel [MODE]` | 关闭 | 省略 `MODE` 为全图第二轮；`MODE=edge` 为仅 y 向边缘带第二轮 |
| `--peak-peel-min-sep` | `2.0` | 像素 |
| `--peak-peel-y-edge-frac` | `0.25` | |
| `--peak-peel-rel-threshold` | 未传 | 与首轮 `rel_threshold` 相同（见下表） |
| `--peak-peel-min-amp-frac` | 未传 | 不设则二轮不按振幅比例过滤 |
| `--save-residual-img` | 关闭 | 须同时 `--peak-peel` |
| `--residual-img-dir` | `outputs/residual_imgs` | 未传时使用 |

**未单独暴露给 CLI、但 `detect_ions` 使用的默认**（见 `ion_detect/pipeline.py`）：`rel_threshold=0.025`，`bg_sigma=(10, 30)`，`peak_size=(5, 9)`，`fit_hw=(3, 4)`（半窗口），`sigma_range=(0.3, 3.5)`，`refine=True`，峰值剥离时 `peak_peel_margin_sigma=4.5`。需在代码中调用 `detect_ions(...)` 才能改这些量。

### 3.2 保存离子中心

```bash
python -m ion_detect "::5" --save-pos
python -m ion_detect 0:20 --save-pos --pos-dir .\outputs\IonPos
```

每帧保存为同名 `npy`，内容为 `N x 2` 的 `[x0, y0]`。

### 3.3 启用 y 向阈值补偿（可选）

```bash
python -m ion_detect 0 ^
  --use-y-thresh-comp ^
  --amp-coef-path .\outputs\amp_y_fit\amp_vs_y_coef_10.npy ^
  --amp-coef-mode even ^
  --comp-floor 0.2
```

参数说明：

- `--use-y-thresh-comp`：开启补偿
- `--amp-coef-path`：亮度-y 拟合系数文件
- `--amp-coef-mode`：
  - `even`：`[a0,a2,a4]`，模型 `a0 + a2*y^2 + a4*y^4`
  - `poly2`：`[p2,p1,p0]`，模型 `p2*y^2 + p1*y + p0`
- `--comp-floor`：行阈值缩放下限（默认 0.2）

### 3.4 固定椭圆转角 θ=0（轴对齐高斯拟合）

```bash
python -m ion_detect 0 --fix-theta-zero
```

高斯拟合时不优化旋转角，假定 PSF 主轴与图像 `x/y` 对齐。可视化标题中椭圆半轴以 `sigma` 标注。

### 3.5 y 向近邻双峰联合拟合（N=2，θ=0）

对**已通过局部极大值与阈值得到的候选峰**，若在竖直方向两两足够接近，则在合并 ROI 内同时拟合两个轴对齐高斯；失败则退回两次单峰拟合。**不改变**峰检测步骤，不能找回从未成为候选的离子。

```bash
python -m ion_detect 0 --joint-pair-y-gap 12
# 可选：限制水平配对范围（省略时默认 max(4, 拟合半宽 hw_x)）
python -m ion_detect 0 --joint-pair-y-gap 12 --joint-pair-x-gap 6
```

- `--joint-pair-y-gap DY`：两候选峰满足 `|Δy| ≤ DY` 且 `|Δx|` 在 x 间隙内时可结成一对做联合拟合。
- `--joint-pair-x-gap DX`：联合配对允许的 `|Δx|`（像素）。

### 3.6 峰值剥离（可选，缓解重叠漏检）

首轮拟合后，从原图减去各峰的高斯核，在残差上再做一轮**同一套**检测与拟合，并按中心距离合并去重。剥离不完美时，中间区域可能出现「傍大峰」小椭圆，可配合 **仅 y 向边缘第二轮** 或提高二轮阈值/振幅比例抑制。

```bash
# 开启剥离
python -m ion_detect 0 --peak-peel

# 同时保存首轮剥离后的残差图（默认 outputs/residual_imgs/peak_peel_residual_XXXX.png）
python -m ion_detect 0 --peak-peel --save-residual-img
# （若存在晶格 boundary，残差 PNG 上以淡洋红填充标出 |y-cy|/b>=1-F 的 y 向边缘带，F 即 --peak-peel-y-edge-frac，与第二轮 filter 准则一致；仅在使用 `--peak-peel edge` 时标题会标明 y 向边缘模式。）

# 第二轮只在晶格椭圆 y 向上下缘带内取候选（需能估计晶格 boundary；`MODE` 写在下标之后，或用 `--peak-peel=edge` 避免与帧编号粘连）
python -m ion_detect 0 --peak-peel edge ^
  --peak-peel-rel-threshold 0.04 ^
  --peak-peel-min-amp-frac 0.35
```

常用参数（默认值亦可查 **§3.1.1**）：

| 选项 | 含义 | 默认 |
|------|------|------|
| `--peak-peel [MODE]` | 启用第二轮剥离；`MODE` 省略=全图，`edge`=仅 y 向边缘候选 | 关 |
| `--peak-peel-min-sep PX` | 新峰与已有峰中心最小距离（像素） | `2` |
| `--peak-peel-y-edge-frac F` | 边缘带：保留 `|y-cy|/b ≥ 1-F` | `F=0.25` |
| `--peak-peel-rel-threshold R` | 第二轮相对阈值 | 与首轮 `0.025` 相同 |
| `--peak-peel-min-amp-frac Q` | 第二轮振幅须 ≥ `Q×` 首轮振幅中位数 | 不限制 |
| `--save-residual-img` | 保存残差图 `原图 − Σ首轮拟合峰`（须与 `--peak-peel` 同用） | 关 |
| `--residual-img-dir DIR` | 残差图目录 | `outputs/residual_imgs` |

代码中等价参数为 `detect_ions(..., peak_peel=True, return_peel_residual=True, ...)` 返回 `(ions, boundary, peel_residual)`，见 `ion_detect.pipeline.detect_ions` 文档字符串。

### 3.7 y 向外缘条带列轮廓（诊断上下缘漏检）

与上表 **`|y-cy|/b ≥ 1-F`** 同一套截线（`y_below = cy - (1-F)*b`，`y_above = cy + (1-F)*b`）。在 **上下两侧各** 取一轴对齐矩形：**y** 从椭圆在 **竖直方向的极点**（`cy ∓ b`，与 `boundary` 的 y 半轴一致）到对应截线；**x** 取该截线与椭圆边界的两交点之间的弦（`|x-cx| ≤ a·√(1 - ((y_cut-cy)/b)²)`）。对条带掩膜内像素按列做 **聚合**（`--col-metric`：`sum` / `mean` / `max`）得到 1D 曲线，脚本会保存曲线图并用抛物线插值细化 **峰值 x**（控制台亦打印）。

**实现结构**：列剖面核心算法在 `ion_detect.edge_strip_profile_analysis`，总览图与「逐列 y 剖面」交互窗口在 `ion_detect.edge_strip_profile_viz`；根目录 **`edge_strip_profile.py`** 仅为命令行入口（用法见其文件头注释）。

独立工具 **不加** `peak-peel` 前缀。`raw` / `bgsub` 时在 **`image -` 高斯背景** 上调用 `estimate_crystal_boundary`（与 `detect_ions` 估计 boundary 的方式一致）。**`peel` / `peel_bgsub`** 时在同一帧上调用 `detect_ions(..., peak_peel=True, return_peel_residual=True)`，从返回值中取 **boundary** 与残差（或可再减背景）图，**须首轮能检出离子** 才有有效残差。**`--preprocess`** 只决定送入条带 **按列聚合** 的二维图（与 `--col-metric` 的 sum/mean/max 正交）。

**图上两类竖线（与实现一致）**：

- **红虚线 + 红点**（子图 `Top/Bottom strip`）：整条 1D 剖面的 **全局最大**，经三点抛物线细化（`outer_y_edge_column_profiles`），**不受** `--peak-dist` 影响。
- **番茄红 / 绿色虚线**（叠在顶栏灰度图上）：上下条带剖面上的 **多峰示意**；由严格离散局部极大经 **`--peak-dist`** 最小间距筛选；过近的一对去低 **prominence** 峰（`scipy.signal.peak_prominences`；平手取 **y 较大** 者）。

**辅助峰列的 y 向诊断（`--peak-col-gallery` / `--plot-center`）**：在轮廓算法得到的每个辅助峰 **列** 上，沿 **y** 采样条带内像素（可用 `--y-fit-frac` 单独加宽约 `--y-edge-frac` 语义，条带 1D 轮廓与辅助峰 **x** 仍由 `--y-edge-frac` 决定）；可选 **x−1,x,x+1 三列求和**（`--add-neighbor-x`）。中心标点：`--plot-center` 可选 `fit`（默认，单/双高斯或 `--prominence`）、`com`（亮度加权质心）、`com_fit`（质心后取最近严格局部极大）；`--double-peak-fit` 强制双高斯曲线拟合；`--prominence [MIN]` 优先用 prominence 选两峰平均。**`--peak-col-gallery`** 会额外打开可滑动切换峰序号、单选项切换上/下条带的交互窗口，并隐含 **`--show`**（若在 IDE 内联后端无响应，请在系统终端设置 GUI 后端，例如 PowerShell：`$env:MPLBACKEND='TkAgg'; python edge_strip_profile.py 0 --peak-col-gallery`）。

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
python edge_strip_profile.py ::20 --out outputs/edge_strip_profiles
# 全部选项：python edge_strip_profile.py -h
```

| 选项 | 含义 | 默认 |
|------|------|------|
| 位置参数 `indices` | 与 `python -m ion_detect` 相同 numpy 风格切片并集 | `0` |
| `--y-edge-frac F` | 与 peak-peel 相同的 F，截线 `|y-cy|/b = 1-F` | `0.25` |
| `--preprocess` | `raw`；`bgsub`（`image -` 高斯背景）；`peel`（首轮 peak-peel 残差）；`peel_bgsub`（残差再减高斯背景，与检测二轮 map 输入一致）；后二者须首轮检出离子 | `raw` |
| `--col-metric` | 条带掩膜内按列 `sum` / `mean` / `max`；`mean` 整除该列掩膜像素数；`max` 在无掩膜列为 NaN；绘图时 `mean`/`max` 在无掩膜列不连线 | `mean` |
| `--plot-peel` | 仅 `peel` / `peel_bgsub`：顶栏 `imshow` 用 peel（或 peel+bgsub）图；**默认** 顶栏为原始载入的 `npy` | 关 |
| `--peak-dist D` | 见上文「叠图虚线」：相邻候选峰在 x 上须 **>** `D`（px）；`D≤0` 则标出全部严格局部极大 | `5` |
| `--peak-col-gallery` | 辅助峰列上沿 y 的 profile + 一维高斯拟合，滑动条切换峰；隐含 `--show` | 关 |
| `--plot-center [MODE]` | 顶图标注辅助峰列的 y 中心；省略 `MODE` 为 `fit`；可选 `com`、`com_fit` | 不标注 |
| `--double-peak-fit` | 列 y-profile 用双高斯（与 gallery / plot-center 联用） | 关 |
| `--prominence [MIN]` | prominence 筛选后取最多两峰平均作为中心；单独写时 `MIN=0` | 未启用 |
| `--y-fit-frac Ff` | 列 y 采样/拟合条带宽度（F 语义同 `--y-edge-frac`）；省略则与 `--y-edge-frac` 一致 | 与 F 相同 |
| `--add-neighbor-x` | 列 profile 每点为 x−1,x,x+1 三列强度和（gallery / plot-center） | 关 |
| `--no-clip-ellipse` | 条带不按椭圆再裁剪（整块轴对齐条带矩形参与掩膜/聚合） | 否 |
| `--show` | 处理完后 **交互弹窗**（仍会写入 `--out` 下 PNG） | 关 |
| `--data-dir` | `.npy` 目录 | `PROJECT_ROOT/20260305_1727`（与包内 CLI 一致） |
| `--out` | PNG 目录 | `outputs/edge_strip_profiles`（`output_paths.OUT_EDGE_STRIP`） |

条带几何与按列聚合：`ion_detect.edge_strip.outer_y_edge_column_profiles`（参数 `col_metric`，默认 `mean`；`outer_y_edge_strip_masks` 亦导出在 `ion_detect` 包根）。

### 3.8 合并离子中心（`merge_ion_centers.py`）

在 **`detect_ions(..., fix_theta_zero=True)`** 与 **§3.7** 同一套外缘条带几何之间做规则合并，输出带图例的单张 PNG（灰度图、晶格边界椭圆、上下缘分界线、`edge-x-range` 竖线、各中心点）。

**区域规则（与实现一致）**

- 在 **椭圆内**、**上下外缘 y 条带**（与 `outer_y_edge_strip_masks` 一致，由 `--y-edge-frac` 控制）且 **x** 落在 **`--edge-x-range X0 X1`**（默认 `250 750`）内：仅保留该 x 段上的 **条带辅助峰 + 列向 COM**（由 `fitted_xy_for_auxiliary_strip_peaks`）；**丢弃**落在此域内的 `ion_detect` 中心。
- **其余**（含外缘行但在 x 段之外）：仅保留 `ion_detect`；条带在该 x 段外的峰不参加合并。
- **`--ion-dist`**（默认 `5`，像素）：对 **detect** 与 **strip_top / strip_bot** 成对、欧氏距离 **≤** 该阈值的，反复合并 **最近** 的一对，**中心取坐标算术平均**，`sigma` / `amplitude` 等仍取自 detect；合并后 `source=fused_mean`（图上为洋红色）。**≤0** 关闭此步。**注意**：阈值若大于约半格距，可能误并相邻两个离子，请按数据微调。

条带侧默认与常用诊断一致：`--y-fit-frac`（默认 `0.35`）、`--strip-center-mode com`、可选用 `--add-neighbor-x`；`--preprocess` 仅作用于送入条带 **按列聚合** 的二维图，`detect_ions` 始终用原图流水线。

```bash
python merge_ion_centers.py 0
python merge_ion_centers.py 0 --add-neighbor-x
python merge_ion_centers.py 0 --edge-x-range 250 750 --y-edge-frac 0.25 ^
  --y-fit-frac 0.35 --peak-dist 5 --ion-dist 5
python merge_ion_centers.py ::10 --out outputs/ion_centers_merged
# 全部选项：python merge_ion_centers.py -h
```

| 选项 | 含义 | 默认 |
|------|------|------|
| 位置参数 `indices` | 与 `python -m ion_detect` 相同 numpy 风格切片并集 | `0` |
| `--edge-x-range X0 X1` | 上下缘「条带优先」的 x 闭区间；外侧外缘行仍只用 detect | `250 750` |
| `--y-edge-frac F` | 外缘条带几何 F（同 §3.7 `--y-edge-frac`） | `0.25` |
| `--y-fit-frac Ff` | 列 y 向 COM 采样条带宽度（同 §3.7） | `0.35` |
| `--peak-dist D` | 条带 1D 剖面上辅助峰最小间距（像素） | `5` |
| `--col-metric` | `sum` / `mean` / `max` | `mean` |
| `--strip-center-mode` | `com` / `com_fit` / `fit` | `com` |
| `--add-neighbor-x` | 列 profile 用 x−1,x,x+1 三列和 | 关 |
| `--ion-dist PX` | detect 与 strip 中心距离 ≤ PX 时合并为均值；≤0 关闭 | `5` |
| `--preprocess` | 送入条带的图：`raw` / `bgsub` / `peel` / `peel_bgsub`（后两者须首轮能检出离子） | `raw` |
| `--no-clip-ellipse` | 条带不按椭圆裁剪 | 否 |
| `--no-matched-filter` | `detect_ions` 禁用匹配滤波 | 否 |
| `--show` | 弹窗显示（仍会写 PNG） | 关 |
| `--data-dir` / `--out` | `.npy` 目录；PNG 目录 | `20260305_1727`；`outputs/ion_centers_merged` |

核心函数：`merge_centers_hybrid`、`fuse_detect_strip_by_distance`（见脚本内文档字符串）。

### 3.9 y 向分层轮廓（`y_layer_profile.py`）

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
| `--data-dir` / `--out` | 数据目录；PNG 目录 | `20260305_1727`；`outputs/y_layer_profiles` |

### 3.10 包内模块分工（便于维护与二次开发）

| 模块 | 职责 |
|------|------|
| `ion_detect.pipeline` | `detect_ions` 端到端流程 |
| `ion_detect.gaussian` | 2D 高斯模型与剥离核叠加 |
| `ion_detect.boundary` | 晶格椭圆边界估计与候选过滤 |
| `ion_detect.preprocess` | 匹配滤波、y 向阈值缩放、局部极大候选 |
| `ion_detect.fitting` | 单峰/联合双峰拟合与精修 |
| `ion_detect.peel` | 合并去重、y 向边缘带过滤 |
| `ion_detect.edge_strip` | y 向外缘条带按列聚合（sum/mean/max）与主峰值（与 peel 同一 F 几何） |
| `ion_detect.edge_strip_profile_analysis` | 条带列剖面辅助分析（y 向列 profile、高斯/prominence 等） |
| `ion_detect.edge_strip_profile_viz` | 条带总览图 `plot_edge_strip_dashboard`、逐列交互 `show_peak_column_gallery` |
| `ion_detect.viz` | `visualize`、`print_summary` |
| `ion_detect.cli_helpers` | 命令行索引解析 |
| `output_paths`（根目录模块） | `outputs/` 下 `ion_detect_imgs`、`IonPos`、`amp_y_fit`、`edge_strip_profiles`、`y_layer_profiles`、`ion_centers_merged` 等默认路径 |

---

## `gallery.py`

交互式浏览 `npy` 图像并叠加检测结果；界面文案为英文。

```bash
python .\gallery.py
```

**界面初始默认（与 CLI 对齐，见 §3.1.1）**：`rel=0.025`，`c_fl=0.2`，`mode=even`；复选框 `theta=0` / `Y thresh comp` 默认关，`Matched filt.` / `Two-pass ref.` 默认开；`jY` / `jX` 空表示不启用联合双峰。Y 补偿开启时用 `outputs/amp_y_fit/amp_vs_y_coef_10.npy`。

### 布局与显示

- **顶部**：与 `ion_detect` / `detect_ions` 对齐的检测选项——复选框（`theta=0`、`Y thresh comp`、`Matched filt.`、`Two-pass ref.`）及文本框（`rel`、`c_fl`、`jY`、`jX`、`mode` 等）。修改参数并回车后缓存会清空，需再次点击 **`Detect`**。
- **中部**：当前帧灰度图，**`aspect="equal"`**，像素在屏幕上近似为正方形。
- **底部**：帧索引滑条、`Go` 文本框、`Detect` / `Prev` / `Next`，以及状态栏。

启用 **Y thresh comp** 时，默认使用 `{项目根}/outputs/amp_y_fit/amp_vs_y_coef_10.npy`（与 CLI 默认一致）；若文件不存在会报错。

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

# 自定义输出
python .\dist.py --count 100 --bins 80 --output .\outputs\histogram\cdist_hist_100.png
```

**参数默认值**：`--pos-dir` 为 `outputs/IonPos`；`--count` 未传则用目录下**全部** `.npy`；`--bins` 为 `50`；`--output` 未传则为 `outputs/histogram/cdist_hist_{所用构型数}.png`；`--show` 默认关闭。

---

## `vis_selected_npy.py`

从数据目录（默认 `20260305_1727`）读取指定的 `.npy` 文件名或 stem，导出灰度 PNG 或弹出窗口预览；默认输出目录为 `outputs/npy_plots/`。

```bash
python .\vis_selected_npy.py 20260305_005542 20260305_010000
python .\vis_selected_npy.py --dir 20260305_1727 --out outputs\npy_plots 20260305_005542
python .\vis_selected_npy.py --one-figure --cmap inferno 20260305_005542
```

**参数默认值**：`--dir` 为项目根下 `20260305_1727`；`--out` 为 `outputs/npy_plots`；`--cmap` 为 `viridis`；`--dpi` 为 `150`；`--zoom-axes` 未传等价于 `1,1`（不上采样）；`--zoom-order` 为 `1`（线性）；`--interpolation` 为 `antialiased`；`--one-figure`、`--per-file-scale`、`--show` 默认关（多帧时默认**共用**色标以便对比）。

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
4. **（可选）启用 y 向阈值补偿再检测**  
   使用 `--use-y-thresh-comp` + `--amp-coef-path`
5. **做构型距离统计**  
   `python .\dist.py --count 100`
6. **（可选）y 向边缘列剖面或竖直分层**  
   `python .\edge_strip_profile.py 0` 或 `python .\y_layer_profile.py 0`
7. **（可选）合并外缘条带与 `ion_detect` 中心并出图**  
   `python .\merge_ion_centers.py 0 --add-neighbor-x`（见 **§3.8**）

---

## 5. 调参建议

- 漏检边缘离子：尝试开启 `--use-y-thresh-comp`，并降低 `--comp-floor`（如 `0.15`）。
- 噪点过多：提高 `--comp-floor` 或减小补偿强度；必要时提高 `rel_threshold`（CLI / `gallery` 中 `rel` 文本框，或代码中 `detect_ions(..., rel_threshold=...)`）。
- **y 向靠得很近、单峰拟合不稳**：可尝试 `--joint-pair-y-gap`（略小于典型竖直间距）；`DY` 过大易误配对。
- **重叠导致漏检（尤其上下缘）**：可试 `--peak-peel`，并配合 `--peak-peel edge` 与略高的 `--peak-peel-rel-threshold` / `--peak-peel-min-amp-frac`，减少中间区域假小峰。
- **固定 θ 拟合**：离子拉长方向与坐标轴一致时可试 `--fix-theta-zero`；一般旋转 PSF 仍用默认旋转高斯。
- 形变分析时：优先保证样本数量（`-n` 更大），并对比 `quadratic / quartic / gaussian` 的拟合稳定性与 `R^2`。
- **合并中心**（`merge_ion_centers.py`）：外缘条带域内漏检时可略减 `--peak-dist`；`--ion-dist` 过大可能误并相邻格点；`--edge-x-range` 需覆盖可靠条带列、又尽量不含左右圆角误判区。

---

## 6. 当前项目文件概览

- `output_paths.py`：统一约定 `outputs/` 下各子目录路径，供脚本默认读写
- `ion_detect/`：检测核心包（`pipeline`、`gaussian`、`boundary`、`preprocess`、`fitting`、`peel`、`edge_strip`、`edge_strip_profile_analysis`、`edge_strip_profile_viz`、`viz`、`cli_helpers`；`python -m ion_detect` 入口）
- `ion_detection.py`：兼容层，再导出 `detect_ions` / `visualize` / `print_summary`，并支持 `python ion_detection.py ...` 调用 CLI
- `gallery.py`：交互式可视化浏览
- `stretching_analysis.py`：y 向统计拟合分析
- `dist.py`：构型距离统计
- `vis_selected_npy.py`：指定帧矩阵导出 PNG
- `edge_strip_profile.py`：y 向外缘条带按列聚合与主峰值/多峰示意；分析作图逻辑在 `ion_detect.edge_strip_profile_*`（见 **§3.7**）
- `merge_ion_centers.py`：`fix-theta-zero` 检测与外缘条带 COM 规则合并、可选 detect–strip 距离融合（见 **§3.8**）
- `y_layer_profile.py`：固定 x 条带内按行积分，y 向 1D 轮廓（见 **§3.9**）
- `project_info.py`：数据集统计与概览图
- `ion_detect/edge_strip_profile_analysis.py`、`ion_detect/edge_strip_profile_viz.py`：条带列剖面分析与可视化（由 `edge_strip_profile.py` 调用）
- `.gitignore`：忽略数据、缓存与本地环境文件

