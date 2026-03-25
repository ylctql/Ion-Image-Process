# IonImage

离子晶格图像处理与分析工具集，包含：

- 单帧离子检测与椭圆拟合：核心实现为 **`ion_detect` 包**；根目录 `ion_detection.py` 为兼容入口（`detect_ions` / 命令行）
- 交互式浏览与手动触发检测（`gallery.py`）
- 形变与亮度随 `y` 方向统计拟合（`stretching_analysis.py`）
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

# 第二轮只在晶格椭圆 y 向上下缘带内取候选（需能估计晶格 boundary）
python -m ion_detect 0 --peak-peel --peak-peel-y-edges-only ^
  --peak-peel-rel-threshold 0.04 ^
  --peak-peel-min-amp-frac 0.35
```

常用参数：

| 选项 | 含义 |
|------|------|
| `--peak-peel` | 启用第二轮剥离检测 |
| `--peak-peel-min-sep PX` | 新峰与已有峰中心最小距离（像素），默认 `2` |
| `--peak-peel-y-edges-only` | 第二轮仅保留 `|y-cy|/b` 较大的 y 向边缘候选 |
| `--peak-peel-y-edge-frac F` | 边缘带：保留 `|y-cy|/b ≥ 1-F`，默认 `F=0.25` |
| `--peak-peel-rel-threshold R` | 第二轮相对阈值（不传则用首轮） |
| `--peak-peel-min-amp-frac Q` | 第二轮振幅须 ≥ `Q×` 首轮振幅中位数 |

代码中等价参数为 `detect_ions(..., peak_peel=True, ...)`，见 `ion_detect.pipeline.detect_ions` 文档字符串。

### 3.7 包内模块分工（便于维护与二次开发）

| 模块 | 职责 |
|------|------|
| `ion_detect.pipeline` | `detect_ions` 端到端流程 |
| `ion_detect.gaussian` | 2D 高斯模型与剥离核叠加 |
| `ion_detect.boundary` | 晶格椭圆边界估计与候选过滤 |
| `ion_detect.preprocess` | 匹配滤波、y 向阈值缩放、局部极大候选 |
| `ion_detect.fitting` | 单峰/联合双峰拟合与精修 |
| `ion_detect.peel` | 合并去重、y 向边缘带过滤 |
| `ion_detect.viz` | `visualize`、`print_summary` |
| `ion_detect.cli_helpers` | 命令行索引解析 |
| `output_paths`（根目录模块） | `outputs/` 下 `ion_detect_imgs`、`IonPos`、`amp_y_fit` 等默认路径 |

---

## `gallery.py`

交互式浏览 `npy` 图像并叠加检测结果；界面文案为英文。

```bash
python .\gallery.py
```

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

---

## `vis_selected_npy.py`

从数据目录（默认 `20260305_1727`）读取指定的 `.npy` 文件名或 stem，导出灰度 PNG 或弹出窗口预览；默认输出目录为 `outputs/npy_plots/`。

```bash
python .\vis_selected_npy.py 20260305_005542 20260305_010000
python .\vis_selected_npy.py --dir 20260305_1727 --out outputs\npy_plots 20260305_005542
python .\vis_selected_npy.py --one-figure --cmap inferno 20260305_005542
```

更多参数（`--zoom-axes`、`--show`、`--dpi` 等）见脚本内 `argparse` 帮助：`python .\vis_selected_npy.py -h`。

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

---

## 5. 调参建议

- 漏检边缘离子：尝试开启 `--use-y-thresh-comp`，并降低 `--comp-floor`（如 `0.15`）。
- 噪点过多：提高 `--comp-floor` 或减小补偿强度；必要时提高 `rel_threshold`（CLI / `gallery` 中 `rel` 文本框，或代码中 `detect_ions(..., rel_threshold=...)`）。
- **y 向靠得很近、单峰拟合不稳**：可尝试 `--joint-pair-y-gap`（略小于典型竖直间距）；`DY` 过大易误配对。
- **重叠导致漏检（尤其上下缘）**：可试 `--peak-peel`，并配合 `--peak-peel-y-edges-only` 与略高的 `--peak-peel-rel-threshold` / `--peak-peel-min-amp-frac`，减少中间区域假小峰。
- **固定 θ 拟合**：离子拉长方向与坐标轴一致时可试 `--fix-theta-zero`；一般旋转 PSF 仍用默认旋转高斯。
- 形变分析时：优先保证样本数量（`-n` 更大），并对比 `quadratic / quartic / gaussian` 的拟合稳定性与 `R^2`。

---

## 6. 当前项目文件概览

- `output_paths.py`：统一约定 `outputs/` 下各子目录路径，供脚本默认读写
- `ion_detect/`：检测核心包（`pipeline`、`gaussian`、`boundary`、`preprocess`、`fitting`、`peel`、`viz`、`cli_helpers`；`python -m ion_detect` 入口）
- `ion_detection.py`：兼容层，再导出 `detect_ions` / `visualize` / `print_summary`，并支持 `python ion_detection.py ...` 调用 CLI
- `gallery.py`：交互式可视化浏览
- `stretching_analysis.py`：y 向统计拟合分析
- `dist.py`：构型距离统计
- `vis_selected_npy.py`：指定帧矩阵导出 PNG
- `project_info.py`：数据集统计与概览图
- `.gitignore`：忽略数据、缓存与本地环境文件

