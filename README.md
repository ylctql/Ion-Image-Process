# IonImage

离子晶格图像处理与分析工具集，包含：

- 单帧离子检测与椭圆拟合（`ion_detection.py`）
- 交互式浏览与手动触发检测（`gallery.py`）
- 形变与亮度随 `y` 方向统计拟合（`stretching_analysis.py`）
- 多构型距离统计与直方图（`dist.py`）

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
- 检测可视化输出：`visualization_output/`
- 离子中心坐标输出：`IonPos/`
- 构型距离直方图输出：`histogram/`

> `20260305_1727` 中每个 `npy` 被视为一帧图像（若为 3D 会按脚本逻辑处理）。

---

## 3. 核心脚本

## `ion_detection.py`

对指定帧做离子检测，输出椭圆叠加图；支持索引切片并集与可选 y 向阈值补偿。

### 3.1 基础用法

```bash
# 默认处理第 0 帧
python .\ion_detection.py

# 处理多个索引（支持整数、负数、切片并集）
python .\ion_detection.py 0 5 -1 "::3,0:10"
```

### 3.2 保存离子中心

```bash
python .\ion_detection.py "::5" --save-pos
python .\ion_detection.py 0:20 --save-pos --pos-dir .\IonPos
```

每帧保存为同名 `npy`，内容为 `N x 2` 的 `[x0, y0]`。

### 3.3 启用 y 向阈值补偿（可选）

```bash
python .\ion_detection.py 0 ^
  --use-y-thresh-comp ^
  --amp-coef-path .\visualization_output\amp_vs_y_coef_10.npy ^
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

---

## `gallery.py`

交互式浏览 `npy` 图像并叠加检测结果。

```bash
python .\gallery.py
```

功能：

- 滑条切换帧
- 文本框输入索引后回车跳转
- `Prev` / `Next` 按钮翻页
- 键盘翻页：`Left/Right`、`Up/Down`、`PageUp/PageDown`、`Home/End`
- `Detect` 按钮对当前页执行检测并叠加结果

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

- 图：`visualization_output/stretching_analysis_10_ratio-quadratic_amp-quadratic.png`
- amplitude 系数：`visualization_output/amp_vs_y_coef_10_quadratic.npy`

---

## `dist.py`

读取 `IonPos/*.npy`，计算任意两构型之间的标准化距离并绘制直方图。

距离定义（A 对 B）：

1. `D = cdist(A, B)`
2. 对每行取最小值
3. 求平均：`mean(min(D, axis=1))`

```bash
# 使用 IonPos 全部文件
python .\dist.py

# 仅用前 100 个构型
python .\dist.py --count 100

# 自定义输出
python .\dist.py --count 100 --bins 80 --output .\histogram\cdist_hist_100.png
```

---

## 4. 推荐工作流

1. **先检测并导出坐标**  
   `python .\ion_detection.py "::5" --save-pos`
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
- 噪点过多：提高 `--comp-floor` 或减小补偿强度；必要时提高 `rel_threshold`（需改脚本调用参数）。
- 形变分析时：优先保证样本数量（`-n` 更大），并对比 `quadratic / quartic / gaussian` 的拟合稳定性与 `R^2`。

---

## 6. 当前项目文件概览

- `ion_detection.py`：检测主算法与 CLI
- `gallery.py`：交互式可视化浏览
- `stretching_analysis.py`：y 向统计拟合分析
- `dist.py`：构型距离统计
- `.gitignore`：忽略数据、缓存与本地环境文件

