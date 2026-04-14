"""合并后轴对齐矩形：按 y 向条带分割并计算离子平衡坐标（矩形中心或条带内前景质心）。"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .blob_components import MinAreaRect, rect_component_labels


def _rect_aabb(r: MinAreaRect) -> tuple[float, float, float, float]:
    c = np.asarray(r.corners_xy, dtype=np.float64)
    xmin = float(c[:, 0].min())
    xmax = float(c[:, 0].max())
    ymin = float(c[:, 1].min())
    ymax = float(c[:, 1].max())
    return xmin, xmax, ymin, ymax


def _no_split_aabbs(
    rects: list[MinAreaRect],
    *,
    split: bool,
    max_ysize: float,
    eps: float = 1e-9,
) -> list[tuple[float, float, float, float]]:
    """``split=True`` 时，y 向不需要分割的矩形（``height <= max_ysize``）的 AABB 列表。"""
    if not split:
        return []
    thr = float(max_ysize)
    out: list[tuple[float, float, float, float]] = []
    for r in rects:
        _, _, ymin, ymax = _rect_aabb(r)
        height = ymax - ymin
        if height <= thr + eps:
            out.append(_rect_aabb(r))
    return out


def _y_bands_for_rect(
    ymin: float,
    ymax: float,
    *,
    split: bool,
    max_ysize: float,
    eps: float,
) -> list[tuple[float, float]]:
    """与旧版 y 分割一致：不 split 或高度不足时单带；否则等分为 ``ceil(height/max_ysize)`` 条。"""
    height = ymax - ymin
    thr = float(max_ysize)
    if not split or height <= thr + eps:
        return [(ymin, ymax)]
    n = int(np.ceil(height / thr))
    if n < 2:
        return [(ymin, ymax)]
    return [
        (ymin + i * height / n, ymin + (i + 1) * height / n)
        for i in range(n)
    ]


def _x_runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 的连通区间（含端点），``mask`` 为一维 bool。"""
    m = np.asarray(mask, dtype=bool).ravel()
    n = int(m.size)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < n and m[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _build_strip_mask_2d(
    bin_mask: np.ndarray,
    lab_im: np.ndarray | None,
    own_labels: Iterable[int],
    xmin: float,
    xmax: float,
    y0: float,
    y1: float,
    *,
    use_other_boxes: bool,
    other_boxes: list[tuple[float, float, float, float]],
    eps: float,
) -> tuple[np.ndarray, int, int]:
    """条带与矩形 AABB 交集上的工作二值掩膜（裁剪块 + 全局偏移 ``yi0, xi0``）。"""
    h, w = bin_mask.shape
    # 与条带 ``[y0,y1]`` 相交的整数行：首行 >= y0、末行 <= y1（避免 floor(y0-eps) 把上边界多收一行）
    yi0 = max(0, int(np.ceil(y0 - eps)))
    yi1 = min(h - 1, int(np.floor(y1 + eps)))
    xi0 = max(0, int(np.floor(xmin - eps)))
    xi1 = min(w - 1, int(np.ceil(xmax + eps)))
    if yi1 < yi0 or xi1 < xi0:
        return np.zeros((0, 0), dtype=bool), yi0, xi0
    sub = np.asarray(bin_mask[yi0 : yi1 + 1, xi0 : xi1 + 1], dtype=bool)
    if lab_im is not None:
        own_arr = np.asarray(list(own_labels), dtype=np.int32)
        sub_lab = lab_im[yi0 : yi1 + 1, xi0 : xi1 + 1]
        sub = sub & np.isin(sub_lab, own_arr)
    yy, xx = np.indices(sub.shape)
    glob_y = (yy + yi0).astype(np.float64)
    glob_x = (xx + xi0).astype(np.float64)
    in_rect = (
        (glob_x >= xmin - eps)
        & (glob_x <= xmax + eps)
        & (glob_y >= y0 - eps)
        & (glob_y <= y1 + eps)
    )
    sub = sub & in_rect
    if use_other_boxes:
        for qx0, qx1, qy0, qy1 in other_boxes:
            m_in_other = (
                (glob_x >= qx0 - eps)
                & (glob_x <= qx1 + eps)
                & (glob_y >= qy0 - eps)
                & (glob_y <= qy1 + eps)
            )
            sub = sub & (~m_in_other)
    return sub, yi0, xi0


def _weighted_centroid_from_mask(
    sub: np.ndarray,
    yi0: int,
    xi0: int,
    *,
    intensity: np.ndarray | None,
) -> tuple[float, float] | None:
    ys_loc, xs_loc = np.where(sub)
    if ys_loc.size == 0:
        return None
    gx = xs_loc.astype(np.float64) + xi0
    gy = ys_loc.astype(np.float64) + yi0
    if intensity is None:
        wts = np.ones(ys_loc.size, dtype=np.float64)
    else:
        z = intensity[gy.astype(np.int64), gx.astype(np.int64)].astype(np.float64)
        wts = np.maximum(z, 0.0)
        if not np.any(wts > 0):
            wts = np.ones(ys_loc.size, dtype=np.float64)
    sw = float(np.sum(wts))
    if sw <= 0:
        return None
    return float(np.sum(wts * gx) / sw), float(np.sum(wts * gy) / sw)


def _strip_positions_refine_x(
    r: MinAreaRect,
    bin_mask: np.ndarray,
    lab_im: np.ndarray | None,
    *,
    bands: list[tuple[float, float]],
    xmin: float,
    xmax: float,
    use_other_boxes: bool,
    other_boxes: list[tuple[float, float, float, float]],
    x_profile_threshold: float,
    x_profile_rel_to_max: bool,
    intensity: np.ndarray | None,
    eps: float,
) -> list[tuple[float, float]]:
    own = rect_component_labels(r)
    out: list[tuple[float, float]] = []
    x_thr = float(x_profile_threshold)
    rel_tm = bool(x_profile_rel_to_max)

    for y0, y1 in bands:
        sub, yi0, xi0 = _build_strip_mask_2d(
            bin_mask,
            lab_im,
            own,
            xmin,
            xmax,
            y0,
            y1,
            use_other_boxes=use_other_boxes,
            other_boxes=other_boxes,
            eps=eps,
        )
        if sub.size == 0 or not np.any(sub):
            out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            continue

        col_mean = sub.mean(axis=0)
        if rel_tm:
            cmx = float(np.max(col_mean))
            x_mask = col_mean > (x_thr * cmx) if cmx > 0 else np.zeros_like(col_mean, dtype=bool)
        else:
            x_mask = col_mean > x_thr
        runs = _x_runs_1d(x_mask)

        if not runs:
            c = _weighted_centroid_from_mask(
                sub, yi0, xi0, intensity=intensity,
            )
            if c is not None:
                out.append(c)
            else:
                out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            continue

        for a, b in runs:
            slab = np.zeros_like(sub, dtype=bool)
            slab[:, a : b + 1] = sub[:, a : b + 1]
            c = _weighted_centroid_from_mask(
                slab, yi0, xi0, intensity=intensity,
            )
            if c is not None:
                out.append(c)
            else:
                cx = float(xi0 + 0.5 * (a + b))
                cy = float(0.5 * (y0 + y1))
                out.append((cx, cy))

    return out


def ion_equilibrium_positions_xy(
    rects: list[MinAreaRect],
    binary: np.ndarray,
    *,
    labeled: np.ndarray | None = None,
    split: bool = False,
    max_ysize: float = 9.0,
    refine_x: bool = False,
    x_profile_threshold: float = 0.5,
    x_profile_rel_to_max: bool = False,
    intensity: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """
    对每个矩形给出一个或多个 ``(x, y)`` 识别位置（像素坐标，与 ``imshow`` 一致）。

    - ``split=False``：每个矩形一个点，为 ``MinAreaRect.center_xy``（轴对齐盒几何中心）。
    - ``split=True``：若矩形 y 向跨度 ``height > max_ysize``，令
      ``n = ceil(height / max_ysize)``，将 ``[ymin, ymax]`` 等分为 ``n`` 段；每段内对
      候选前景像素求形心。若传入 ``labeled``（与 ``label_connected_components`` 同形），则仅统计
      ``rect`` 的 ``component_labels`` 所含连通域标签的像素，使 AABB 相交处
      不会混入其它矩形的连通域。未传 ``labeled`` 时保持旧行为：矩形框内凡 ``binary`` 为真者
      均参与质心；且若条带与某个**不需 split** 的矩形 AABB 相交，会剔除落在这些盒子内的像素
      （用于缓解无标签时的重叠混淆）。

    不需分割的矩形仍为一个点 ``center_xy``。

    ``refine_x=True`` 时：对 **每一个** y 子带（含未做 y 细分时的整条带），在条带掩膜内将二值前景按列
    对 y 求平均得到 ``[0,1]`` 的占有率曲线，``x_profile_threshold``（默认 0.5）二值化后取 x 上连通段，
    每段内用 ``intensity``（若给定，否则等权）对前景像素求加权质心；多段则对应多个离子。若列阈值后无
    True 列但条带内仍有前景，则退化为整条带一次质心。建议 ``refine_x`` 与 ``split`` 同开，并在有去噪
    浮点图时传入 ``intensity`` 以加权。

    ``x_profile_rel_to_max=True`` 时，列条件改为 ``col_mean > x_profile_threshold * max(col_mean)``，
    适用于离子在 y 向只占条带少数行、绝对占有率难以超过 0.5 的情形；默认仍为绝对占有率阈值。
    """
    if not rects:
        return []
    bin_mask = np.asarray(binary, dtype=bool)
    lab_im: np.ndarray | None = None
    if labeled is not None:
        lab_im = np.asarray(labeled, dtype=np.int32)
        if lab_im.shape != bin_mask.shape:
            raise ValueError(
                f"labeled shape {lab_im.shape} != binary shape {bin_mask.shape}",
            )
    if intensity is not None:
        intensity = np.asarray(intensity, dtype=np.float64)
        if intensity.shape != bin_mask.shape:
            raise ValueError(
                f"intensity shape {intensity.shape} != binary shape {bin_mask.shape}",
            )
    thr = float(max_ysize)
    if thr <= 0:
        raise ValueError("max_ysize must be positive")
    out: list[tuple[float, float]] = []
    eps = 1e-9
    use_other_boxes = split and lab_im is None
    other_boxes = (
        _no_split_aabbs(rects, split=split, max_ysize=max_ysize, eps=eps)
        if use_other_boxes
        else []
    )

    for r in rects:
        xmin, xmax, ymin, ymax = _rect_aabb(r)
        height = ymax - ymin

        if refine_x:
            bands = _y_bands_for_rect(
                ymin, ymax, split=split, max_ysize=max_ysize, eps=eps,
            )
            out.extend(
                _strip_positions_refine_x(
                    r,
                    bin_mask,
                    lab_im,
                    bands=bands,
                    xmin=xmin,
                    xmax=xmax,
                    use_other_boxes=use_other_boxes,
                    other_boxes=other_boxes,
                    x_profile_threshold=x_profile_threshold,
                    x_profile_rel_to_max=x_profile_rel_to_max,
                    intensity=intensity,
                    eps=eps,
                ),
            )
            continue

        if not split or height <= thr + eps:
            out.append((float(r.center_xy[0]), float(r.center_xy[1])))
            continue

        n = int(np.ceil(height / thr))
        if n < 2:
            out.append((float(r.center_xy[0]), float(r.center_xy[1])))
            continue

        ys_all, xs_all = np.where(bin_mask)
        if ys_all.size == 0:
            for i in range(n):
                y0 = ymin + i * height / n
                y1 = ymin + (i + 1) * height / n
                out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            continue
        xf = xs_all.astype(np.float64)
        yf = ys_all.astype(np.float64)
        m_rect = (
            (xf >= xmin - eps)
            & (xf <= xmax + eps)
            & (yf >= ymin - eps)
            & (yf <= ymax + eps)
        )
        if lab_im is not None:
            own = rect_component_labels(r)
            own_arr = np.asarray(own, dtype=np.int32)
            labs_flat = lab_im[ys_all, xs_all]
            m_rect = m_rect & np.isin(labs_flat, own_arr)
        xs_r = xf[m_rect]
        ys_r = yf[m_rect]
        for i in range(n):
            y0 = ymin + i * height / n
            y1 = ymin + (i + 1) * height / n
            m_strip = (ys_r >= y0 - eps) & (ys_r <= y1 + eps)
            m_use = m_strip
            if use_other_boxes:
                for qx0, qx1, qy0, qy1 in other_boxes:
                    m_in_other = (
                        (xs_r >= qx0 - eps)
                        & (xs_r <= qx1 + eps)
                        & (ys_r >= qy0 - eps)
                        & (ys_r <= qy1 + eps)
                    )
                    m_use = m_use & (~m_in_other)

            if not np.any(m_use):
                out.append((0.5 * (xmin + xmax), 0.5 * (y0 + y1)))
            else:
                out.append((float(xs_r[m_use].mean()), float(ys_r[m_use].mean())))

    return out


def _denoised_weight_at_pixel(intensity: np.ndarray, x: float, y: float) -> float:
    """最近像素上 ``max(0, value)``，与 x 细化加权时正部一致。"""
    im = np.asarray(intensity, dtype=np.float64)
    h, w = im.shape
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return max(0.0, float(im[yi, xi]))


def merge_close_ion_positions_xy(
    positions: list[tuple[float, float]],
    ion_dist: float,
    *,
    intensity: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], int]:
    """
    在完成条带 / x 向细化等步骤得到的 ``(x, y)`` 列表上，按「距离从小到大」贪心合并过近点：

    反复在**当前**点集中选取欧氏距离**最小**的一对；若该距离严格小于 ``ion_dist``，则合并为加权质心
    （``intensity`` 非空时权重为最近像素 ``max(0, value)``，否则等权），并重复直至不存在这样的对。

    ``ion_dist <= 0`` 时不合并。合并次数作为第二项返回。
    """
    if ion_dist <= 0 or len(positions) < 2:
        return [(float(x), float(y)) for x, y in positions], 0
    thr2 = float(ion_dist) ** 2
    pts: list[np.ndarray] = [
        np.array([float(p[0]), float(p[1])], dtype=np.float64) for p in positions
    ]
    if intensity is not None:
        im = np.asarray(intensity, dtype=np.float64)
        wts: list[float] = [_denoised_weight_at_pixel(im, p[0], p[1]) for p in pts]
    else:
        wts = [1.0] * len(pts)
    if sum(wts) <= 1e-18:
        wts = [1.0] * len(pts)

    n_merges = 0
    while len(pts) >= 2:
        best_i, best_j = -1, -1
        best_d2 = float("inf")
        L = len(pts)
        for i in range(L):
            for j in range(i + 1, L):
                d2 = float(np.sum((pts[i] - pts[j]) ** 2))
                if d2 < best_d2:
                    best_d2 = d2
                    best_i, best_j = i, j
        if best_i < 0 or best_d2 >= thr2:
            break
        wi, wj = float(wts[best_i]), float(wts[best_j])
        wsum = wi + wj
        if wsum <= 1e-18:
            new_p = 0.5 * (pts[best_i] + pts[best_j])
            new_w = 1.0
        else:
            new_p = (wi * pts[best_i] + wj * pts[best_j]) / wsum
            new_w = wsum
        for idx in sorted((best_i, best_j), reverse=True):
            pts.pop(idx)
            wts.pop(idx)
        pts.append(new_p)
        wts.append(new_w)
        n_merges += 1

    return [(float(p[0]), float(p[1])) for p in pts], n_merges
