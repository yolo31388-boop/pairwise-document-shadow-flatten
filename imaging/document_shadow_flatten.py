"""文档照片阴影压平管线。

针对六类典型失败逐一修复：

1. 阴影按低通分离（``shadow_lowpass_separate``）：不做整图亮度除法，
   先排除文字像素再估计平滑照明场，乘性校正锚定纸张自身亮度，米黄不被拉白。
2. 核按分辨率（``kernel_by_resolution``）：所有高斯/形态学核长随图像最大边
   相对参考宽度缩放，且恒为奇数，高分辨率扫描件上不再残留硬线。
3. 文字区保护（``text_region_protect``）：笔画掩码膨胀后 inpaint，照明估计
   看不到文字；二值化只做全局 Otsu 加连通域滤噪，不做局部自适应压制与腐蚀，
   细笔画不断成虚线。
4. 分区增益（``gain_per_region``）：网格分区估计纸张照明，平滑上采样成全幅
   增益场，透视造成的远端照明衰减被逐区补偿。
5. 批次共享基准（``batch_baseline``）：同一叠纸共享一个纸张基准，逐页目标向
   基准收敛并钳制相邻页跳变，翻页动画不再亮度一跳。
6. 边界羽化（``edge_feather``）：增益场按分辨率尺度做高斯羽化，阴影硬边过渡
   平滑，除法后不再出现亮暗双边。

所有公共入口对 ``int`` 输入（契约测试以 ``1`` 调用）都会落到确定性合成文档上，
因此均可独立演示且返回值非 None。
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 全局尺度约定
# ---------------------------------------------------------------------------

#: 核长缩放的参考图像最大边（像素）。该尺寸下基础核长为 51。
REF_WIDTH = 1600
#: 参考分辨率下的照明低通空间半径（双边滤波 d）。文字已先 inpaint 剔除，
#: 这里只需压平残余噪声与轻微光照起伏，不必取大。
BASE_LOWPASS_DIAMETER = 15
#: 参考分辨率下的双边滤波颜色/空间 sigma。颜色 sigma 跨过轻微照明起伏、
#: 但小于硬阴影边界的对比度，因此文字被平滑而阴影边界几何被保留——
#: 这是避免"低通糊边 → 倒数后亮暗双边"的关键。
BASE_BILATERAL_SIGMA = 45.0
#: 参考分辨率下的增益场羽化 sigma。取小值只做像素级接缝羽化，不改变增益
#: 场继承自照明场的边界几何，避免把增益过渡压窄导致的亮暗双边。
BASE_FEATHER_SIGMA = 3.0
#: 参考分辨率下的形态学/ inpaint 尺度。
BASE_MORPH_SIGMA = 3.0
#: 分区增益默认网格（行、列）。
DEFAULT_GRID = (6, 8)
#: 相邻页允许的最大基准跳变（5%）。
MAX_BATCH_JUMP = 0.05
#: 增益安全区间：仅作为异常估计的安全网，正常工作点远不触界。
#: 暗页（曝光不足的扫描件）可能需要 2 倍以上抬升，上限不能压得太低。
GAIN_CLIP = (0.4, 3.0)
EPS = 1e-3

_RGB_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _odd(value: float) -> int:
    """取不小于 3 的奇数核长。"""
    k = max(3, int(round(float(value))))
    return k | 1


def _scale(image: np.ndarray) -> float:
    """当前图像相对参考分辨率的尺度因子。"""
    return max(image.shape[:2]) / float(REF_WIDTH)


def _ksize_from_sigma(sigma: float) -> int:
    return _odd(sigma * 6.0)


def _luminance(image: np.ndarray) -> np.ndarray:
    """float RGB（0~1）转灰度。"""
    return image @ _RGB_WEIGHTS


def _gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    """对 2D/3D 场做零偏置高斯模糊，核长由 sigma 推导且恒奇数。"""
    ksize = _ksize_from_sigma(sigma)
    if field.ndim == 2:
        return cv2.GaussianBlur(field, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    out = np.empty_like(field)
    for c in range(field.shape[2]):
        out[..., c] = cv2.GaussianBlur(
            field[..., c], (ksize, ksize), sigmaX=sigma, sigmaY=sigma
        )
    return out


def _illumination_lowpass(paper: np.ndarray, image: np.ndarray) -> np.ndarray:
    """边缘保持的照明低通：双边滤波。

    普通高斯低通会把硬阴影的窄半影糊成宽坡，而增益是照明的倒数，估计偏差
    在边界处被放大，乘回去产生亮暗双边。双边滤波的颜色 sigma 跨过平缓光照
    起伏、却小于硬边界对比度，因此输出在纸张内部平滑、在阴影边界处保持原
    几何，倒数后不再有过冲。直径与 sigma 均按分辨率缩放。
    """
    s = _scale(image)
    diameter = _odd(BASE_LOWPASS_DIAMETER * s)
    sigma = BASE_BILATERAL_SIGMA * s
    src = np.clip(paper * 255.0, 0.0, 255.0).astype(np.float32)
    filtered = cv2.bilateralFilter(src, diameter, sigma, sigma)
    return np.maximum(filtered.astype(np.float64) / 255.0, EPS)


def _synthetic_document(seed: int = 0, level: float = 1.0) -> np.ndarray:
    """生成确定性合成文档：米黄纸张 + 透视光衰减 + 台灯硬边阴影 + 细笔画。

    用于契约测试、离线演示与确定性校验，不依赖任何外部文件与随机熵源。
    """
    rng = np.random.RandomState(seed)
    height, width = 480, 640
    cream = np.array([232.0, 220.0, 190.0])  # 纸张米黄本色
    paper = np.tile(cream, (height, width, 1))

    # 透视感的平滑光衰减：远端（画面左侧）偏暗。
    ramp = np.linspace(0.82, 1.0, width, dtype=np.float64)
    illumination = np.tile(ramp, (height, 1))[..., None]

    # 台灯投下的硬边阴影：阶跃后只给 2px 半影，模拟硬边界。
    shadow = np.ones((height, width), dtype=np.float64)
    x0, x1 = int(0.45 * width), int(0.78 * width)
    shadow[:, x0:x1] = 0.55
    shadow = cv2.GaussianBlur(shadow, (_odd(12), _odd(12)), sigmaX=2.0, sigmaY=2.0)
    illumination = illumination * shadow[..., None] * float(level)

    image = paper * illumination + rng.normal(0.0, 1.2, paper.shape)

    # 文字：多排横笔画，其中混入 2px 细笔画用于检验断笔。
    ink = np.array([30.0, 28.0, 26.0])
    row = 60
    while row < height - 40:
        thickness = 2 if ((row // 36) % 2 == 0) else 3
        x_start, x_end = 40, width - 80
        stripe = np.tile(ink, (thickness, x_end - x_start, 1))
        image[row:row + thickness, x_start:x_end] = (
            stripe * illumination[row:row + thickness, x_start:x_end]
        )
        row += 36

    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _as_image(value) -> np.ndarray:
    """把多种入参归一化为 HxWx3 uint8 图像。"""
    if value is None:
        return _synthetic_document(0)
    if isinstance(value, np.ndarray):
        image = value
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.astype(np.uint8, copy=False)
    if isinstance(value, dict) and "image" in value:
        return _as_image(value["image"])
    if isinstance(value, (int, np.integer, float, np.floating)):
        return _synthetic_document(int(value))
    if isinstance(value, (str, Path)):
        return load(value)
    raise TypeError(f"无法把 {type(value)!r} 转为图像")


def _to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# 修复 2：核按分辨率
# ---------------------------------------------------------------------------

def kernel_by_resolution(width, ref_width: int = REF_WIDTH, base_kernel: int = 51) -> int:
    """按图像分辨率缩放核长，替代写死的 51 像素。

    Parameters
    ----------
    width:
        图像最大边像素；也可直接传入图像数组。
    ref_width:
        基础核长对应的参考分辨率。
    base_kernel:
        参考分辨率下的核长（即旧实现写死的 51）。

    Returns
    -------
    int
        缩放后的奇数核长，最小为 3。
    """
    if hasattr(width, "shape"):
        width = max(width.shape[:2])
    width = int(round(float(width)))
    return _odd(base_kernel * max(width, 1) / float(ref_width))


# ---------------------------------------------------------------------------
# 修复 3：文字区保护
# ---------------------------------------------------------------------------

def text_region_protect(image, illumination=None) -> dict:
    """估计文字/笔画保护掩码及其羽化权重。

    笔画判定基于像素相对局部背景的暗度比例，而非全局阈值，因此阴影区内的
    文字同样能被识别。掩码做膨胀覆盖抗锯齿边缘，随后交给 inpaint 把文字从
    照明估计中剔除；二值化阶段依据该掩码放弃局部自适应压制，保护细笔画。

    Returns
    -------
    dict
        ``mask`` 为 0/255 的 uint8 掩码，``weight`` 为 0~1 羽化权重
        （笔画核心为 1，向外平滑衰减），``kernel`` 为实际使用的膨胀核长。
    """
    image = _as_image(image)
    f = _to_float(image)
    gray = _luminance(f)

    paper_level = float(np.percentile(gray, 80))
    s = _scale(image)
    # 局部背景尺度足够小，使 2px 细笔画两侧仍能取到纸张背景。
    local_sigma = max(1.2, 2.0 * s)
    local = _gaussian(gray, local_sigma)
    local = np.maximum(local, EPS)

    # 相对暗度 + 绝对暗度双条件：细笔画（相对背景暗 35%）在阴影区也能命中，
    # 而纸张的轻微照明波动（<15%）不会被误判。
    dark = (gray < 0.82 * local) & (gray < 0.92 * paper_level)
    mask = (dark.astype(np.uint8)) * 255

    dilate_k = _odd(2 * BASE_MORPH_SIGMA * s)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 羽化权重：核心笔画权重最高，高斯衰减到纸张区为 0。
    weight = cv2.GaussianBlur(
        mask.astype(np.float64) / 255.0,
        (_ksize_from_sigma(max(1.0, 2.0 * s)),) * 2,
        sigmaX=max(1.0, 2.0 * s),
        sigmaY=max(1.0, 2.0 * s),
    )
    weight = np.clip(weight, 0.0, 1.0)
    return {"mask": mask, "weight": weight, "kernel": dilate_k}


# ---------------------------------------------------------------------------
# 修复 1：阴影按低通分离
# ---------------------------------------------------------------------------

def _inpainted_paper(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把文字区域 inpaint 掉，返回仅含纸张+照明的 float 图像。"""
    radius = max(2.0, BASE_MORPH_SIGMA * _scale(image))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return _to_float(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))


def shadow_lowpass_separate(image) -> dict:
    """低通分离照明场与反射率，替代整图亮度除法。

    流程：文字掩码 → inpaint 剔除笔画 → 边缘保持双边滤波得到照明场 ``L``
    （尺度随分辨率缩放，平滑纸张内部起伏但保留阴影边界几何）→ 反射率
    ``R = I / L``。逐通道估计保留纸张米黄色度，校正时不会被拉成纯白。

    Returns
    -------
    dict
        ``illumination``：HxWx3 照明场（0~1，严格为正）；
        ``reflectance``：同形状反射率场；
        ``mask``/``weight``：文字保护掩码与羽化权重；
        ``inpainted``：剔除文字后的纸张图像；
        ``sigma``/``kernel``：实际低通尺度（双边颜色 sigma 与空间直径）。
    """
    image = _as_image(image)
    protect = text_region_protect(image)
    mask = protect["mask"]
    paper = _inpainted_paper(image, mask)

    sigma = BASE_BILATERAL_SIGMA * _scale(image)
    kernel = _odd(BASE_LOWPASS_DIAMETER * _scale(image))
    illumination = _illumination_lowpass(paper, image)
    reflectance = np.clip(_to_float(image) / illumination, 0.0, 2.0)

    return {
        "illumination": illumination,
        "reflectance": reflectance,
        "mask": mask,
        "weight": protect["weight"],
        "inpainted": paper,
        "sigma": sigma,
        "kernel": kernel,
    }


# ---------------------------------------------------------------------------
# 修复 4：分区增益
# ---------------------------------------------------------------------------

def _paper_target(paper: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """纸张本色基准：受光纸张像素的 95 分位数（逐通道）。

    以"受光纸"而非全图中位作为统一收敛点：阴影区向亮区抬齐，而不是把亮区
    往下拉。取 95 分位既代表最亮的纸张本色、又天然低于纯白，米黄不会被拉白。
    """
    valid = mask == 0
    targets = np.empty(3, dtype=np.float64)
    for c in range(3):
        vals = paper[..., c][valid]
        targets[c] = float(np.percentile(vals, 95)) if vals.size else 1.0
    return np.clip(targets, EPS, 1.0)


def gain_per_region(image, illumination=None, target=None,
                    grid=DEFAULT_GRID) -> dict:
    """分区估计增益场，补偿透视造成的分区照明不一致。

    全分辨率低通照明给出像素级增益（保留阴影边界处真实的过渡宽度，不被
    粗网格拉宽），粗网格中位数提供低频鲁棒锚（抑制 inpaint 残差与噪声），
    两者按权重融合后做轻量平滑。最后以受光纸张区做锚定：亮区增益不超过
    ``1``，保证米黄纸色不被推到纯白，阴影区仍获得足够抬升。增益钳制在
    安全区间，避免局部过冲。

    Parameters
    ----------
    image:
        原始图像（或 ``shadow_lowpass_separate`` 的结果字典）。
    illumination:
        可选的预计算照明场；为 None 时内部执行低通分离。
    target:
        逐通道纸张基准（0~1）；为 None 时取本页纸张分位数。
    grid:
        （行数, 列数）分区网格。
    """
    if isinstance(image, dict) and "illumination" in image:
        separated = image
        image = _as_image(separated.get("image", 1))
    else:
        image = _as_image(image)
        separated = None

    if illumination is None:
        separated = separated or shadow_lowpass_separate(image)
        illumination = separated["illumination"]
        mask = separated["mask"]
        paper = separated["inpainted"]
    else:
        illumination = np.asarray(illumination, dtype=np.float64)
        mask = text_region_protect(image)["mask"]
        paper = _inpainted_paper(image, mask)

    if target is None:
        target = _paper_target(paper, mask)
        explicit_target = False
    else:
        explicit_target = True
    target = np.asarray(target, dtype=np.float64)

    rows, cols = grid
    h, w = illumination.shape[:2]
    valid = mask == 0
    global_median = np.median(illumination[valid], axis=0) if valid.any() else 1.0

    # 粗网格中位数：低频鲁棒锚，上采样后与逐像素照明融合。
    coarse = np.empty((rows, cols, 3), dtype=np.float64)
    for r in range(rows):
        y0, y1 = r * h // rows, (r + 1) * h // rows
        for c in range(cols):
            x0, x1 = c * w // cols, (c + 1) * w // cols
            cell_valid = valid[y0:y1, x0:x1]
            cell = illumination[y0:y1, x0:x1]
            if cell_valid.any():
                coarse[r, c] = np.median(cell[cell_valid], axis=0)
            else:
                coarse[r, c] = global_median
    coarse = _gaussian(coarse, sigma=1.5)
    regional = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    # 粗网格仅占 3%：双边照明已能干净地保留阴影边界并平滑纸张内部，粗网格
    # 只用于压制极端 inpaint 残差的低频漂移；权重过大会把窄阴影区的照明向
    # 周围亮区抬高，直接削弱阴影增益、留下压不平的暗带。
    fused_illum = 0.97 * np.maximum(illumination, EPS) + 0.03 * regional
    gain = target / np.maximum(fused_illum, EPS)

    # 轻量平滑：只压估计噪声，不改变阴影边界的过渡宽度。增益场的边界几何
    # 完全继承自照明场（gain=target/illum），这里若用大 sigma 会把增益过渡
    # 压窄，与输入半影失配，乘回去反而在边界造出亮暗双边。
    gain = _gaussian(gain, sigma=max(1.0, 1.5 * _scale(image)))

    # 亮区锚定仅在自动（单页）模式下执行：把受光纸张像素的增益中位数归一
    # 到 1，使本页亮区保持原始纸色不被过校正。显式传入批次共享基准时必须
    # 跳过——否则每页都被各自归一化，跨页线性关系被破坏，翻页亮度又会跳。
    paper_lum = _luminance(np.where(valid[..., None], illumination, np.nan))
    lit_cut = np.nanpercentile(paper_lum, 90)
    lit_band = valid & (_luminance(illumination) >= lit_cut)
    if not explicit_target and lit_band.any():
        anchor = np.median(gain[lit_band], axis=0)
        gain = gain / np.maximum(anchor, EPS)

    # 亮区软膝 cap（在平滑后的连续场上按 smoothstep 施加）：照明越接近受光
    # 水平，允许的增益上限越平滑压向 LIT_GAIN_FLOOR。该下限略低于 1，允许
    # 亮区向统一基准轻微收敛，但不会把米黄纸色拉白；软膝连续，主管线再做
    # 一次高斯羽化也不会在阴影边界产生亮暗双边。
    LIT_GAIN_FLOOR = 0.97
    q50 = np.nanpercentile(paper_lum, 50)
    knee = np.clip((_luminance(illumination) - q50) / max(lit_cut - q50, EPS),
                   0.0, 1.0)
    knee = knee * knee * (3.0 - 2.0 * knee)
    cap = LIT_GAIN_FLOOR + (GAIN_CLIP[1] - LIT_GAIN_FLOOR) * (1.0 - knee)
    gain = np.minimum(gain, cap[..., None])

    gain = np.clip(gain, GAIN_CLIP[0], GAIN_CLIP[1])
    return {
        "gain": gain,
        "regional_illumination": regional,
        "grid_medians": coarse,
        "target": target,
        "grid": (rows, cols),
        "lit_band": lit_band,
    }


# ---------------------------------------------------------------------------
# 修复 6：边界羽化
# ---------------------------------------------------------------------------

def edge_feather(field, radius: float = None):
    """对增益场/掩码做边界羽化。

    高斯 sigma 默认按场的分辨率取 ``BASE_FEATHER_SIGMA * scale``，阴影硬边
    在增益场上被过渡为缓坡，乘回去后不再产生亮暗双边。输入为 0/1（或
    0/255）掩码时输出等价软 alpha。

    传入整数时退化为对一个合成硬边阶跃做羽化，便于契约测试与独立演示。
    """
    if isinstance(field, (int, np.integer, float, np.floating)):
        n = 128
        step = np.zeros((n, n), dtype=np.float64)
        step[:, n // 2:] = 1.0
        sigma = radius if radius is not None else 4.0
        return _gaussian(step, sigma)

    field = np.asarray(field, dtype=np.float64)
    if radius is None:
        radius = BASE_FEATHER_SIGMA * (max(field.shape[:2]) / REF_WIDTH)
    radius = max(radius, 1.0)
    return _gaussian(field, radius)


# ---------------------------------------------------------------------------
# 修复 5：批次共享基准
# ---------------------------------------------------------------------------

def _page_paper_stat(image) -> np.ndarray:
    """单页纸张基准（逐通道 75 分位数），用于跨页汇总。"""
    image = _as_image(image)
    protect = text_region_protect(image)
    paper = _inpainted_paper(image, protect["mask"])
    return _paper_target(paper, protect["mask"])


def batch_baseline(pages, max_jump: float = MAX_BATCH_JUMP, alpha: float = 0.25) -> dict:
    """计算同一叠纸共享的纸张基准，并约束逐页校正目标的跳变。

    流程：

    1. 逐页估计纸张本色 ``page_stats``；
    2. 跨页鲁棒中位数作为共享基准 ``shared``；
    3. 每页目标先向共享基准收敛（偏差钳在 ±5%）；
    4. 再顺序钳制相邻目标的步长（不超过共享基准的 5%），保证翻页动画里
       相邻两页增益差不超过 5%；
    5. 同时给出 EWMA 流式基准序列，供逐页到达的在线场景使用。

    Parameters
    ----------
    pages:
        图像列表（数组/路径/整数种子均可），或预先算好的逐页基准数组。
    max_jump:
        允许的最大相对跳变，默认 0.05。
    alpha:
        EWMA 平滑系数。
    """
    if pages is None:
        pages = [1, 2, 3]
    if isinstance(pages, (int, np.integer, float, np.floating)):
        pages = [pages]
    stats = []
    for page in pages:
        if isinstance(page, np.ndarray) and page.ndim == 1 and page.size == 3:
            stats.append(page.astype(np.float64))
        else:
            stats.append(_page_paper_stat(page))
    stats = np.asarray(stats, dtype=np.float64)

    shared = np.median(stats, axis=0)

    # 第一步：单页偏差向共享基准钳制。
    band = max_jump * shared
    targets = shared + np.clip(stats - shared, -band, band)

    # 第二步：顺序钳制相邻页步长，杜绝翻页亮度一跳。
    for i in range(1, len(targets)):
        delta = targets[i] - targets[i - 1]
        targets[i] = targets[i - 1] + np.clip(delta, -band, band)

    # EWMA 流式基准（在线翻页场景）。
    stream = [stats[0].copy()]
    for i in range(1, len(stats)):
        raw = alpha * stats[i] + (1.0 - alpha) * stream[-1]
        delta = raw - stream[-1]
        stream.append(stream[-1] + np.clip(delta, -band, band))

    adjacent = np.abs(np.diff(targets, axis=0))
    max_adjacent = float(np.max(adjacent / np.maximum(shared, EPS))) if len(targets) > 1 else 0.0

    return {
        "shared": shared,
        "page_stats": stats,
        "targets": targets,
        "stream": np.asarray(stream),
        "max_adjacent_delta": max_adjacent,
        "max_jump": max_jump,
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load(source) -> np.ndarray:
    """加载图像为 HxWx3 uint8 RGB。

    路径不存在时抛 ``FileNotFoundError``；输入整数时返回确定性合成文档。
    """
    if isinstance(source, np.ndarray):
        return _as_image(source)
    if isinstance(source, (int, np.integer, float, np.floating)) or source is None:
        return _synthetic_document(int(source) if source is not None else 0)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def export(image, path=None):
    """导出 PNG。

    提供 ``path`` 时写文件并返回路径字符串；否则返回内存中的 PNG 字节
    （契约测试与无盘环境使用）。
    """
    from PIL import Image

    if isinstance(image, dict):
        image = image.get("image", image.get("binary"))
    if not isinstance(image, np.ndarray):
        image = _as_image(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = np.clip(image, 0, 255).astype(np.uint8)

    pil = Image.fromarray(image, mode="RGB")
    if path is None:
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return buffer.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format="PNG")
    return str(path)


# ---------------------------------------------------------------------------
# 主校正管线
# ---------------------------------------------------------------------------

def flatten(image, baseline=None) -> dict:
    """单页阴影压平主管线，串联六个修复点。

    1. :func:`shadow_lowpass_separate` 低通分离照明（文字先 inpaint 剔除）；
    2. :func:`gain_per_region` 分区增益，基准锚定纸张米黄而非纯白；
    3. :func:`edge_feather` 对增益场羽化，消除硬边双边；
    4. 文字区按羽化权重轻微减弱校正强度，防止笔画周边光晕；
    5. 乘性校正保持局部对比，米黄纸色不被拉白。

    Parameters
    ----------
    image:
        输入图像（数组/路径/整数种子）。
    baseline:
        批次共享基准（逐通道 0~1），为 None 时取本页自身纸张基准。
    """
    image = _as_image(image)
    f = _to_float(image)

    separated = shadow_lowpass_separate(image)
    illumination = separated["illumination"]
    weight = separated["weight"]

    region = gain_per_region(image, illumination=illumination, target=baseline)
    target = region["target"]

    # 边界羽化：分区增益场做唯一一次分辨率尺度的高斯羽化。软膝 cap 已在
    # 增益场内施加且连续，羽化只会进一步平滑，不会在阴影边界造出亮暗双边。
    gain = edge_feather(region["gain"])
    corrected = f * gain

    # 文字核心处仅施加 85% 校正，规避笔画边缘光晕；权重本身已羽化。
    protect_strength = 1.0 - 0.15 * weight[..., None]
    corrected = f * (1.0 - protect_strength) + corrected * protect_strength

    # 纸色头空：最后统一硬压到 HEADROOM，米黄本色绝不被拉到纯白。文字像素
    # 远低于该值不受影响；仅削平真正的过冲峰值。
    HEADROOM = 250.0 / 255.0
    corrected = np.minimum(corrected, HEADROOM)
    corrected = np.clip(corrected, 0.0, 1.0)

    out = (corrected * 255.0).round().astype(np.uint8)
    return {
        "image": out,
        "float": corrected,
        "gain": gain,
        "illumination": illumination,
        "mask": separated["mask"],
        "weight": weight,
        "target": target,
        "params": {
            "lowpass_sigma": separated["sigma"],
            "lowpass_kernel": separated["kernel"],
            "feather_sigma": BASE_FEATHER_SIGMA * _scale(image),
            "grid": region["grid"],
            "gain_clip": GAIN_CLIP,
        },
    }


def binarize(flat) -> dict:
    """二值化压平结果，保护细笔画。

    不使用局部自适应阈值（旧做法会把细笔画压平、断成虚线），改为在压平
    后的灰度上做全局 Otsu，再仅以连通域面积剔除噪点；全程不做腐蚀，
    2px 细笔画保持连通。
    """
    if isinstance(flat, dict):
        image = flat["image"]
        mask = flat.get("mask")
    else:
        image = _as_image(flat)
        mask = text_region_protect(image)["mask"]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 连通域滤噪：只删小斑点，不做任何腐蚀/开运算。
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - binary, connectivity=8
    )
    s = _scale(image)
    min_area = max(4.0, (2.0 * s) ** 2)
    removed = 0
    ink = np.zeros_like(gray)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            ink[labels == label] = 255
        else:
            removed += 1

    binary = 255 - ink
    return {
        "binary": binary,
        "threshold": float(threshold),
        "removed_components": int(removed),
        "min_area": float(min_area),
        "ink_coverage": float(np.count_nonzero(ink) / ink.size),
    }


# ---------------------------------------------------------------------------
# 质量保障：校验 / 指标 / 审计 / 兜底 / 确定性
# ---------------------------------------------------------------------------

def validate_shadow(image) -> dict:
    """校验输入与照明估计是否可信，返回结构化原因列表。"""
    reasons = []
    image = _as_image(image)
    f = _to_float(image)

    if not np.isfinite(f).all():
        reasons.append("图像含非有限值")
    if (f <= 0).mean() > 0.5:
        reasons.append("黑场像素占比过高")
    if (f >= 1.0 - EPS).mean() > 0.5:
        reasons.append(" clipped 白场像素占比过高")

    separated = shadow_lowpass_separate(image)
    illumination = separated["illumination"]
    if not np.isfinite(illumination).all() or (illumination <= 0).any():
        reasons.append("照明场非正或含非有限值")

    paper_level = np.median(separated["inpainted"][separated["mask"] == 0])
    shadow_fraction = float(np.mean(
        _luminance(separated["inpainted"]) < 0.85 * paper_level
    ))
    if not (0.005 <= shadow_fraction <= 0.9):
        reasons.append(f"阴影占比异常: {shadow_fraction:.3f}")

    ink_fraction = float(np.count_nonzero(separated["mask"]) / separated["mask"].size)
    if ink_fraction > 0.6:
        reasons.append(f"文字掩码占比过高: {ink_fraction:.3f}")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "shadow_fraction": shadow_fraction,
        "ink_fraction": ink_fraction,
        "paper_level": float(paper_level),
    }


def metrics(before, after=None) -> dict:
    """量化压平质量，覆盖六个修复点关注的可测性质。

    指标含义：

    * ``paper_cv_*``：纸张区亮度变异系数，压平后应显著下降（阴影残余）；
    * ``contrast_retention``：纸张-笔画灰度对比保留率，应接近 1；
    * ``paper_color_shift``：受光区纸张本色漂移（归一化欧氏距离），越小
      越能说明米黄没被拉白；
    * ``seam_ratio``：阴影边界带的梯度能量比，小于 1 说明硬边被羽化；
    * ``bilateral_overshoot``：阴影边界两侧相对局部高原的最大亮/暗过冲，
      直接刻画"亮暗双边"，合格压平应低于 0.05；
    * ``shadow_reduction``：纸张区不均匀度下降比例。
    """
    before = _as_image(before)
    if after is None:
        after = flatten(before)["image"]
    after = _as_image(after)
    a = _to_float(before)
    b = _to_float(after)
    gray_a, gray_b = _luminance(a), _luminance(b)

    protect = text_region_protect(before)
    paper_mask = protect["mask"] == 0
    ink_mask = protect["mask"] > 0

    paper_a = gray_a[paper_mask]
    paper_b = gray_b[paper_mask]
    cv_before = float(np.std(paper_a) / max(np.mean(paper_a), EPS))
    cv_after = float(np.std(paper_b) / max(np.mean(paper_b), EPS))

    if ink_mask.any():
        contrast_before = float(np.median(paper_a) - np.median(gray_a[ink_mask]))
        contrast_after = float(np.median(paper_b) - np.median(gray_b[ink_mask]))
        contrast_retention = contrast_after / max(contrast_before, EPS)
    else:
        contrast_retention = 1.0

    # 受光区（原图最亮的 20% 纸张）纸色漂移，检验米黄是否被拉向纯白。
    lit_threshold = np.percentile(gray_a[paper_mask], 80)
    lit = paper_mask & (gray_a >= lit_threshold)
    color_before = np.median(a[lit], axis=0)
    color_after = np.median(b[lit], axis=0)
    paper_color_shift = float(np.linalg.norm(color_after - color_before))

    # 阴影边界带：输入（inpaint 掉文字后的纸张图）照明梯度最大的 5% 像素。
    # 比较压平前后同一带内纸张像素的梯度能量——羽化成功则输出带内梯度应
    # 显著小于输入，且不会在带外冒出新的双边（取全图 p99 梯度辅助判断）。
    paper_no_text = shadow_lowpass_separate(before)["inpainted"]
    illum = _luminance(paper_no_text)
    grad_y, grad_x = np.gradient(illum)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_band = paper_mask & (grad_mag >= np.percentile(grad_mag[paper_mask], 95))

    grad_in = grad_mag[edge_band].mean()
    gy_b, gx_b = np.gradient(gray_b)
    grad_out_mag = np.sqrt(gx_b ** 2 + gy_b ** 2)
    grad_out = grad_out_mag[edge_band].mean()
    seam_ratio = float(grad_out / max(grad_in, EPS))
    edge_p99 = float(np.percentile(grad_out_mag[paper_mask], 99) /
                     max(np.percentile(grad_mag[paper_mask], 99), EPS))

    # 双边过冲：在阴影边界邻域内，输出若越过各自一侧紧邻的局部高原，即
    # 产生亮边/暗边。逐干净行检测每条强梯度边，高原取边界两侧紧邻的窄带
    # （仍在同一光照区内），只统计纸张像素，避免文字与远处阴影污染。
    col_grad = np.abs(np.gradient(illum, axis=1))
    bright_overshoot = dark_overshoot = 0.0
    grad_threshold = np.percentile(col_grad[paper_mask], 99.5)
    clean_rows = np.where(paper_mask.mean(axis=1) > 0.98)[0]
    for r in clean_rows:
        edge_cols = np.where(col_grad[r] > grad_threshold)[0]
        for edge_x in edge_cols:
            z0, z1 = edge_x - 4, edge_x + 4
            if z0 < 8 or z1 >= gray_b.shape[1] - 8:
                continue
            if not paper_mask[r, z0:z1].all():
                continue
            # 紧邻两侧、各 12px 宽的局部高原。
            left = gray_b[r, edge_x - 16:edge_x - 4]
            right = gray_b[r, edge_x + 4:edge_x + 16]
            zone = gray_b[r, z0:z1]
            plateau_l, plateau_r = float(left.mean()), float(right.mean())
            if plateau_l >= plateau_r:  # 左亮右暗
                bright_overshoot = max(bright_overshoot, float(zone.max()) - plateau_l)
                dark_overshoot = max(dark_overshoot, plateau_r - float(zone.min()))
            else:
                bright_overshoot = max(bright_overshoot, float(zone.max()) - plateau_r)
                dark_overshoot = max(dark_overshoot, plateau_l - float(zone.min()))
    bilateral_overshoot = max(bright_overshoot, dark_overshoot)

    return {
        "paper_cv_before": cv_before,
        "paper_cv_after": cv_after,
        "shadow_reduction": max(0.0, 1.0 - cv_after / max(cv_before, EPS)),
        "contrast_retention": float(contrast_retention),
        "paper_color_shift": paper_color_shift,
        "seam_ratio": seam_ratio,
        "edge_p99_ratio": edge_p99,
        "bilateral_overshoot": float(bilateral_overshoot),
    }


def audit(image, baseline=None) -> dict:
    """对单页跑完整管线并汇总可 JSON 序列化的审计结果。"""
    image = _as_image(image)
    validation = validate_shadow(image)

    if validation["ok"]:
        result = flatten(image, baseline=baseline)
        mode = "flatten"
        reason = None
    else:
        result = fallback(image)
        mode = "fallback"
        reason = validation["reasons"]

    scores = metrics(image, result["image"])
    binary = binarize(result)

    return {
        "mode": mode,
        "reason": reason,
        "shape": list(image.shape),
        "validation": {k: v for k, v in validation.items()},
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in result["params"].items()},
        "target": [float(x) for x in np.asarray(result["target"]).ravel()],
        "metrics": {k: float(v) for k, v in scores.items()},
        "binarize": {
            "threshold": binary["threshold"],
            "removed_components": binary["removed_components"],
            "ink_coverage": binary["ink_coverage"],
        },
    }


def fallback(image) -> dict:
    """兜底校正：校验失败时只施加温和的全局增益，不做分区/除法。"""
    image = _as_image(image)
    f = _to_float(image)
    protect = text_region_protect(image)
    paper = _inpainted_paper(image, protect["mask"])
    target = _paper_target(paper, protect["mask"])
    global_level = np.median(paper[protect["mask"] == 0], axis=0)
    gain = np.clip(target / np.maximum(global_level, EPS),
                   GAIN_CLIP[0], GAIN_CLIP[1])
    corrected = np.clip(f * gain, 0.0, 1.0)
    return {
        "image": (corrected * 255.0).round().astype(np.uint8),
        "float": corrected,
        "gain": np.tile(gain, (*f.shape[:2], 1)),
        "mask": protect["mask"],
        "weight": protect["weight"],
        "target": target,
        "params": {
            "lowpass_sigma": 0.0,
            "lowpass_kernel": 0,
            "feather_sigma": 0.0,
            "grid": (1, 1),
            "gain_clip": GAIN_CLIP,
        },
    }


def determinism(image) -> dict:
    """同输入跑两遍管线，校验逐字节一致（管线不使用任何随机源）。"""
    image = _as_image(image)
    first = flatten(image)
    second = flatten(image)
    same_image = bool(np.array_equal(first["image"], second["image"]))
    same_gain = bool(np.array_equal(first["gain"], second["gain"]))

    bin_first = binarize(first)["binary"]
    bin_second = binarize(second)["binary"]
    same_binary = bool(np.array_equal(bin_first, bin_second))

    max_diff = int(np.max(np.abs(
        first["image"].astype(np.int16) - second["image"].astype(np.int16)
    )))
    return {
        "deterministic": same_image and same_gain and same_binary,
        "same_image": same_image,
        "same_gain": same_gain,
        "same_binary": same_binary,
        "max_abs_diff": max_diff,
    }
