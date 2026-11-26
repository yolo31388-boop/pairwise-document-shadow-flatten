"""六个修复点的行为级测试（契约之外的实质验收）。"""

import cv2
import numpy as np
import pytest

import imaging.document_shadow_flatten as mod


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def page():
    return mod._synthetic_document(seed=0)


@pytest.fixture(scope="module")
def hi_res_page():
    """2 倍线性放大的高分辨率页，核长必须随之放大。"""
    small = mod._synthetic_document(seed=0)
    return cv2.resize(small, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def flat(page):
    return mod.flatten(page)


# ---------------------------------------------------------------------------
# 修复 2：核按分辨率
# ---------------------------------------------------------------------------

def test_kernel_scales_with_resolution_and_stays_odd(page, hi_res_page):
    k_small = mod.kernel_by_resolution(max(page.shape[:2]))
    k_large = mod.kernel_by_resolution(max(hi_res_page.shape[:2]))
    assert k_small % 2 == 1 and k_large % 2 == 1
    assert k_large > k_small
    # 2 倍分辨率 → 核长近似 2 倍（奇数取整误差 1px）。
    assert abs(k_large - 2 * k_small) <= 2
    assert k_small >= 3


def test_lowpass_kernel_follows_image_scale(page, hi_res_page):
    s_small = mod.shadow_lowpass_separate(page)
    s_large = mod.shadow_lowpass_separate(hi_res_page)
    assert s_large["kernel"] > s_small["kernel"]
    assert s_large["sigma"] == pytest.approx(2 * s_small["sigma"], rel=0.05)


# ---------------------------------------------------------------------------
# 修复 1：阴影按低通分离，米黄不被拉白
# ---------------------------------------------------------------------------

def test_reflectance_and_illumination_shapes_and_ranges(page):
    sep = mod.shadow_lowpass_separate(page)
    assert sep["illumination"].shape == page.shape
    assert sep["reflectance"].shape == page.shape
    assert np.all(sep["illumination"] > 0)
    assert np.all(np.isfinite(sep["reflectance"]))
    # 阴影区照明显著低于受光区，且分离场是平滑的（低通性质）。
    lum = mod._luminance(sep["illumination"])
    h, w = lum.shape
    shadow = lum[:, int(0.5 * w):int(0.7 * w)]
    lit = lum[:, int(0.05 * w):int(0.2 * w)]
    assert shadow.mean() < 0.8 * lit.mean()


def test_cream_paper_not_pulled_to_white(page, flat):
    before = mod._to_float(page)
    after = mod._to_float(flat["image"])
    paper = flat["mask"] == 0
    # 取原图受光纸张像素，校正后仍保持米黄：R>B 的暖色关系不丢，且不到纯白。
    lit = paper & (mod._luminance(before) > np.percentile(
        mod._luminance(before)[paper], 80))
    warm_before = (before[lit, 0] - before[lit, 2]).mean()
    warm_after = (after[lit, 0] - after[lit, 2]).mean()
    assert warm_before > 0.05
    assert warm_after == pytest.approx(warm_before, abs=0.04)
    assert after[lit].max() < 250 / 255.0


def test_shadow_uniformity_improves(page, flat):
    m = mod.metrics(page, flat["image"])
    assert m["paper_cv_after"] < m["paper_cv_before"]
    assert m["shadow_reduction"] > 0.3


# ---------------------------------------------------------------------------
# 修复 3：文字区保护，细笔画不断
# ---------------------------------------------------------------------------

def test_text_mask_covers_ink_not_paper(page):
    protect = mod.text_region_protect(page)
    # 合成图细笔画行位于 y=60..62（厚度 2px）+ 膨胀边缘。
    band = protect["mask"][59:64, 40:560]
    assert (band > 0).mean() > 0.85
    # 非文字行带不应被大面积误判。
    blank = protect["mask"][80:95, 40:560]
    assert (blank > 0).mean() < 0.05


def test_thin_strokes_remain_connected(flat):
    binary = mod.binarize(flat)["binary"]
    ink = 255 - binary
    # 细笔画行（y≈60，厚度 2px）：该行墨像素在横向上的连通行程应接近整行长度，
    # 若断成虚线则最长行程会远小于行宽。
    row = ink[61, 40:560]
    longest = current = 0
    for v in row:
        current = current + 1 if v else 0
        longest = max(longest, current)
    assert longest > 400


def test_no_local_adaptive_erasure(flat):
    # 二值化墨点覆盖率与输入文字掩码覆盖率同量级（未被压平）。
    binary = mod.binarize(flat)
    mask_coverage = np.count_nonzero(flat["mask"]) / flat["mask"].size
    assert 0.5 * mask_coverage < binary["ink_coverage"] < 2.0 * mask_coverage


# ---------------------------------------------------------------------------
# 修复 4：分区增益补偿透视远端
# ---------------------------------------------------------------------------

def test_regional_gain_boosts_far_shadow_side(page):
    region = mod.gain_per_region(page)
    gain = region["gain"]
    h, w = gain.shape[:2]
    far_shadow = mod._luminance(gain[:, int(0.5 * w):int(0.7 * w)])
    near_lit = mod._luminance(gain[:, int(0.05 * w):int(0.2 * w)])
    assert far_shadow.mean() > 1.15 * near_lit.mean()
    assert gain.min() >= mod.GAIN_CLIP[0] and gain.max() <= mod.GAIN_CLIP[1]


def test_far_end_no_longer_darker_after_flatten(page, flat):
    before = mod._luminance(mod._to_float(page))
    after = mod._luminance(mod._to_float(flat["image"]))
    paper = flat["mask"] == 0
    far, near = after[:, :100][paper[:, :100]], after[:, -100:][paper[:, -100:]]
    # 原图远端更暗，压平后两端纸张亮度差应显著收窄。
    assert before[:, :100][paper[:, :100]].mean() < \
        before[:, -100:][paper[:, -100:]].mean()
    assert abs(far.mean() - near.mean()) < 0.06


# ---------------------------------------------------------------------------
# 修复 5：批次共享基准，相邻页跳变 ≤ 5%
# ---------------------------------------------------------------------------

def test_batch_jump_clamped():
    # 相邻页纸张亮度差远大于 5%，模拟翻页时的曝光漂移。
    pages = [mod._synthetic_document(seed=i, level=level)
             for i, level in enumerate([1.0, 0.8, 1.05, 0.9])]
    batch = mod.batch_baseline(pages)
    assert batch["max_adjacent_delta"] <= mod.MAX_BATCH_JUMP + 1e-9
    # 共享基准是逐页统计的鲁棒中位数。
    assert np.allclose(batch["shared"], np.median(batch["page_stats"], axis=0))


def test_batch_baseline_stabilizes_page_brightness():
    pages = [mod._synthetic_document(seed=i, level=level)
             for i, level in enumerate([1.0, 0.9, 1.1])]
    batch = mod.batch_baseline(pages)
    # 同一叠纸共享同一基准：所有页都向 shared 基准校正，翻页亮度一致。
    flat_pages = [mod.flatten(p, baseline=batch["shared"])["image"]
                  for p in pages]
    paper_means = []
    for img in flat_pages:
        mask = mod.text_region_protect(img)["mask"] == 0
        paper_means.append(mod._luminance(mod._to_float(img))[mask].mean())
    gaps = np.abs(np.diff(paper_means))
    assert np.all(gaps < 0.06)


# ---------------------------------------------------------------------------
# 修复 6：边界羽化，消除亮暗双边
# ---------------------------------------------------------------------------

def test_feather_smooths_hard_edge():
    step = np.zeros((128, 128))
    step[:, 64:] = 1.0
    soft = mod.edge_feather(step, radius=4.0)
    # 硬阶跃在边界处梯度极大，羽化后峰值梯度应显著下降。
    hard_grad = np.abs(np.diff(step, axis=1)).max()
    soft_grad = np.abs(np.diff(soft, axis=1)).max()
    assert soft_grad < 0.25 * hard_grad
    assert 0.0 < soft[64, 64] < 1.0  # 边界处为中间过渡值


def test_seam_gradient_reduced(page, flat):
    m = mod.metrics(page, flat["image"])
    assert m["seam_ratio"] < 0.6
    # 直接刻画亮暗双边：边界两侧相对局部高原的过冲必须很小。
    assert m["bilateral_overshoot"] < 0.05


# ---------------------------------------------------------------------------
# 质量保障入口
# ---------------------------------------------------------------------------

def test_determinism(page):
    report = mod.determinism(page)
    assert report["deterministic"] is True
    assert report["max_abs_diff"] == 0


def test_validate_audit_fallback_and_export(page, flat, tmp_path):
    assert mod.validate_shadow(page)["ok"] is True
    audit = mod.audit(page)
    assert audit["mode"] == "flatten"
    assert audit["metrics"]["shadow_reduction"] > 0.3

    # 全黑异常图必须走兜底而不是抛异常。
    broken = np.zeros_like(page)
    bad = mod.validate_shadow(broken)
    assert bad["ok"] is False and bad["reasons"]
    fb = mod.fallback(broken)
    assert fb["image"].shape == page.shape

    out = mod.export(flat["image"], tmp_path / "flat.png")
    assert mod.load(out).shape == page.shape
    assert isinstance(mod.export(flat["image"]), bytes)
