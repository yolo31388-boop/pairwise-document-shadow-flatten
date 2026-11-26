"""文档阴影压平管线的端到端演示。

生成一张确定性合成文档（米黄纸张 + 透视光衰减 + 台灯硬边阴影 + 细笔画），
跑完整压平/二值化/批次稳定管线，把可视化结果与审计 JSON 写入 artifacts/。

用法：
    python scripts/demo_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from imaging import document_shadow_flatten as mod

OUT = ROOT / "artifacts" / "demo"


def _save_rgb(name: str, image: np.ndarray) -> None:
    path = OUT / name
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 同一叠纸的三页，模拟翻页时 ±10% 的曝光漂移。
    pages = [mod._synthetic_document(seed=i, level=lv)
             for i, lv in enumerate([1.0, 0.9, 1.1])]
    batch = mod.batch_baseline(pages)

    audits = []
    for i, page in enumerate(pages):
        flat = mod.flatten(page, baseline=batch["shared"])
        binary = mod.binarize(flat)
        report = mod.audit(page, baseline=batch["shared"])
        audits.append(report)

        strip = np.concatenate([
            page,
            flat["image"],
            cv2.cvtColor(binary["binary"], cv2.COLOR_GRAY2RGB),
        ], axis=1)
        _save_rgb(f"page{i}_before_flat_binary.png", strip)

        # 增益场（放大到可视范围）与文字保护掩码。
        gain_vis = np.clip(
            mod._luminance(flat["gain"]) / mod.GAIN_CLIP[1] * 255.0, 0, 255
        ).astype(np.uint8)
        cv2.imwrite(str(OUT / f"page{i}_gain.png"), gain_vis)
        cv2.imwrite(str(OUT / f"page{i}_textmask.png"), flat["mask"])

    # 批次稳定后的逐页纸张亮度，应几乎一致。
    paper_means = []
    for i, page in enumerate(pages):
        img = mod.flatten(page, baseline=batch["shared"])["image"]
        m = mod.text_region_protect(img)["mask"] == 0
        paper_means.append(float(mod._luminance(mod._to_float(img))[m].mean()))

    summary = {
        "shared_baseline": [float(x) for x in batch["shared"]],
        "max_adjacent_target_delta": batch["max_adjacent_delta"],
        "paper_means_after_shared_baseline": paper_means,
        "max_paper_gap": float(np.max(np.abs(np.diff(paper_means)))),
        "per_page": audits,
        "determinism": mod.determinism(pages[0]),
    }
    json_path = OUT / "audit.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  wrote {json_path.relative_to(ROOT)}")
    print("\n批次纸张亮度:", [round(x, 4) for x in paper_means],
          "最大页间差:", round(summary["max_paper_gap"], 4))


if __name__ == "__main__":
    main()
