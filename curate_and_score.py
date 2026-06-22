# type: ignore
"""Curate the scraped set to images that actually SHOW the ingredient panel, and score
trie vs segment on all images and on the valid subset. One OCR pass (cached).

Validity is judged by whether the GT ingredients are visible in the raw OCR (fair: it checks the
IMAGE depicts the list, independent of whether matching then succeeds), not by pipeline success.
"""
import json
import os
from pathlib import Path

import cv2
import yaml
from rapidfuzz import fuzz, process

from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor
from zug_toxfox.modules.preprocessing import PreProcessor

GT_DIR = Path("data/scraped/ground_truth")
IMG_DIR = Path("data/scraped/images")


def norm(items):
    return sorted(str(x).lower().strip().replace("*", "") for x in items)


def coverage(gt, raw_lines):
    """Fraction of GT ingredient names that fuzzily appear in the raw OCR lines."""
    if not raw_lines:
        return 0.0
    hit = 0
    for name in gt:
        m = process.extractOne(name, raw_lines, scorer=fuzz.partial_ratio)
        if m and m[1] >= 85 and len(name) >= 4:
            hit += 1
    return hit / max(1, len(gt))


def main():
    cases = []
    for f in sorted(GT_DIR.glob("*.yaml")):
        img = IMG_DIR / f"{f.stem}.jpg"
        if img.exists():
            cases.append((f.stem, img, norm(yaml.safe_load(open(f))["INCI_list"])))

    pre, ocr, post = PreProcessor(), OCR(), PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    ev = Evaluation()

    raw, cov = {}, {}
    for stem, img_path, gt in cases:
        r = ocr.process_image(pre.preprocess_image(cv2.imread(str(img_path))), debug=False)
        raw[stem] = r
        cov[stem] = coverage(gt, r)
    valid = {stem for stem in raw if cov[stem] >= 0.40}
    print("valid panels (GT visible in OCR >=40%%): %d / %d" % (len(valid), len(cases)))
    print("invalid (data quality):", sorted(s for s in raw if s not in valid))

    for strat in ["trie", "segment", "auto"]:
        post.match_strategy = strat
        allf, valf = [], []
        for stem, img_path, gt in cases:
            try:
                preds = norm(post.get_ingredients(raw[stem]).get("ingredients", []))
            except Exception:  # noqa: BLE001
                preds = []
            f1, _ = ev.get_metrics(preds, gt, "exact")
            lf1, _ = ev.get_metrics(preds, gt, "levenshtein")
            allf.append((f1, lf1))
            if stem in valid:
                valf.append((f1, lf1))
        n, nv = len(allf), len(valf)
        print("  [%-7s] all(n=%d): exF1=%.3f lvF1=%.3f | valid(n=%d): exF1=%.3f lvF1=%.3f" % (
            strat, n, sum(a for a, _ in allf) / n, sum(b for _, b in allf) / n,
            nv, sum(a for a, _ in valf) / nv, sum(b for _, b in valf) / nv))


if __name__ == "__main__":
    main()
