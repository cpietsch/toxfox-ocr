# type: ignore
"""OCR the eval sets ONCE, cache the raw tokens to disk (so matching experiments are instant),
and diagnose the matching headroom: GT ingredients that ARE visible in the OCR but go unmatched.
"""
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml
from rapidfuzz import fuzz, process

from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor
from zug_toxfox.modules.preprocessing import PreProcessor

SETS = {
    "scraped": ("data/scraped/ground_truth", "data/scraped/images"),
    "curated": ("data/ground_truth", "data/images"),
    "realworld": ("data/realworld/ground_truth", "data/realworld/images"),
}


def norm(xs):
    return sorted(str(x).lower().strip().replace("*", "") for x in xs)


def main():
    pre, ocr, post = PreProcessor(), OCR(), PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()

    for name, (gtd, imgd) in SETS.items():
        gtd, imgd = Path(gtd), Path(imgd)
        cache = {}
        for f in sorted(gtd.glob("*.yaml")):
            img = next((imgd / f"{f.stem}{e}" for e in (".jpg", ".jpeg", ".png") if (imgd / f"{f.stem}{e}").exists()), None)
            if img is None:
                continue
            try:
                cache[f.stem] = ocr.process_image(pre.preprocess_image(cv2.imread(str(img))), debug=False)
            except Exception:  # noqa: BLE001
                cache[f.stem] = []
        json.dump(cache, open(f"/tmp/cache_{name}.json", "w"))
        print(f"cached {name}: {len(cache)} images -> /tmp/cache_{name}.json")

    # --- matching-headroom diagnosis on the scraped set ---
    cache = json.load(open("/tmp/cache_scraped.json"))
    examples, cats = [], Counter()
    for f in sorted(Path("data/scraped/ground_truth").glob("*.yaml")):
        stem = f.stem
        if stem not in cache:
            continue
        gt = norm(yaml.safe_load(open(f))["INCI_list"])
        raw = cache[stem]
        preds = set(norm(post.get_ingredients(raw).get("ingredients", [])))
        for g in gt:
            if g in preds or len(g) < 4:
                continue
            m = process.extractOne(g, raw, scorer=fuzz.partial_ratio) or (None, 0)
            if m[1] >= 85:  # visible in OCR but unmatched
                cats["score>=95" if m[1] >= 95 else "score85-95"] += 1
                if len(examples) < 45:
                    examples.append((g, m[1], str(m[0])[:60]))
    print("\n=== FN-in-OCR (visible but unmatched) categories ===", dict(cats))
    print("=== examples: GT name | partial-score | closest OCR line ===")
    for g, sc, line in examples:
        print(f"  {g[:32]:32} | {sc:3} | {line}")


if __name__ == "__main__":
    main()
