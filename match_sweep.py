# type: ignore
"""Fast matching experiments: score all eval sets from the cached OCR tokens (no OCR re-run).
Edit CONFIGS to try matcher/threshold variants; each is scored on scraped / curated / realworld.
Run ocr_cache_diag.py first to build /tmp/cache_*.json.
"""
import json
from pathlib import Path

import yaml

from benchmark import canon
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor

SETS = {
    "scraped": "data/scraped/ground_truth",
    "curated": "data/ground_truth",
    "realworld": "data/realworld/ground_truth",
}


def norm(xs):
    return sorted({canon(x) for x in xs if canon(x)})


def load(name):
    cache = json.load(open(f"/tmp/cache_{name}.json"))
    gts = {}
    for f in Path(SETS[name]).glob("*.yaml"):
        if f.stem in cache:
            gts[f.stem] = norm(yaml.safe_load(open(f))["INCI_list"])
    return cache, gts


def main():
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    ev = Evaluation()
    data = {n: load(n) for n in SETS}

    # (label, attribute overrides applied to `post` before scoring)
    CONFIGS = [
        ("union@80", {"match_strategy": "union", "segment_threshold": 80}),
        ("union@82", {"match_strategy": "union", "segment_threshold": 82}),
        ("union@84", {"match_strategy": "union", "segment_threshold": 84}),
        ("union@85", {"match_strategy": "union", "segment_threshold": 85}),
        ("union@86", {"match_strategy": "union", "segment_threshold": 86}),
    ]

    print(f"{'config':28}" + "".join(f"{n[:9]:>20}" for n in SETS))
    print(f"{'':28}" + "".join(f"{'exF1/lvF1':>20}" for _ in SETS))
    for label, ov in CONFIGS:
        for k, v in ov.items():
            setattr(post, k, v)
        row = f"{label:28}"
        for n in SETS:
            cache, gts = data[n]
            exs, lvs = [], []
            for stem, gt in gts.items():
                try:
                    preds = norm(post.get_ingredients(cache[stem]).get("ingredients", []))
                except Exception:  # noqa: BLE001
                    preds = []
                exs.append(ev.get_metrics(preds, gt, "exact")[0])
                lvs.append(ev.get_metrics(preds, gt, "levenshtein")[0])
            row += f"{sum(exs)/len(exs):.3f}/{sum(lvs)/len(lvs):.3f}".rjust(20)
        print(row)


if __name__ == "__main__":
    main()
