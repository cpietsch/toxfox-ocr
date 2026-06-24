# type: ignore
"""Fast matching experiments on cached OCR (oriented cache + v2 GT via score.py). No OCR re-run.
Edit CONFIGS to try matcher/threshold/filter variants; each scored on scraped + curated.
Build the cache first (build_cache.py orient)."""
import os

import score as S
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor

CONFIGS = [
    ("union@80 frag", {"match_strategy": "union", "segment_threshold": 80, "drop_fragments": True}),
    ("union@80 nofrag", {"match_strategy": "union", "segment_threshold": 80, "drop_fragments": False}),
    ("union@84 frag", {"match_strategy": "union", "segment_threshold": 84, "drop_fragments": True}),
    ("union@88 frag", {"match_strategy": "union", "segment_threshold": 88, "drop_fragments": True}),
    ("trie frag", {"match_strategy": "trie", "drop_fragments": True}),
    ("trie nofrag", {"match_strategy": "trie", "drop_fragments": False}),
    ("segment@84 frag", {"match_strategy": "segment", "segment_threshold": 84, "drop_fragments": True}),
]


def main():
    key, gt_mode = os.environ.get("CACHE", "orient"), os.environ.get("GT", "v2")
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    ev = Evaluation()
    data = {n: S.load(n, key, gt_mode) for n in S.SETS}
    print(f"CACHE={key} GT={gt_mode}")
    print(f"{'config':22}" + "".join(f"{n[:9]:>26}" for n in S.SETS))
    print(f"{'':22}" + "".join(f"{'exF1/lvF1 (P/R)':>26}" for _ in S.SETS))
    for label, ov in CONFIGS:
        for k, v in ov.items():
            setattr(post, k, v)
        row = f"{label:22}"
        for n in S.SETS:
            cache, gts, _ = data[n]
            exF1, lvF1, P, R, _ = S.score_set(post, ev, cache, gts)
            row += f"{exF1:.3f}/{lvF1:.3f} ({P:.2f}/{R:.2f})".rjust(26)
        print(row)


if __name__ == "__main__":
    main()
