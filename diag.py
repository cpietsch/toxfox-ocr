# type: ignore
"""Decompose exact-F1 loss into actionable buckets, using score.py's loaders (oriented cache,
v2 GT). FN buckets: not_in_dict / ocr_visible (matcher headroom) / ocr_missing (OCR headroom).
FP buckets: near_gt (canon/near-miss) / spurious. Reports per-lever ceilings + worst images.

    CACHE=orient GT=v2 python diag.py [strategy]
"""
import os
import sys
from collections import Counter

from rapidfuzz import fuzz, process

import score as S
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def main():
    key, gt_mode = os.environ.get("CACHE", "orient"), os.environ.get("GT", "v2")
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    if len(sys.argv) > 1:
        post.match_strategy = sys.argv[1]
    ev = Evaluation()
    dict_canon = {S.canon(x) for x in post.combined_tokens}
    print(f"dict={len(dict_canon)} strategy={post.match_strategy} CACHE={key} GT={gt_mode}\n")

    for name in S.SETS:
        cache, gts, _ = S.load(name, key, gt_mode)
        fn_cats, fp_cats = Counter(), Counter()
        TP = FP = FN = 0
        exF1s, ceil_m, ceil_mnofp = [], [], []
        worst = []
        for stem, gt in gts.items():
            raw = cache[stem]
            preds = S.norm(post.get_ingredients(raw).get("ingredients", []))
            gset, pset = set(gt), set(preds)
            tp = [p for p in preds if p in gset]
            fp = [p for p in preds if p not in gset]
            fn = [g for g in gt if g not in pset]
            TP += len(tp); FP += len(fp); FN += len(fn)
            exF1s.append(ev.get_metrics(preds, gt, "exact")[0])
            nvis = 0
            for g in fn:
                if g not in dict_canon:
                    fn_cats["not_in_dict"] += 1; continue
                m = process.extractOne(g, raw, scorer=fuzz.partial_ratio) if raw else (None, 0)
                if m and m[1] >= 85:
                    fn_cats["ocr_visible"] += 1; nvis += 1
                else:
                    fn_cats["ocr_missing"] += 1
            for p in fp:
                m = process.extractOne(p, gt, scorer=fuzz.ratio) if gt else (None, 0)
                fp_cats["near_gt" if (m and m[1] >= 88) else "spurious"] += 1
            ceil_m.append(f1(len(tp) + nvis, len(fp), len(fn) - nvis))
            ceil_mnofp.append(f1(len(tp) + nvis, 0, len(fn) - nvis))
            worst.append((exF1s[-1], stem, len(gt), len(preds), fn[:5], fp[:5]))
        n = len(gts)
        print(f"===== {name} (n={n}) exact-F1={sum(exF1s)/n:.4f}  microP/R={TP/(TP+FP):.3f}/{TP/(TP+FN):.3f}")
        print(f"  TP={TP} FP={FP} FN={FN}")
        tfn = sum(fn_cats.values()) or 1
        print("  FN: " + "  ".join(f"{k}={v}({100*v/tfn:.0f}%)" for k, v in fn_cats.most_common()))
        tfp = sum(fp_cats.values()) or 1
        print("  FP: " + "  ".join(f"{k}={v}({100*v/tfp:.0f}%)" for k, v in fp_cats.most_common()))
        print(f"  CEILING recover ocr_visible (keep FP): {sum(ceil_m)/n:.4f}   +drop all FP: {sum(ceil_mnofp)/n:.4f}")
        worst.sort()
        for s, stem, ng, npd, fn, fp in worst[:8]:
            print(f"   {s:.2f} {stem[:20]:20} g{ng:2}p{npd:2} FN{fn} FP{fp}")
        print()


if __name__ == "__main__":
    main()
