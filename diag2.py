# type: ignore
"""Decompose exact-F1 loss for the PRODUCTION ensemble config (matches score.py exactly).

For both eval sets, run get_ingredients_multi([(primary,80),(secondary,90)]) on the sep-union
ensemble cache, then bucket every FN and FP:

  FN: not_in_dict (GT name absent from our INCI vocab -> GT/canon issue, not readable)
      ocr_visible (despaced-present in OCR blob -> MATCHER headroom, free)
      ocr_missing (absent from OCR -> OCR-recall headroom)
  FP: near_gt (close to some GT name -> canon/variant) / spurious

Usage: OMP_NUM_THREADS=4 CACHE=ensemble GT=v2 python diag2.py [strategy] [primary_seg]
"""
import os, re, sys
from collections import Counter

from rapidfuzz import fuzz, process

import score as S
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor


def ns(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    key, gt_mode = os.environ.get("CACHE", "ensemble"), os.environ.get("GT", "v2")
    strategy = sys.argv[1] if len(sys.argv) > 1 else "union3"
    primary_seg = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    sec_seg = float(os.environ.get("ENSEMBLE_SECONDARY_SEG", "90"))
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    post.match_strategy = strategy
    post.segment_threshold = primary_seg
    ev = Evaluation()
    dict_canon = {S.canon(x) for x in post.combined_tokens}
    print(f"dict={len(dict_canon)} strategy={strategy} primary_seg={primary_seg} sec_seg={sec_seg} CACHE={key}\n")

    for name in S.SETS:
        cache, gts, _ = S.load(name, key, gt_mode)
        fn_cats, fp_cats = Counter(), Counter()
        fn_examples = {"ocr_visible": [], "ocr_missing": [], "not_in_dict": []}
        fp_examples = {"near_gt": [], "spurious": []}
        TP = FP = FN = 0
        exF1s, worst = [], []
        for stem, gt in gts.items():
            c = cache[stem]
            srcs = c["srcs"] if isinstance(c, dict) and "srcs" in c else [c]
            sources = [(srcs[0], primary_seg)] + [(s, sec_seg) for s in srcs[1:]]
            preds = S.norm(post.get_ingredients_multi(sources).get("ingredients", []))
            blob = ns(" ".join(t for s in srcs for t in s))
            gset, pset = set(gt), set(preds)
            tp = [p for p in preds if p in gset]
            fp = [p for p in preds if p not in gset]
            fn = [g for g in gt if g not in pset]
            TP += len(tp); FP += len(fp); FN += len(fn)
            exF1s.append(ev.get_metrics(preds, gt, "exact")[0])
            for g in fn:
                if S.canon(g) not in dict_canon:
                    fn_cats["not_in_dict"] += 1
                    if len(fn_examples["not_in_dict"]) < 40: fn_examples["not_in_dict"].append((stem, g))
                    continue
                g_ns = ns(g)
                sc = fuzz.partial_ratio(g_ns, blob) if len(g_ns) >= 5 else 0
                if sc >= 88:
                    fn_cats["ocr_visible"] += 1
                    if len(fn_examples["ocr_visible"]) < 60: fn_examples["ocr_visible"].append((stem, g, int(sc)))
                else:
                    fn_cats["ocr_missing"] += 1
                    if len(fn_examples["ocr_missing"]) < 40: fn_examples["ocr_missing"].append((stem, g, int(sc)))
            for p in fp:
                m = process.extractOne(p, gt, scorer=fuzz.ratio) if gt else (None, 0, 0)
                if m and m[1] >= 85:
                    fp_cats["near_gt"] += 1
                    if len(fp_examples["near_gt"]) < 40: fp_examples["near_gt"].append((stem, p, m[0], int(m[1])))
                else:
                    fp_cats["spurious"] += 1
                    if len(fp_examples["spurious"]) < 40: fp_examples["spurious"].append((stem, p))
            worst.append((exF1s[-1], stem, len(gt), len(preds), fn, fp))
        n = len(gts)
        P, R = TP/(TP+FP) if TP+FP else 0, TP/(TP+FN) if TP+FN else 0
        print(f"===== {name} (n={n}) exact-F1={sum(exF1s)/n:.4f}  microP/R={P:.3f}/{R:.3f}  TP={TP} FP={FP} FN={FN}")
        tfn = sum(fn_cats.values()) or 1
        print("  FN: " + "  ".join(f"{k}={v}({100*v/tfn:.0f}%)" for k, v in fn_cats.most_common()))
        tfp = sum(fp_cats.values()) or 1
        print("  FP: " + "  ".join(f"{k}={v}({100*v/tfp:.0f}%)" for k, v in fp_cats.most_common()))
        print(f"  --- ocr_visible FN (MATCHER headroom, present but unmatched) [{len(fn_examples['ocr_visible'])}] ---")
        for stem, g, sc in fn_examples["ocr_visible"]:
            print(f"      {stem[:24]:24} '{g}' (sc={sc})")
        print(f"  --- not_in_dict FN (GT/canon) [{len(fn_examples['not_in_dict'])}] ---")
        for stem, g in fn_examples["not_in_dict"]:
            print(f"      {stem[:24]:24} '{g}'")
        print(f"  --- near_gt FP (canon/variant) [{len(fp_examples['near_gt'])}] ---")
        for stem, p, g, sc in fp_examples["near_gt"]:
            print(f"      {stem[:24]:24} pred='{p}' ~ gt='{g}' ({sc})")
        print()


if __name__ == "__main__":
    main()
