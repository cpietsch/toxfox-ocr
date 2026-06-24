# type: ignore
"""For each MISSED GT ingredient on the ensemble OCR, is it actually present in the OCR text
(despaced, low bar) -> matcher headroom, or genuinely absent -> OCR headroom? Print worst images
with full OCR blob so we can SEE what's recoverable."""
import os, re
from collections import Counter
import score as S
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor
from rapidfuzz import fuzz

def ns(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def main():
    post = PostProcessor(FAISSIndexer()); post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    ev = Evaluation(); sec = 90.0
    for name in S.SETS:
        cache, gts, _ = S.load(name, "ensemble", "v2")
        cats = Counter(); worst = []
        for stem, gt in gts.items():
            c = cache[stem]
            blob = ns(" ".join(c["p"]) + " " + " ".join(c["s"]))   # despaced full OCR (both engines)
            preds = set(S.norm(post.get_ingredients_ensemble(c["p"], c["s"], sec).get("ingredients", [])))
            missed = [g for g in gt if g not in preds]
            f1 = ev.get_metrics(sorted(preds), gt, "exact")[0]
            recov = []
            for g in missed:
                g_ns = ns(g)
                # is the (despaced) ingredient present as a near-substring of the despaced blob?
                sc = fuzz.partial_ratio(g_ns, blob) if len(g_ns) >= 5 else 0
                if sc >= 88: cats["recoverable>=88"] += 1; recov.append(g)
                elif sc >= 78: cats["partial78-88"] += 1
                else: cats["absent<78"] += 1
            worst.append((f1, stem, gt, missed, recov))
        tot = sum(cats.values()) or 1
        print(f"\n===== {name}: missed-GT recoverability (despaced) =====")
        print("  " + "  ".join(f"{k}={v}({100*v/tot:.0f}%)" for k, v in cats.most_common()))
        worst.sort()
        print("  --- 6 worst images: missed GT, and which are present-in-OCR(*) ---")
        for f1, stem, gt, missed, recov in worst[:6]:
            rs = set(recov)
            tagged = [f"{g}{'*' if g in rs else ''}" for g in missed]
            print(f"  [{f1:.2f}] {stem}: missed={tagged}")

if __name__ == "__main__":
    main()
