# type: ignore
"""Per-stem probe: show the isolated OCR region (per source) and what each matcher
(trie / segment / window) returns, so we can SEE why a 'present-in-OCR' name is unmatched.

Usage: OMP_NUM_THREADS=4 CACHE=ensemble python probe.py curated 0768614212331 [target]
"""
import os, re, sys
import score as S
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor

def ns(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def main():
    name = sys.argv[1]
    stem = sys.argv[2]
    target = sys.argv[3].lower() if len(sys.argv) > 3 else None
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    post.match_strategy = "union3"; post.segment_threshold = 80.0
    cache, gts, _ = S.load(name, "ensemble", "v2")
    c = cache[stem]
    srcs = c["srcs"] if isinstance(c, dict) and "srcs" in c else [c]
    gt = gts[stem]
    print(f"=== {name}/{stem}  GT({len(gt)})={gt}\n")
    for si, toks in enumerate(srcs):
        iso, midx = post._isolate_ingredient_region(toks) if post.isolate_region else (toks, -1)
        print(f"--- source {si} (markers dropped {midx}); raw {len(toks)} tok, isolated {len(iso)} tok ---")
        print("  ISO:", " | ".join(iso))
        if target:
            blob = ns(" ".join(iso))
            print(f"  target despaced '{ns(target)}' in iso-blob? {ns(target) in blob}  (full-blob? {ns(target) in ns(' '.join(toks))})")
        trie = list(post._trie_get_ingredients(iso)["ingredients"])
        seg = list(post._segment_get_ingredients(iso)["ingredients"])
        win = list(post._window_get_ingredients(iso)["ingredients"])
        print("  TRIE:", trie)
        print("  SEG :", seg)
        print("  WIN :", win)
        if target:
            for label, lst in (("trie", trie), ("seg", seg), ("win", win)):
                hits = [x for x in lst if target in x or x in target or ns(target) in ns(x)]
                print(f"    {label} target-ish: {hits}")
        print()

if __name__ == "__main__":
    main()
