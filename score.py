# type: ignore
"""Central scoring harness: score the pipeline's matching on cached OCR, for both eval sets.

Reads OCR from /tmp/cache_<KEY>_<set>.json (KEY via env CACHE, default 'orient'); falls back to
/tmp/cache_<set>.json. GT for the scraped set is RE-EXTRACTED from each yaml's stored
`raw_ingredients_list` (the authoritative Open Beauty Facts text) with clean_gt_v2 -- the same
region-isolation + tokenization the pipeline applies to predictions, so the comparison is fair.
Enable with GT=v2 (default); GT=orig uses the yaml INCI_list verbatim. Curated GT is always used
verbatim (it is the maintainers' clean curated list).

Usage:
    CACHE=orient GT=v2 python score.py                 # score current config
    CACHE=orient python score.py union 80              # strategy + segment_threshold override
"""
import json
import os
import re
import sys
from pathlib import Path

import yaml

from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor, _MARKER_STRIP, _INGREDIENT_MARKER

SETS = {"scraped": "data/scraped/ground_truth", "curated": "data/ground_truth"}

_WATER = {"water", "eau", "wasser", "acqua", "agua"}


def canon(t: str) -> str:
    """Fair, symmetric canonicalization (applied to BOTH predictions and ground truth)."""
    t = str(t).lower().strip().replace("*", "")
    t = re.sub(r"\([^)]*\)", "", t)            # drop parentheticals: 'aqua (water)' -> 'aqua'
    t = re.sub(r"\s+", " ", t).strip().strip(".").strip()   # collapse spaces, drop trailing period
    if t in _WATER or t.startswith("aqua/") or t.startswith("aqua /") or t.startswith("aqua "):
        return "aqua"
    if t in ("fragrance", "parfum/fragrance", "fragrance/parfum", "parfum / fragrance"):
        return "parfum"
    return t


# ingredient-agnostic prose/junk markers (NOT dict-based, so this never inflates recall)
_JUNK = re.compile(r"(?i)caution|questions|call toll|1-?800|www\.|\.com|poison|medical help|"
                   r"art\.?-?\s?no|grossesse|chirurg|régime|teneurs|enthält|verwende|p flaschen|dermatolog|"
                   r"avoid contact|keep out|&gt|&lt|&amp|cont\.?\s*net|floz|recicla|botella|reciclable")


def clean_gt_v2(raw: str) -> list[str]:
    """Re-extract the ingredient list from the raw OBF text.

    Mirrors the pipeline's prediction-side handling: (1) drop everything up to and including the
    'Ingredients:'/'Inhaltsstoffe:' marker (the header + any marketing prose before it), then
    (2) split on the full set of INCI delimiters (comma, semicolon, bullet, newline, period-space),
    which the original comma-only clean_gt missed -- so period/newline-separated names are no longer
    merged into one unmatchable token. Junk filtering is purely textual (prose markers, numbers,
    >7 words), never dictionary membership.
    """
    text = re.split(r"(?i)\b(?:may contain|peut contenir|kann enthalten|\+/-|\+\\-)\b", raw)[0]
    # Strip the (possibly multilingual) header. Real ingredients have no colons, so within the
    # leading header zone the LAST colon ends the header ('INGREDIENTS/INGREDIENTES: Water',
    # '/Sastojci:/Ingredientes: Aqua'). Bound the zone to the first comma (or 120 chars) so a
    # later in-list colon can't truncate the list.
    zone_end = text.find(",")
    zone_end = zone_end if 0 <= zone_end <= 120 else 120
    head = text[:zone_end]
    mm = list(_INGREDIENT_MARKER.finditer(head))
    if mm:
        last_colon = head.rfind(":")
        text = text[last_colon + 1:] if last_colon > mm[0].start() else text[mm[-1].end():]
    out = []
    # NB: do NOT split on newlines -- in the OBF text they are line-wraps INSIDE names
    # ('Sodium\nHydroxide'), not separators. Split on real INCI delimiters + period-space.
    text = text.replace("\n", " ").replace("\r", " ")
    for tok in re.split(r"[,;•·]+|\.\s+", text):
        t = re.sub(r"\([^)]*\)", "", tok)
        t = re.sub(r"\[[^\]]*\]", "", t)
        t = re.sub(r"[*]+", "", t)
        t = re.sub(r"\s+", " ", t).strip().strip(".").strip()
        if len(t) < 3 or re.fullmatch(r"[\d.\s%/+\-]+", t):
            continue
        if re.match(r"^\d{3,}", t):   # token starting with a long number is a code/junk, not an ingredient
            continue
        if _JUNK.search(t) or len(t.split()) > 7:
            continue
        out.append(t)
    return list(dict.fromkeys(out))


def norm(xs):
    return sorted({canon(x) for x in xs if canon(x)})


def _load_cache(key, name):
    cf = f"/tmp/cache_{key}_{name}.json"
    if not os.path.exists(cf):
        cf = f"/tmp/cache_{name}.json"
    return json.load(open(cf)), cf


def load(name, key, gt_mode):
    if key in ("ensemble", "ensemble3"):
        # Match each engine's tokens SEPARATELY then union the results (see
        # PostProcessor.get_ingredients_multi) -- mirrors the production pipeline. Stored as
        # {"srcs": [tokens, ...]} so score_set knows to ensemble.
        engs = ["orient", "rapid"] + (["easy"] if key == "ensemble3" else [])
        loaded = [_load_cache(e, name)[0] for e in engs]
        keys = set().union(*[set(d) for d in loaded])
        cache = {k: {"srcs": [list(d.get(k, [])) for d in loaded]} for k in keys}
        cf = f"{key}({'+'.join(engs)}, sep-union)"
    else:
        cache, cf = _load_cache(key, name)
    gts = {}
    for f in Path(SETS[name]).glob("*.yaml"):
        if f.stem not in cache:
            continue
        d = yaml.safe_load(open(f))
        if name == "scraped" and gt_mode == "v2" and d.get("raw_ingredients_list"):
            gts[f.stem] = norm(clean_gt_v2(d["raw_ingredients_list"]))
        else:
            gts[f.stem] = norm(d["INCI_list"])
    return cache, gts, cf


def score_set(post, ev, cache, gts):
    exs, lvs, TP, FP, FN = [], [], 0, 0, 0
    sec_seg = float(os.environ.get("ENSEMBLE_SECONDARY_SEG", "90"))
    for stem, gt in gts.items():
        try:
            c = cache[stem]
            if isinstance(c, dict) and "srcs" in c:
                srcs = c["srcs"]
                # primary at its own segment threshold; the rest at the stricter secondary cutoff
                sources = [(srcs[0], post.segment_threshold)] + [(s, sec_seg) for s in srcs[1:]]
                res = post.get_ingredients_multi(sources)
            else:
                res = post.get_ingredients(c)
            preds = norm(res.get("ingredients", []))
        except Exception:  # noqa: BLE001
            preds = []
        exs.append(ev.get_metrics(preds, gt, "exact")[0])
        lvs.append(ev.get_metrics(preds, gt, "levenshtein")[0])
        gs = set(gt)
        TP += sum(1 for p in preds if p in gs)
        FP += sum(1 for p in preds if p not in gs)
        FN += sum(1 for g in gt if g not in set(preds))
    n = len(gts)
    P = TP / (TP + FP) if TP + FP else 0
    R = TP / (TP + FN) if TP + FN else 0
    return sum(exs) / n, sum(lvs) / n, P, R, n


def main():
    key = os.environ.get("CACHE", "orient")
    gt_mode = os.environ.get("GT", "v2")
    post = PostProcessor(FAISSIndexer())
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    if len(sys.argv) > 1:
        post.match_strategy = sys.argv[1]
    if len(sys.argv) > 2:
        post.segment_threshold = float(sys.argv[2])
    ev = Evaluation()
    print(f"CACHE={key} GT={gt_mode} strategy={post.match_strategy} seg_thr={post.segment_threshold}")
    for name in SETS:
        cache, gts, cf = load(name, key, gt_mode)
        exF1, lvF1, P, R, n = score_set(post, ev, cache, gts)
        print(f"  {name:9} n={n:3}  exact-F1={exF1:.4f}  lev-F1={lvF1:.4f}  (P={P:.3f} R={R:.3f})  [{cf.split('/')[-1]}]")


if __name__ == "__main__":
    main()
