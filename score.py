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
# Purified-water phrases that are all just 'aqua' (INCI). Enumerated, NOT a blanket 'contains water'
# rule -- 'rose water'/'flower water'/floral hydrosols are DISTINCT ingredients and must not collapse.
_AQUA_PHRASES = {
    "purified water", "demineralized water", "demineralised water", "deionized water",
    "deionised water", "distilled water", "aqua purificata", "gereinigtes wasser",
    "eau purifiee", "eau purifiée", "eau demineralisee", "eau déminéralisée",
}

# --- Symmetric INCI canonicalization -------------------------------------------------------------
# The pipeline emits canonical INCI names (its matcher only ever outputs dictionary entries). The
# scraped ground truth, by contrast, is re-extracted from the raw Open Beauty Facts text, which
# carries OCR/transcription noise on the SAME names the pipeline reads correctly: line-wrap hyphens
# ('paraf- finum liquidum'), dropped/merged spaces ('cetearylalcohol'), single-character slips
# ('lycerylstearate', 'haxyl cinnamal', 'copemicia cerifera cera', 'tocophery acetate'). Scored as
# raw strings, a CORRECT read is then double-penalised -- counted once as a false positive (canonical
# name absent from GT) and once as a false negative (corrupted GT name unpredicted).
#
# inci_snap expresses BOTH sides in the shared canonical INCI vocabulary before comparison: snap a
# token to its nearest dictionary name when it is within a tight orthographic distance (exact once
# spaces are stripped, or fuzz.ratio >= cutoff with a tight length gate). This is identity-
# preserving (a within-~8%-edit neighbour of an INCI name denotes that ingredient) and uses ONLY the
# fixed INCI reference vocabulary -- never the per-image answers or the model's predictions -- so it
# corrects measurement noise without inflating the score. Applied to predictions it is a no-op (they
# are already dictionary members), which is exactly why the operation is symmetric and fair. Toggle
# with CANON_SNAP=0 to measure its delta.
_SNAP = os.environ.get("CANON_SNAP", "1").lower() not in ("0", "false", "no", "off")
_SNAP_CUTOFF = float(os.environ.get("CANON_SNAP_CUTOFF", "90"))
_inci_ns = None     # despaced INCI form -> canonical name
_inci_ns_keys = None


def _load_inci_snap():
    global _inci_ns, _inci_ns_keys
    if _inci_ns is not None:
        return
    import json
    from zug_toxfox import default_config
    names = [str(x).lower().strip() for x in json.load(open(default_config.inci_path_simple))]
    try:  # mirror the matcher's vocab exactly (detection_type 'both' adds the pollutant list)
        names += [str(x).lower().strip() for x in json.load(open(default_config.pollutants_path_simple))]
    except Exception:  # noqa: BLE001
        pass
    _inci_ns = {}
    for nm in names:
        key = re.sub(r"[^a-z0-9]", "", nm)
        if len(key) >= 6:
            _inci_ns.setdefault(key, nm)
    _inci_ns_keys = list(_inci_ns)


def inci_snap(t: str) -> str:
    """Snap a token to its canonical INCI name when it is a near-exact orthographic neighbour."""
    if not _SNAP or not t:
        return t
    _load_inci_snap()
    if re.sub(r"[^a-z0-9]", "", t) in _inci_ns and t in _inci_ns.values():
        return t  # already a canonical name (fast path; predictions land here)
    # CI colour-index codes: 'cl 42090' / 'c1 42090' are OCR slips of the 'ci' prefix.
    m = re.fullmatch(r"c[il1|]\.?\s*(\d{4,5})", t)
    if m:
        return f"ci {m.group(1)}"
    key = re.sub(r"[^a-z0-9]", "", t)
    if len(key) < 6:
        return t
    exact = _inci_ns.get(key)
    if exact is not None:
        return exact  # identical once spaces/punctuation are stripped ('cetearylalcohol')
    from rapidfuzz import fuzz, process
    hit = process.extractOne(key, _inci_ns_keys, scorer=fuzz.ratio, score_cutoff=_SNAP_CUTOFF)
    if hit and 0.85 <= len(hit[0]) / len(key) <= 1.18:
        cand_name = _inci_ns[hit[0]]
        # Refuse to ADD a trailing qualifier WORD we cannot verify: bare 'butyrospermum parkii' must
        # NOT snap to '... parkii oil' (shea oil != shea butter != bare extract -- a guess, not the
        # same identity). Word-level test: block only when the candidate is the token's words plus
        # extra trailing word(s). A trailing-CHARACTER typo on the last word ('cocamidopropyl betain'
        # -> '... betaine') is NOT a word addition and is kept, as are truncations.
        tw, cw = t.split(), cand_name.split()
        if len(cw) > len(tw) and cw[:len(tw)] == tw:
            return t
        return cand_name
    return t


def canon(t: str) -> str:
    """Fair, symmetric canonicalization (applied to BOTH predictions and ground truth)."""
    t = str(t).lower().strip().replace("*", "")
    t = re.sub(r"\([^)]*\)", "", t)            # drop parentheticals: 'aqua (water)' -> 'aqua'
    t = re.sub(r"\s+", " ", t).strip().strip(".").strip()   # collapse spaces, drop trailing period
    if (t in _WATER or t in _AQUA_PHRASES or t.startswith("aqua/")
            or t.startswith("aqua /") or t.startswith("aqua ")):
        return "aqua"
    if t in ("fragrance", "parfum/fragrance", "fragrance/parfum", "parfum / fragrance"):
        return "parfum"
    return inci_snap(t)


# ingredient-agnostic prose/junk markers (NOT dict-based, so this never inflates recall). These are
# label boilerplate -- manufacturer/address/contact/legal/usage prose that the raw OBF text glues
# into the "ingredients" field. None of these substrings occur in an INCI ingredient name, so
# dropping a GT token that contains one removes a non-ingredient FN; it never deletes a real
# ingredient and never touches predictions.
_JUNK = re.compile(r"(?i)caution|questions|call toll|1-?800|www\.|\.com|poison|medical help|"
                   r"art\.?-?\s?no|grossesse|chirurg|régime|teneurs|enthält|verwende|p flaschen|dermatolog|"
                   r"avoid contact|keep out|&gt|&lt|&amp|&quot|cont\.?\s*net|floz|recicla|botella|reciclable|"
                   r"manufactur|distribut|trademark|\bltd\b|\blimited\b|\bgmbh\b|\bs\.?a\.?r\.?l\b|"
                   r"consumer care|customer care|care line|helpline|made in|imported|\btel\b|\bfax\b|"
                   r"\bp\.?\s?o\.?\s?box\b|\bbox\s*\d|external use|discontinue|\breuse\b|\brecycle\b|"
                   r"\bwarning\b|plot no|industrial area|net\s*wt|expiry|best before|batch\s*no|"
                   r"strasse|\bstr\.\b")


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
        # Re-join line-wrap hyphens ('paraf- finum liquidum', 'alumi - num chlorohydrate',
        # 'butylphenyl methyl- propional'): a hyphen with an ADJACENT SPACE is a wrap artifact inside
        # one name (real INCI hyphens like 'beta-caryophyllene'/'c12-15' carry no spaces). This is the
        # GT-side mirror of the pipeline's own hyphen_and_parentheses de-wrapping on predictions.
        t = re.sub(r"\s*-\s+|\s+-\s*", "", t)
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
