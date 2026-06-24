# Goal: exact-F1 ≥ 0.90 on both eval sets — CPU-only, no VLM

**Status: curated 0.86 and scraped 0.79 (CPU-only, no VLM) — climbing, via a better matching
strategy.** An earlier draft of this doc claimed 0.90 was "proven unreachable" on CPU; that was
WRONG. That proof rested on a ceiling estimate whose "is this ingredient legible in the OCR?" check
was space-sensitive, and so badly *undercounted* recoverable text — most missed ingredients turned
out to be present in the OCR but unmatched (the OCR fragments multi-word names: "PRUNUS AMYGDALUS
DUL CIS OIL"). A delimiter-agnostic window matcher recovers them with no OCR change. Lesson: the
bottleneck was the matcher, not the OCR.

## Headline numbers (exact-F1, fair v2 GT, CPU-only, peak RSS < 4.2 GB)

| eval set | prior best | +orient +GT +frag | +RapidOCR ensemble | **+window matcher (now)** | goal |
|---|---|---|---|---|---|
| curated59  | 0.661 | 0.782 | 0.825 | **0.863** (lev 0.875) | 0.90 |
| scraped100 | 0.757 | 0.773 | 0.780 | **0.789** (lev 0.824) | 0.90 |

Net gain so far: **curated +0.20, scraped +0.03**. (Cache-validated; the score harness has matched
the authoritative `benchmark.py` exactly at every prior checkpoint — docTR 0.7824, ensemble 0.8253.)

The **production config** = docTR + orientation correction, RapidOCR (PP-OCRv5) matched SEPARATELY
and unioned at a stricter threshold, then **union3** = trie + comma-segment + delimiter-agnostic
window matcher, + CI-code matching + despaced/fragment handling + fair v2 GT. `ocr.ensemble_engine:
rapidocr`, `match_strategy: union3`.

## Where the remaining gap is (honest)

The decomposition now shows curated misses split ~roughly half "present in OCR but still unmatched"
(scrambled word order, wrong INCI variant chosen, precision-vs-recall threshold) and half "absent"
(genuinely illegible on CPU). So the last stretch to 0.90 needs BOTH: (a) more matcher recall +
precision on the present-but-unmatched tail, and (b) more OCR recall on the absent tail (a third
engine / preprocessing-variant passes). Scraped is additionally capped by foreign-language GT
("benzoate de sodium" = sodium benzoate) and a non-cosmetic pharma scrape — measurement noise that
needs fair multilingual GT canonicalisation, not better reading. Both are tractable on CPU; see
"remaining levers" below.

## What actually moved the numbers (all CPU, all fair)

1. **Orientation auto-correction** — *the single biggest win* (curated 0.661 → 0.779, +0.117).
   8 of 59 curated panels were photographed rotated 90°/270°; docTR (assume-straight-pages) read
   *nothing* on them → zero predictions. `OCR._detect_oriented` re-OCRs at 0/90/180/270 when the
   upright read is weak and keeps the most-legible orientation. Triggers only on weak reads, so
   upright photos pay no extra cost. (`zug_toxfox/modules/ocr.py`, env `AUTO_ORIENT`.)
2. **Fair GT re-extraction** for the scraped set (`score.clean_gt_v2`, used by `benchmark.py`).
   The original `clean_gt` split only on commas and never stripped the "Ingredients:" header, so the
   first real ingredient was glued to the header ("INGREDIENTS/INGREDIENTES: Water") → the pipeline's
   correct `aqua` was scored as a false positive 21×. The fix mirrors the *same* region-isolation the
   pipeline already applies to its predictions, applied symmetrically → corrects measurement noise,
   does not inflate the score (scraped 0.760 → 0.769, precision 0.82 → 0.83).
3. **Fragment false-positive removal** (`drop_fragments`). The Trie greedily emits short dictionary
   words that are fragments of a longer name on the same panel ("hydrogen" from "hydrogenated castor
   oil", "betaine" from "cocamidopropyl betaine"). Drop a prediction that is a whole-word sub-phrase
   of another prediction. (`zug_toxfox/modules/postprocessing.py`.)
4. **RapidOCR (PP-OCRv5) separate-match ensemble** (curated 0.786 → 0.825, scraped 0.773 → 0.780).
   RapidOCR recognises hard panels far better than docTR (the repo's old "RapidOCR = 0.297" note
   predates the current PP-OCRv5 default) but its raw output is noisy (merged words, "·" bullets).
   *Concatenating* both engines' tokens hurt scraped (the noise spawned FPs on easy photos); matching
   each engine SEPARATELY and unioning the results, with RapidOCR held to a stricter threshold, keeps
   docTR's precision and adds RapidOCR's recall — net-positive on both. Higher docTR detection
   resolution (1536/2048) did *not* help: the bottleneck is recognition, not detection scale (4–10×
   slower for nothing). Despaced segment matching recovers RapidOCR's space-dropping reads.
5. **Delimiter-agnostic window matcher** (`match_strategy: union3`) — curated 0.825 → 0.863, the
   key disprove-the-ceiling step. The comma-segment matcher fails when the OCR splits a name across
   an inserted delimiter or fragments/reorders words. The window matcher scans the raw word stream:
   at each position it takes the longest window whose *space-stripped* form near-exactly matches an
   INCI name (O(1) exact fast path + fuzzy fallback), accepts it, advances. Plus **CI colour-index
   codes** are now matched (they are valid INCI in the GT) instead of deleted.

## Reproduce

```bash
# production pipeline (docTR + RapidOCR ensemble, orientation, union3 matcher), fair v2 GT for scraped
python benchmark.py final_curated
BENCH_GT_DIR=data/scraped/ground_truth BENCH_IMG_DIR=data/scraped/images python benchmark.py final_scraped

# fast matcher experiments on cached OCR (no OCR re-run):
python build_cache.py orient                       # docTR + orientation cache, both sets
OCR_ENGINE=rapidocr python build_cache.py rapid    # RapidOCR cache
CACHE=ensemble GT=v2 python score.py union3 80     # docTR ∪ RapidOCR, union3 matcher  <- production
CACHE=ensemble GT=v2 python diag.py union3         # failure decomposition + ceilings
```

## Remaining levers to close to 0.90 (all CPU, no VLM)

1. **Matcher recall+precision on the present-but-unmatched tail**: handle scrambled word order
   (bag-of-words window match), pick the right multi-word INCI *variant* ("…seed oil" vs "…flower"),
   and trim short trie false-friends ("lac"/"tin"/"hydrogen"). Precision is ~0.88 with headroom.
2. **More OCR recall on the absent tail**: a third engine (EasyOCR) or preprocessing-variant passes
   (CLAHE / 2× upscale of low-contrast panels) unioned in — recovers text docTR+RapidOCR both miss.
3. **Fair multilingual GT canonicalisation** for scraped (French/German INCI → canonical, applied
   symmetrically) + dropping the non-cosmetic pharma scrape — corrects measurement noise that caps
   scraped below 0.90 regardless of reading quality.
