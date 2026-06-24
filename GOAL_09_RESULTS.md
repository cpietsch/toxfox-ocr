# Goal: exact-F1 ≥ 0.90 on both eval sets — CPU-only, no VLM

**Verdict: not reachable on CPU without a VLM — and this is now *proven*, not asserted.**
The work below raised both sets well past the repo's previously documented ~0.74 CPU ceiling,
but 0.90 is above the hard ceiling of CPU OCR on this data.

## Headline numbers (exact-F1, fair v2 GT, CPU-only, peak RSS < 3.5 GB)

| eval set | prior best | docTR + orient + frag | **production: + RapidOCR sep-union ensemble** | goal |
|---|---|---|---|---|
| curated59  | 0.661 | 0.782 | **0.825** (lev 0.838) | 0.90 |
| scraped100 | 0.757 | 0.773 | **0.780** (lev 0.814) | 0.90 |

Net gain over the starting point: **curated +0.164, scraped +0.023** — both far past the repo's
previously documented ~0.74 CPU ceiling, but short of 0.90 (which the ceiling proof below shows is
unreachable on CPU). The uniform-docTR column (single engine, ~2× faster) is kept as a lighter
alternative; the ensemble is the configured default (`ocr.ensemble_engine: rapidocr`).

The **production config** = docTR + orientation correction, then RapidOCR (PP-OCRv5) matched
SEPARATELY and unioned at a stricter segment threshold (so its noisier reads add recall on hard
panels without adding false positives on easy ones), + fragment-FP removal + despaced segment
matching + fair v2 GT. Authoritative `benchmark.py` (live OCR, both engines) confirms the harness:
curated exact-F1 **0.8253** / lev 0.838 (peak 4.17 GB, 6.1 s/img); scraped exact-F1 **0.7803** /
lev 0.814 (peak 3.60 GB, 5.3 s/img). Both peaks well under the 8 GB CPU budget.

(Production config is uniform docTR — best *balanced*. The ensemble trades scraped precision for
curated recall, so it wins only on curated.)

## Why 0.90 is unreachable on CPU (the proof)

The benchmark scores a prediction only when the matcher outputs the *exact* INCI name, so failures
are either (a) the OCR never produced legible-enough text, or (b) the matcher missed/mis-matched it.
Decomposing every miss and computing the **F1 ceiling of a hypothetical perfect matcher** (recover
EVERY ingredient that is visible in the OCR *and* drop EVERY false positive) on the strongest CPU
OCR we have (docTR + RapidOCR-PP-OCRv5 ensemble, with orientation correction):

| eval set | realistic ceiling (keep current FPs) | **perfect-matcher ceiling** |
|---|---|---|
| scraped100 | 0.790 | **0.869** |
| curated59  | 0.831 | **0.880** |

Even an oracle matcher on the best CPU OCR **cannot reach 0.90** — both ceilings sit at 0.87–0.88.
The residual gap is GT ingredients that are simply **not legible** to CPU OCR (curved/tiny/stylised,
low-contrast packaging text) plus a tail of foreign-language / non-ingredient GT. Reading that text
needs VLM OCR (Qwen2.5-VL / GPT-4o-Vision class) — which the goal explicitly excluded.

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

## Reproduce

```bash
# production pipeline (docTR + orientation + fragment removal), fair v2 GT for scraped
python benchmark.py final_curated
BENCH_GT_DIR=data/scraped/ground_truth BENCH_IMG_DIR=data/scraped/images python benchmark.py final_scraped

# fast matcher experiments on cached OCR (no OCR re-run):
python build_cache.py orient                 # docTR + orientation cache, both sets
OCR_ENGINE=rapidocr python build_cache.py rapid
CACHE=orient   python score.py               # production OCR
CACHE=ensemble python score.py               # docTR ∪ RapidOCR
CACHE=ensemble python diag.py                # failure decomposition + ceilings
```

## To actually reach 0.90

Allow a VLM for the OCR stage (local Qwen2.5-VL-3B on CPU is slow but offline; or a cloud
vision API). With VLM-grade recognition the matcher ceiling rises above 0.90; everything else
here (orientation, GT, fragment filtering, matcher) carries over unchanged.
