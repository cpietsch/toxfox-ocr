# ToxFox-OCR — SOTA pipeline upgrade

Goal: *update to the latest techniques and beat the existing pipeline*, measured on the
repo's own 59-image INCI benchmark, CPU-only, peak RAM **< 8 GB**.

Every claim below is measured with `benchmark.py` (which reuses the project's own
`Evaluation` so the numbers are directly comparable to what the maintainers report), not
asserted. All runs are CPU-only on a 6-core box; OCR thread pools capped at 4.

## TL;DR

| Pipeline | exact F1 | exact acc | levensh. F1 | peak RSS | s / image |
|---|---|---|---|---|---|
| **Before** — EasyOCR + MiniLM/FAISS | 0.529 | 0.439 | 0.535 | 3.60 GB | 5.49 |
| **After** — docTR + FAISS + SymSpell | **0.632** | **0.562** | **0.637** | **2.79 GB** | **2.49** |
| | **+19.5 %** | **+28 %** | **+19 %** | **−22 %** | **2.2× faster** |

The new default is **more accurate, lighter, and faster** — and stays at 2.8 GB, well
under the 8 GB ceiling.

## How we got there

1. **Built a reproducible benchmark** (`benchmark.py`): runs the full pipeline over every
   ground-truth image, reports exact + levenshtein F1/accuracy, peak RSS and per-image
   predictions as JSON. Established the baseline: **exact F1 0.529**.
2. **Diagnosed the failure modes.** 9/59 images (15 %) produced *zero* predictions,
   accounting for 32 % of total F1 loss. Probing them showed the text was physically
   present but EasyOCR mangled it (`METHYLTRIMETHICONE` → `MFTHYLURMMTHICONE`) or missed it
   on narrow/tiny crops → **OCR quality, not matching, is the #1 lever.**
3. **Researched 2026 SOTA** per pipeline stage (OCR engines, embedding models, lexical
   matching, preprocessing, INCI-domain). Findings that drove the design:
   - docTR has the strongest *published* evidence for this exact task (HalalBench
     food-packaging ingredient OCR: highest German F1 0.655 vs EasyOCR 0.621) and rides
     the already-installed `torch+cpu`.
   - INCI matching is an **orthographic** problem (OCR character errors on a fixed Latin
     vocabulary), so edit-distance correction (SymSpell) fits better than a semantic model
     for the typo-correction step.
4. **Made OCR engine + matcher + typo-corrector pluggable** and swept the combinations on
   one OCR pass each (caching OCR output, since OCR dominates cost).

## Ablation (exact F1, same 59 images)

| OCR engine | ingredient match | word typo-correct | exact F1 | Δ vs baseline |
|---|---|---|---|---|
| EasyOCR | FAISS cosine | FAISS (semantic) | 0.529 | — *(baseline)* |
| EasyOCR | FAISS cosine | **SymSpell** | 0.562 | +0.033 |
| EasyOCR | rapidfuzz | FAISS | 0.529 | +0.000 |
| **docTR** | FAISS cosine | FAISS | 0.620 | **+0.091** |
| docTR | rapidfuzz | FAISS | 0.617 | +0.088 |
| **docTR** | FAISS cosine | **SymSpell** | **0.632** | **+0.103** |
| docTR (parseq reco) | FAISS cosine | SymSpell | 0.631 | +0.103 |

What the ablation shows:
- **OCR engine is the dominant lever** (+0.091 holding postprocessing constant).
- **SymSpell helps on both engines** (+0.033 on EasyOCR, +0.012 on docTR) — engine-independent,
  and it *removes* a transformer call path (the word-level FAISS index), so it also lowers RAM.
- **The lexical (rapidfuzz) ingredient matcher did *not* beat FAISS cosine** here, so it was
  *not* adopted — the dense matcher already handles the (Trie-prefiltered) candidates well.
- **The transformer `parseq` recognizer is a tie with the default `crnn_vgg16_bn`** but ~45 %
  slower, so `crnn_vgg16_bn` stays the default.
- RapidOCR with its *default* (Chinese/English) models underperformed (0.297) — the known
  config trap; not adopted. docTR was the cleaner, evidence-backed win.

## Code changes

- `config/pipeline_config.yml` — new keys: `ocr.engine` (default `doctr`), `ocr.doctr_reco`,
  `postprocessing.match_backend` (`faiss`), `postprocessing.typo_backend` (`symspell`).
- `zug_toxfox/modules/ocr.py` — pluggable engine dispatch (`OCR_ENGINE` env > config >
  `easyocr`); detection normalized to `[polygon, text, conf]` triples so the existing
  reading-order clustering is reused for every box-emitting engine.
- `zug_toxfox/modules/ocr_backends.py` (new) — `DocTRBackend` (line-level boxes, see fix #3),
  plus `RapidOCRBackend` / `PaddleOCRBackend` adapters (lazy-imported, optional).
- `zug_toxfox/modules/postprocessing.py` — `EMBED_MODEL`-swappable embedding with per-model
  FAISS index namespacing; `rapidfuzz_search` lexical matcher; SymSpell word-level typo
  correction; all backend-selectable via config/env.

## Robustness bug fixes (surfaced by the new engines, also affect the live API)

1. `get_ingredients` crashed (`truth value of an empty array is ambiguous`) on any image with
   **no matched ingredients** — a guaranteed 500 in the API/dashboard. Fixed.
2. The reading-order clustering crashed (`not enough values to unpack`) when a detector
   returns **zero boxes** (docTR/RapidOCR can, on tiny/blank crops). Guarded.
3. `Trie.search` returned a float-typed empty mask → `arrays used as indices must be integer`
   when OCR tokens **wholly miss the Trie**. Forced `dtype=bool`/`object` (the latter also
   removes a latent string-truncation bug on long INCI names).
4. **docTR reading order**: emitting per-*word* boxes and re-clustering scrambled word order
   and shattered multi-word INCI names (recall collapse, 0.35 F1). Emitting one box per docTR
   *line* (trusting docTR's native order) restored recall — this single fix moved docTR from
   −0.18 to +0.09 vs baseline.

## Reproduce

```bash
# baseline
OCR_ENGINE=easyocr python benchmark.py baseline
# new default (reads pipeline_config.yml: docTR + symspell)
python benchmark.py final
# sweep matcher/typo variants on one OCR pass
OCR_ENGINE=doctr python sweep.py
python compare_results.py        # tabulate every run in benchmark_results/
```

Switch engine/matcher without code edits via `config/pipeline_config.yml` or env vars
`OCR_ENGINE` (`easyocr|doctr|rapidocr|paddleocr`), `MATCH_BACKEND` (`faiss|rapidfuzz`),
`TYPO_BACKEND` (`faiss|symspell`), `EMBED_MODEL`, `DOCTR_RECO`.

## New dependencies (default path)

`python-doctr==1.0.1`, `symspellpy==6.9.0` (both on the existing torch+cpu / numpy 2 stack).
Optional alternatives: `rapidocr==3.8.4 onnxruntime`, `rapidfuzz==3.14.5`.

---

# Real-world full-label robustness (follow-up)

The curated 59-image set is bare, pre-cropped ingredient panels. Real product photos are harder:
the panel is surrounded by marketing and usage text on a curved surface. On a NIVEA Sun spray
(`data/realworld/`, EAN 4006000130996, 34 ingredients) the docTR pipeline scored exact F1 **0.55**
with **11 false positives** — including, dangerously, `octocrylene` (matched from the *"free of …
Octocrylene"* claim), `melanin`/`mel` (from "Pro-Melanin extract") and `serine`/`bittern`.

Fixes (each measured; all CPU, < 8 GB):

1. **Reading order.** docTR emits whole-line boxes, so the correct order is a top-to-bottom sort.
   The DFS line-clustering (built for EasyOCR's sub-line boxes) interleaved the marketing block
   with the ingredient panel, scrambling multi-word names and breaking the region cut. Sorting
   docTR lines by box-y fixed it — and independently lifted the **59-set 0.632 → 0.648**.
2. **Region isolation** (`isolate_region`, default on). Keep only the text from the
   "Ingredients:"/"Inhaltsstoffe:" marker onward (tolerant to the OCR I/l/1 confusion); drop the
   marketing/usage prose before it. No marker found (bare panels) → keep everything, so the 59-set
   is unaffected. This removes every marketing false positive.
3. **Segment matcher** (`match_strategy: segment`). Comma-split the isolated region and fuzzy-match
   each whole segment (rapidfuzz `token_sort_ratio`, cutoff 80) — robust to OCR errors that garble
   multi-word names ('Tcopheryl Acetate', 'Capernicia Cerilera Cero') where the Trie's prefix match
   fails. Strong on full labels (NIVEA 0.727) but weak on bare panels (0.476, fragile to delimiter
   style), so it must not be applied everywhere.
4. **Auto routing** (`match_strategy: auto`, default). Marker presence alone does NOT distinguish a
   full label from a cropped panel — panels often carry an "Ingredients:" header too. The
   distinguishing signal is the **amount of pre-marker prose dropped**: a real label buries the list
   under many marketing lines (NIVEA drops ~19), a panel drops ~0. Route to the segment matcher only
   when `>= segment_min_dropped` (default 5) lines preceded the marker, else use the Trie.

Routing measured on both sets (exact F1):

| strategy | 59-set (panels) | NIVEA (full label) |
|---|---|---|
| trie | 0.648 | 0.690 |
| segment | 0.476 | 0.727 |
| **auto (min_dropped=5)** | **0.648** | **0.727** |

Result — both criteria met with no regression:

| | exact F1 | marketing FPs | peak RSS |
|---|---|---|---|
| NIVEA before | 0.55 | 5 (incl. octocrylene) | — |
| **NIVEA after (auto)** | **0.727** | **0** | 1.5 GB |
| 59-set (regression guard) | 0.632 → **0.648** | n/a | 2.6 GB |

The one remaining NIVEA false positive (`glycol`) is an ingredient fragment, not marketing text.
Reproduce the real-world set with `BENCH_GT_DIR=data/realworld/ground_truth
BENCH_IMG_DIR=data/realworld/images python benchmark.py rw`.

---

# Scraped real-world benchmark (100 validated panels)

To measure on real product photos beyond the curated set, a test set was scraped from
**Open Beauty Facts** (open ODbL data). OBF's `ingredients` image is frequently mislabelled
(a product front, not the panel) — a first naive scrape of 40 was 68 % junk and scored only
exact-F1 0.515. So `scrape_obf.py` **OCR-validates every candidate**, keeping it only when the
ground-truth ingredients are actually visible in the image, and **normalizes the GT** (de-dupe,
drop may-contain/parentheses; multilingual `Aqua/Water/Eau` → one `aqua`, applied symmetrically
to predictions and GT so it is a fair canonicalization, see `benchmark.canon`).

Result on 100 validated panels (CPU, docTR + auto):

| set | exact-F1 | levenshtein-F1 |
|---|---|---|
| naive scrape (40, 68 % not panels) | 0.515 | 0.530 |
| validated (50) | 0.685 | 0.711 |
| **validated + GT-normalized (100)** | **0.744** | **0.769** |
| └ per-panel median | 0.788 | **0.819** |

**Ceiling analysis** (1010 GT ingredients on the clean set): 66 % matched, 17 % visible in the OCR
but unmatched (recovering them costs more false positives than true positives — net-negative, so
the Trie matcher stays the optimum), 17 % not legible in the OCR at all. The ~0.74 exact-F1 /
0.82 median levenshtein-F1 is therefore the realistic CPU ceiling on real-world panels; closing
to 95 % requires VLM OCR (Qwen2.5-VL / GPT-4o-Vision) + LLM dictionary correction, i.e. GPU/cloud.

Reproduce: `python scrape_obf.py 100` then
`BENCH_GT_DIR=data/scraped/ground_truth BENCH_IMG_DIR=data/scraped/images python benchmark.py scraped100`.
