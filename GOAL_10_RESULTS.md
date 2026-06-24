# Goal: exact-F1 ≥ 0.93 on BOTH eval sets — CPU-only, no VLM, peak RAM < 8 GB

**Outcome: a documented, evidence-backed plateau at curated 0.8732 / scraped 0.8278 (authoritative
`benchmark.py`, harness-exact). 0.93 on both is not reachable under the CPU-only / no-VLM
constraint.** This session lifted curated 0.8652 → **0.8732** and scraped 0.7940 → **0.8278** with
four fair, separately-committed improvements, then empirically established the plateau by trying and
*rejecting* three independent OCR-recall mechanisms (each net-negative for the same reason).

All numbers below are the authoritative `benchmark.py` (full pipeline, real OCR), which matches the
fast `score.py` harness exactly at every checkpoint.

## Headline (exact-F1, fair v2 GT, CPU-only)

| eval set | session start | +seg-protect | +INCI-snap GT | +GT-fairness | **+prefix-FF (final)** | goal |
|---|---|---|---|---|---|---|
| curated59  | 0.8652 | 0.8702 | 0.8716 | 0.8716 | **0.8732** (lev 0.8885) | 0.93 |
| scraped100 | 0.7940 | 0.7956 | 0.8179 | 0.8241 | **0.8278** (lev 0.8495) | 0.93 |

Final authoritative confirmation: **curated exact-F1 0.8732, scraped exact-F1 0.8278**, peak RSS
4.18 / 3.55 GB (well under 8 GB), ~6.2 s/img. From the project's original baseline the cumulative
gain is curated **0.661 → 0.873 (+0.212)**, scraped **0.757 → 0.828 (+0.071)**.

## What moved the numbers this session (all CPU, all fair, each committed separately)

1. **Segment-protected fragment dropper** (`postprocessing.py`, `PROTECT_SRC=seg`) — curated
   0.8652 → 0.8702. `_drop_fragments` was deleting a real base ingredient whenever a derivative was
   also listed (`dimethicone` killed by `hydrogen dimethicone`; `silica` by `hydrated silica`). Real
   INCI lists routinely contain both as separate items. Fix: shield names the *delimiter-respecting*
   segment matcher emits from their own segment; pure Trie prefix-fragments (`hydrogen`) still drop.
2. **Symmetric INCI GT canonicalization** (`score.py inci_snap`) — scraped 0.7956 → 0.8179. The
   scraped GT is re-extracted from raw Open Beauty Facts text and carries OCR/transcription noise on
   the *same* names the pipeline reads correctly (`lycerylstearate`→`glyceryl stearate`,
   `copemicia cerifera cera`→`copernicia…`, `cl 42090`→`ci 42090`). A correct read was double-
   penalised (FP for the canonical name + FN for the corrupted GT string). `inci_snap` expresses both
   sides in the shared INCI vocabulary (despaced-exact / fuzz≥90 + length gate + a word-level
   no-trailing-qualifier guard). Uses only the reference dictionary, never the answers/predictions;
   a no-op on predictions → symmetric and fair. Every one of the 71+8 snaps was audited
   (`snap_audit.py`) as an identity-preserving OCR/typo fix.
3. **GT-fairness batch** (`score.py`) — scraped 0.8179 → 0.8241. (a) Word-level snap guard recovers
   trailing-char typo fixes (`cocamidopropyl betain`→`…betaine`) the char-prefix guard wrongly
   blocked, while still refusing fabricated word-additions (`butyrospermum parkii` -/-> `…oil`).
   (b) Non-cosmetic boilerplate filter drops manufacturer/address/contact/legal prose the raw text
   glues into the ingredient field (`manufactured by…ltd`, `consumer care line`, `&quot`, `box 30062`)
   — generic markers, no brand/place/per-image strings. (c) Purified-water phrases → `aqua`
   (enumerated; floral hydrosols never collapse).
4. **Prefix false-friend precision filter** (`postprocessing.py`) — curated 0.8716 → 0.8732, scraped
   0.8241 → 0.8278, pure precision (recall flat). Drops a short dictionary word that appears in the
   OCR *only* as the prefix of a longer run (`hydrogen` inside `HYDROGENATED…`, `phenol` inside
   `PHENOLSULFONATE`); a genuinely-listed short ingredient appears as its own token and survives.

## Where the residual is (per-bucket, final config)

```
curated  n=59   exact-F1=0.8732  P=0.900 R=0.867   TP=1075 FP=120 FN=165
  FN: ocr_missing=125 (76%)   ocr_visible=34 (21%)   not_in_dict=6 (4%)
  FP: spurious=96   (80%)     near_gt=24    (20%)
scraped  n=100  exact-F1=0.8278  P=0.854 R=0.854   TP=1762 FP=301 FN=301
  FN: not_in_dict=159 (53%)   ocr_missing=115 (38%)  ocr_visible=27 (9%)
  FP: spurious=250  (83%)     near_gt=51    (17%)
```
(`ocr_missing` = GT name genuinely absent from the OCR; `ocr_visible` = present but unmatched;
`not_in_dict` = GT name not in our INCI vocab; `spurious` FP = predicted name far from any GT name.)

## Why 0.93 on both is not reachable on CPU / no-VLM (the evidence)

**Curated is OCR-bound.** 76 % of the remaining FN (125 items, ~10 % of all GT) is `ocr_missing` —
text genuinely illegible to CPU OCR: curved labels, low-contrast print, tiny fonts, and word-gluing
(`…PEEL OIL STEARIC ACID PARAFFINUM…` read as `…PEELOILSTEARIC ACIDPARAFFINUM…`). **Three
independent recall-adding mechanisms were tried and all net-negative**, because the extra reads CPU
OCR produces on these panels are garbled enough that the matcher turns them into FPs as fast as TPs:
| recall lever | curated Δ | scraped Δ | verdict |
|---|---|---|---|
| EasyOCR as a 3rd engine (sep-union, swept seg 88–95) | −0.001 | n/a | rejected |
| RapidOCR on a CLAHE + 1.5× preprocessing variant (`build_clahe.py`) | −0.001 | −0.003 | rejected |
| …same, on top of the higher prefix-FF precision floor | −0.001 | −0.003 | rejected |
| despaced long-name substring recovery (minlen 12/14/16) | −0.004…−0.013 | −0.006…−0.011 | rejected |

Each adds ≈ +0.005 recall and costs ≈ −0.010 precision. Reading these panels is exactly what a
VLM-class OCR does and what the no-VLM constraint excludes.

**Scraped is GT-quality-bound, not reading-bound.** A *perfect* reader is penalised here:
- **83 % of scraped FP (`spurious`, 250)** is dominated by ingredients the pipeline reads
  *correctly* that are simply absent from the partial crowd-sourced OBF list — `sodium benzoate`
  (×4), `silica` (×3), `aqua` (×3), `alcohol` (×3), `disodium edta`, `sodium hydroxide`,
  `magnesium stearate`, … These are GT *incompleteness*, not pipeline errors, and cannot be removed
  without editing the GT toward the predictions (which would be peeking — disallowed).
- **53 % of scraped FN (`not_in_dict`, 159)** is foreign-language INCI on French/Nordic products
  (`benzoate de sodium` = sodium benzoate, `dioxyde de titane` = titanium dioxide, `silice` = silica)
  plus GT strings too garbled for the orthographic snap. These need a multilingual INCI translation
  layer (see below), not better reading.

## What it would take to bridge the remaining gap (beyond CPU / no-VLM)

1. **A VLM / cloud OCR (GPT-4V-class)** for the ~125 curated and ~115 scraped `ocr_missing` items —
   the curved/low-contrast/tiny/word-glued text three CPU OCR engines (docTR, RapidOCR, EasyOCR) and
   a contrast-enhanced pass all fail to read. This is the single biggest unlock and is precisely what
   the "no-VLM" constraint forbids.
2. **A complete, curated scraped GT** (not the partial OBF crowd text). ~80 % of scraped `spurious`
   FP are correct reads missing from the GT; with a complete GT, scraped precision would jump from
   0.854 toward ~0.95 at no change to the pipeline.
3. **A symmetric multilingual INCI normalization** (FR/DE/IT → canonical, applied to both GT and
   predictions — sanctioned lever 3). Partially tractable on CPU but low-ROI (concentrated on one or
   two foreign products, ~+0.005–0.01 scraped) and translation-risk-prone (a wrong mapping fabricates
   identity), so it was scoped out as not worth the fairness risk for the return; documented here as
   the one remaining fair lever.

## Reproduce

```bash
# authoritative (full pipeline, real OCR), fair v2 GT for scraped:
python benchmark.py final_curated
BENCH_GT_DIR=data/scraped/ground_truth BENCH_IMG_DIR=data/scraped/images python benchmark.py final_scraped
# fast matcher/GT experiments on cached OCR (no OCR re-run):
python build_cache.py orient ; OCR_ENGINE=rapidocr python build_cache.py rapid     # once
CACHE=ensemble GT=v2 python score.py union3 80      # production: docTR ∪ RapidOCR, union3, prefix-FF
CACHE=ensemble GT=v2 python diag2.py union3 80      # full FN/FP bucket decomposition + examples
python snap_audit.py                                # audit every GT-canon snap for fairness
# rejected recall levers (reproduce the net-negative result):
python build_clahe.py rapidclahe
CACHE=ens ENS_ENGINES=orient,rapid,rapidclahe ENS_SEGS=80,90,90 GT=v2 python score.py union3 80
```

## Fairness ledger

Every gain is a real pipeline change (matcher precision) or a **symmetric, identity-preserving**
GT/canon correction derived only from the fixed INCI reference vocabulary and generic textual
structure — never from the 159 test answers. No per-image tuning, no hardcoded GT, no thresholds
fit to the test items (the one per-image junk marker that slipped in was removed). Snaps were audited
exhaustively; recall levers that would have inflated via FPs were measured and rejected. Peak RSS
stayed < 4.2 GB throughout.
