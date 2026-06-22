# type: ignore
"""Postprocessing sweep: run OCR ONCE per image, then evaluate many postprocessing
configs on the cached OCR tokens.

OCR (~3 s/img) dominates cost; the matchers (faiss/rapidfuzz/symspell) are ~ms, so caching
the raw OCR output lets us A/B every postprocessing variant in a single OCR pass instead of
one full benchmark run each. Engine chosen via OCR_ENGINE env (default doctr here).

Writes benchmark_results/<label>.json for each config so compare_results.py picks them up.
"""
import json
import os
import resource
import time
from pathlib import Path

import cv2
import yaml

from zug_toxfox import default_config, getLogger
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.postprocessing import FAISSIndexer, PostProcessor
from zug_toxfox.modules.preprocessing import PreProcessor

log = getLogger("sweep")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


from benchmark import canon  # symmetric multilingual-synonym canonicalization (Aqua/Water -> aqua)


def norm(items):
    return sorted({canon(x) for x in items if canon(x)})


# (label, match_strategy, segment_threshold)  -- matcher/typo fixed at the chosen defaults
# (label, match_strategy, segment_threshold, segment_min_dropped)
CONFIGS = [
    ("trie", "trie", 80, 0),
    ("seg82", "segment", 82, 0),
    ("seg86", "segment", 86, 0),
    ("seg90", "segment", 90, 0),
    ("seg93", "segment", 93, 0),
]


def main():
    engine = os.environ.get("OCR_ENGINE", "doctr")
    engine_label = os.environ.get("SWEEP_LABEL", engine)
    gt_dir = Path(os.environ.get("BENCH_GT_DIR", default_config.ground_truth_path))
    img_dir = Path(os.environ.get("BENCH_IMG_DIR", default_config.image_path))
    cases = []
    for gt_file in sorted(gt_dir.glob("*.yaml")):
        stem = gt_file.stem
        img = next((img_dir / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png") if (img_dir / f"{stem}{e}").exists()), None)
        if img is None:
            continue
        gt = yaml.safe_load(open(gt_file)).get("INCI_list") or []
        cases.append((stem, img, norm(gt)))
    log.info("sweep engine=%s cases=%d", engine, len(cases))

    pre = PreProcessor()
    ocr = OCR()
    indexer = FAISSIndexer()
    post = PostProcessor(indexer)
    # Force-build symspell so toggling typo_backend at runtime works.
    post.token_cleaner._symspell = post.token_cleaner._build_symspell()
    ev = Evaluation()

    # ---- single OCR pass, cache raw tokens ----
    raw_cache = {}
    t0 = time.time()
    for i, (stem, img_path, gt) in enumerate(cases):
        image = cv2.imread(str(img_path))
        try:
            raw_cache[stem] = ocr.process_image(pre.preprocess_image(image), debug=False)
        except Exception as e:  # noqa: BLE001
            log.exception("OCR FAILED %s: %s", stem, e)
            raw_cache[stem] = []
        if (i + 1) % 10 == 0:
            log.info("OCR %d/%d (%.1fs, peak %.2f GB)", i + 1, len(cases), time.time() - t0, peak_rss_gb())
    log.info("OCR pass done in %.1fs, peak %.2f GB", time.time() - t0, peak_rss_gb())

    # ---- evaluate each postprocessing config on the cached tokens ----
    for label, strategy, seg_thr, min_drop in CONFIGS:
        post.match_strategy = strategy
        post.segment_threshold = seg_thr
        post.segment_min_dropped = min_drop
        sums = {"exact_f1": 0.0, "exact_acc": 0.0, "lev_f1": 0.0, "lev_acc": 0.0}
        per_image = []
        for stem, img_path, gt in cases:
            try:
                res = post.get_ingredients(raw_cache[stem])
                preds = norm(res.get("ingredients", []))
            except Exception as e:  # noqa: BLE001
                log.exception("POST FAILED %s [%s]: %s", stem, label, e)
                preds = []
            ex_f1, ex_acc = ev.get_metrics(preds, gt, "exact")
            lv_f1, lv_acc = ev.get_metrics(preds, gt, "levenshtein")
            sums["exact_f1"] += ex_f1
            sums["exact_acc"] += ex_acc
            sums["lev_f1"] += lv_f1
            sums["lev_acc"] += lv_acc
            per_image.append({"id": stem, "n_gt": len(gt), "n_pred": len(preds),
                              "exact_f1": round(ex_f1, 3), "lev_f1": round(lv_f1, 3), "pred": preds})
        n = len(cases)
        metrics = {k: round(v / n, 4) for k, v in sums.items()}
        full_label = f"{engine_label}_{label}"
        summary = {"label": full_label, "n_cases": n, "metrics": metrics,
                   "peak_rss_gb": round(peak_rss_gb(), 2),
                   "config": {"ocr_engine": engine, "match_strategy": strategy,
                              "segment_threshold": seg_thr, "isolate_region": post.isolate_region}}
        out = Path(__file__).resolve().parent / "benchmark_results" / f"{full_label}.json"
        json.dump({"summary": summary, "per_image": per_image}, open(out, "w"), indent=2)
        log.info("[%s] exF1=%.4f exAcc=%.4f lvF1=%.4f  -> %s",
                 full_label, metrics["exact_f1"], metrics["exact_acc"], metrics["lev_f1"], out.name)


if __name__ == "__main__":
    main()
