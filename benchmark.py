# type: ignore
"""Reproducible benchmark for the ToxFox OCR pipeline.

Runs the full pipeline over every ground-truth image and reports BOTH the
exact and levenshtein F1 / accuracy (true-positive rate), computed with the
project's own Evaluation logic so the numbers are directly comparable to what
the maintainers report. Also records peak RSS and wall time so we can prove
improvements stay within the 8 GB CPU budget.

Usage:
    python benchmark.py [label]

Writes benchmark_results/<label>.json with aggregate metrics and per-image
predictions (for diffing which products improved / regressed between runs).
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import yaml

from zug_toxfox import default_config, getLogger
from zug_toxfox.modules.evaluation import Evaluation
from zug_toxfox.modules.postprocessing import FAISSIndexer
from zug_toxfox.pipeline import Pipeline

log = getLogger("benchmark")


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB (Linux ru_maxrss is kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def normalize_gt(items):
    return sorted([gt.lower().strip().replace("*", "") for gt in items])


def normalize_pred(items):
    return sorted([p.lower().strip() for p in items])


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    gt_dir = Path(default_config.ground_truth_path)
    img_dir = Path(default_config.image_path)

    # Pair every ground-truth file with its image.
    cases = []
    for gt_file in sorted(gt_dir.glob("*.yaml")):
        stem = gt_file.stem
        img = next((img_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (img_dir / f"{stem}{ext}").exists()), None)
        if img is None:
            log.warning("No image for %s, skipping", stem)
            continue
        with open(gt_file) as f:
            gt = yaml.safe_load(f).get("INCI_list") or []
        cases.append((stem, img, normalize_gt(gt)))

    log.info("Benchmark '%s': %d cases", label, len(cases))

    ev = Evaluation()  # reused only for its get_metrics(); thresholds from config
    indexer = FAISSIndexer()
    t0 = time.time()
    pipeline = Pipeline(indexer=indexer, evaluation=False)
    build_s = time.time() - t0
    log.info("Pipeline built in %.1fs (peak RSS so far %.2f GB)", build_s, peak_rss_gb())

    per_image = []
    sums = {"exact_f1": 0.0, "exact_acc": 0.0, "lev_f1": 0.0, "lev_acc": 0.0}
    t_infer = 0.0
    for i, (stem, img_path, gt) in enumerate(cases):
        image = cv2.imread(str(img_path))
        ts = time.time()
        try:
            result = pipeline.process_image(image)
            preds = normalize_pred(result.get("ingredients", []))
        except Exception as e:  # noqa: BLE001
            log.exception("FAILED on %s: %s", stem, e)
            preds = []
        dt = time.time() - ts
        t_infer += dt

        ex_f1, ex_acc = ev.get_metrics(preds, gt, "exact")
        lv_f1, lv_acc = ev.get_metrics(preds, gt, "levenshtein")
        sums["exact_f1"] += ex_f1
        sums["exact_acc"] += ex_acc
        sums["lev_f1"] += lv_f1
        sums["lev_acc"] += lv_acc
        per_image.append({
            "id": stem, "n_gt": len(gt), "n_pred": len(preds),
            "exact_f1": round(ex_f1, 3), "exact_acc": round(ex_acc, 3),
            "lev_f1": round(lv_f1, 3), "lev_acc": round(lv_acc, 3),
            "secs": round(dt, 2), "pred": preds,
        })
        log.info("[%2d/%d] %-16s gt=%2d pred=%2d  exF1=%.2f lvF1=%.2f  %.1fs",
                 i + 1, len(cases), stem, len(gt), len(preds), ex_f1, lv_f1, dt)

    n = len(cases)
    agg = {k: round(v / n, 4) for k, v in sums.items()}
    summary = {
        "label": label,
        "n_cases": n,
        "metrics": agg,
        "build_secs": round(build_s, 1),
        "infer_secs_total": round(t_infer, 1),
        "infer_secs_mean": round(t_infer / n, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "config": {
            "ocr_engine": pipeline.ocr.engine,
            "match_backend": pipeline.postprocessor.match_backend,
            "typo_backend": pipeline.postprocessor.token_cleaner.typo_backend,
            "embed_model": pipeline.postprocessor.indexer.model_name,
        },
    }

    out_dir = Path(__file__).resolve().parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    log.info("=" * 64)
    log.info("RESULT '%s' (n=%d)", label, n)
    log.info("  exact:       F1=%.4f  acc=%.4f", agg["exact_f1"], agg["exact_acc"])
    log.info("  levenshtein: F1=%.4f  acc=%.4f", agg["lev_f1"], agg["lev_acc"])
    log.info("  peak RSS=%.2f GB  build=%.1fs  infer=%.1fs (%.2fs/img)",
             summary["peak_rss_gb"], build_s, t_infer, summary["infer_secs_mean"])
    log.info("  -> %s", out_path)
    log.info("=" * 64)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
