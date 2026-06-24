# type: ignore
"""Cache raw OCR text lines per (engine, set) so matching experiments run in seconds.

Engine is selected by the OCR_ENGINE / DOCTR_DET / DOCTR_RECO env vars (read by OCR()).
Cache key is passed as argv[1] (e.g. 'doctr', 'doctr_fast', 'rapidocr', 'ensemble').
Writes /tmp/cache_<key>_<set>.json = {stem: [ocr_line, ...]}.

Usage:
    OCR_ENGINE=doctr python build_cache.py doctr
    OCR_ENGINE=doctr DOCTR_DET=fast_base DOCTR_RECO=master python build_cache.py doctr_fastmaster
    SETS=curated,scraped python build_cache.py doctr     # subset of sets
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2

from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.preprocessing import PreProcessor

ALL_SETS = {
    "scraped": ("data/scraped/ground_truth", "data/scraped/images"),
    "curated": ("data/ground_truth", "data/images"),
    "realworld": ("data/realworld/ground_truth", "data/realworld/images"),
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "run"
    want = os.environ.get("SETS")
    sets = {k: ALL_SETS[k] for k in want.split(",")} if want else ALL_SETS
    pre, ocr = PreProcessor(), OCR()
    for name, (gtd, imgd) in sets.items():
        gtd, imgd = Path(gtd), Path(imgd)
        cache, t0 = {}, time.time()
        for f in sorted(gtd.glob("*.yaml")):
            img = next((imgd / f"{f.stem}{e}" for e in (".jpg", ".jpeg", ".png") if (imgd / f"{f.stem}{e}").exists()), None)
            if img is None:
                continue
            try:
                cache[f.stem] = ocr.process_image(pre.preprocess_image(cv2.imread(str(img))), debug=False)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {f.stem}: {e}")
                cache[f.stem] = []
        out = f"/tmp/cache_{key}_{name}.json"
        json.dump(cache, open(out, "w"))
        print(f"cached {name}: {len(cache)} imgs in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
