# type: ignore
"""Cache RapidOCR reads on a CONTRAST-ENHANCED preprocessing variant, to recover low-contrast /
small text that the default read misses. Pipeline-fair: a symmetric preprocessing pass whose
matched ingredients are UNIONED with the default ensemble (same idea as a 2nd engine, but a 2nd
*view* of the image). CLAHE equalises local contrast; a 1.5x upscale helps the detector on small
fonts. Writes /tmp/cache_rapidclahe_<set>.json = {stem: [ocr_line, ...]}.

    OMP_NUM_THREADS=4 python build_clahe.py [rapidclahe]
"""
import json, os, sys, time
from pathlib import Path
import cv2
import numpy as np

from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.preprocessing import PreProcessor

ALL_SETS = {
    "curated": ("data/ground_truth", "data/images"),
    "scraped": ("data/scraped/ground_truth", "data/scraped/images"),
}


def enhance(bgr: np.ndarray, upscale: float) -> np.ndarray:
    """CLAHE on the luminance channel + mild upscale. Returns a BGR image (engine-agnostic)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if upscale and upscale != 1.0:
        out = cv2.resize(out, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return out


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "rapidclahe"
    upscale = float(os.environ.get("CLAHE_UPSCALE", "1.5"))
    want = os.environ.get("SETS")
    sets = {k: ALL_SETS[k] for k in want.split(",")} if want else ALL_SETS
    os.environ.setdefault("OCR_ENGINE", "rapidocr")
    pre, ocr = PreProcessor(), OCR(engine="rapidocr")
    for name, (gtd, imgd) in sets.items():
        gtd, imgd = Path(gtd), Path(imgd)
        cache, t0 = {}, time.time()
        for f in sorted(gtd.glob("*.yaml")):
            img = next((imgd / f"{f.stem}{e}" for e in (".jpg", ".jpeg", ".png") if (imgd / f"{f.stem}{e}").exists()), None)
            if img is None:
                continue
            try:
                bgr = pre.downscale(cv2.imread(str(img)))
                cache[f.stem] = ocr.process_image(enhance(bgr, upscale), debug=False)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {f.stem}: {e}", flush=True)
                cache[f.stem] = []
        out = f"/tmp/cache_{key}_{name}.json"
        json.dump(cache, open(out, "w"))
        print(f"cached {name}: {len(cache)} imgs in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
