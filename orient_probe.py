# type: ignore
"""Validate the rotation hypothesis: OCR the zero-prediction curated images at 0/90/180/270
and report recognized-text yield per angle. High yield at a non-zero angle == rotated panel.
"""
import cv2
import numpy as np

from zug_toxfox.modules.ocr import OCR

ZEROS = ["0842101102074", "4015100718799", "4025089085034", "4056489764137",
         "4058172927928", "4305615418667", "9120082221559"]
ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def yield_score(triples):
    # total chars of recognized text weighted by confidence (engine-agnostic legibility proxy)
    s = 0.0
    for t in triples:
        txt = (t[1] or "").strip()
        conf = t[2] if len(t) > 2 and t[2] is not None else 1.0
        s += len(txt) * float(conf)
    return s, len(triples)


def main():
    ocr = OCR()
    for stem in ZEROS:
        img = cv2.imread(f"data/images/{stem}.jpg")
        if img is None:
            print(stem, "READ-FAIL"); continue
        # bound size so a rotated tall strip isn't crushed: cap longest side ~2000
        h, w = img.shape[:2]
        print(f"\n{stem}  {w}x{h} (aspect {max(w,h)/min(w,h):.1f})")
        best = (0, -1)
        for ang, code in ROT.items():
            im = img if code is None else cv2.rotate(img, code)
            try:
                triples = ocr._detect(im)
            except Exception as e:  # noqa: BLE001
                print(f"  {ang:3}: ERR {e}"); continue
            sc, n = yield_score(triples)
            sample = " | ".join((t[1] or "")[:40] for t in triples[:2])
            print(f"  {ang:3}: yield={sc:7.0f} boxes={n:3}  {sample[:80]}")
            if sc > best[1]:
                best = (ang, sc)
        print(f"  -> best angle {best[0]} (yield {best[1]:.0f})")


if __name__ == "__main__":
    main()
