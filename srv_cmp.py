# type: ignore
"""Fast server-vs-mobile RapidOCR recognition compare on hard panels (no orientation, to isolate
recognition quality). Does the heavier PP-OCRv4 server rec model garble dense INCI text LESS?"""
import os, time
os.environ["AUTO_ORIENT"] = "0"
import cv2
from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.preprocessing import PreProcessor

pre = PreProcessor()
IMGS = ["data/images/4260602121510.jpg", "data/images/3600542169950.jpg",
        "data/images/4005800162350.jpg", "data/images/3367729117264.jpg"]
for mtype in ["mobile", "server"]:
    os.environ["RAPIDOCR_MODEL_TYPE"] = mtype
    ocr = OCR(engine="rapidocr")
    print(f"\n########## RapidOCR {mtype} ##########", flush=True)
    for p in IMGS:
        im = pre.preprocess_image(cv2.imread(p))
        t = time.time()
        try:
            toks = ocr.process_image(im, debug=False)
        except Exception as e:  # noqa: BLE001
            toks = [f"ERR {e}"]
        print(f"--- {os.path.basename(p)} ({time.time()-t:.1f}s) ---", flush=True)
        print("  ", (" ".join(toks)[:260]) or "NOTHING", flush=True)
print("\nSRV_CMP_DONE", flush=True)
