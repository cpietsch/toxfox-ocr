# type: ignore
"""Feasibility probe: can a small, CPU-runnable OCR VLM read the panels CPU OCR (docTR+RapidOCR)
MISSES? Runs candidate models on the hardest curated images (those with the most ocr_missing GT)
and prints extracted text + wall time, so we can judge recall gain vs CPU speed before integrating.

    OMP_NUM_THREADS=6 python vlm_probe.py got        # GOT-OCR2.0 (0.58B, doc-OCR, native HF)
    OMP_NUM_THREADS=6 python vlm_probe.py florence    # Florence-2-base (0.23B, remote code)
"""
import os, sys, time
import torch
from PIL import Image

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))


def load_img(p):
    """Upscale small/strip crops so the VLM has enough pixels-per-glyph (min side >= MINSIDE),
    and cap the long side to bound CPU cost."""
    img = Image.open(p).convert("RGB")
    minside = int(os.environ.get("MINSIDE", "1024"))
    maxside = int(os.environ.get("MAXSIDE", "2048"))
    w, h = img.size
    s = 1.0
    if min(w, h) < minside:
        s = minside / min(w, h)
    if max(w * s, h * s) > maxside:
        s = maxside / max(w, h)
    if abs(s - 1.0) > 0.01:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return img

# Hard curated panels (lots of ocr_missing GT in diag5) + one easy one as a sanity check.
IMGS = [
    "data/images/4260602121510.jpg",
    "data/images/3600542169950.jpg",
    "data/images/4005800162350.jpg",
]


def run_got():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = "stepfun-ai/GOT-OCR-2.0-hf"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    print(f"[got] loaded in {time.time()-t0:.0f}s")
    for p in IMGS:
        if not os.path.exists(p):
            print(f"  !! missing {p}"); continue
        img = load_img(p)
        t0 = time.time()
        inputs = proc(img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024,
                                 tokenizer=proc.tokenizer, stop_strings="<|im_end|>")
        txt = proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n=== {os.path.basename(p)}  ({time.time()-t0:.0f}s, {img.size}) ===\n{txt}")


def run_florence():
    from transformers import AutoProcessor, AutoModelForCausalLM
    mid = "microsoft/Florence-2-base"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.float32).eval()
    print(f"[florence] loaded in {time.time()-t0:.0f}s")
    for p in IMGS:
        if not os.path.exists(p):
            print(f"  !! missing {p}"); continue
        img = load_img(p)
        t0 = time.time()
        inputs = proc(text="<OCR>", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 max_new_tokens=1024, num_beams=1, do_sample=False)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"\n=== {os.path.basename(p)}  ({time.time()-t0:.0f}s, {img.size}) ===\n{txt}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "got"
    {"got": run_got, "florence": run_florence}[which]()
    print("\nVLM_PROBE_DONE")
