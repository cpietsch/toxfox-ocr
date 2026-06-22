# type: ignore
"""Scrape a CLEAN real-world INCI test set from Open Beauty Facts (open ODbL data).

OBF's 'ingredients' image is frequently mislabelled (a product front, not the ingredient panel).
So every candidate is OCR-validated: we keep it only when the ground-truth ingredients are actually
visible in the image (a fair "the photo depicts the list" check, lenient to OCR noise). Ground truth
is taken from ingredients_text_<lang>, cleaned and de-duplicated. Prefers EN/DE INCI panels.

Usage: python scrape_obf.py [target_valid_count]
"""
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import yaml
from rapidfuzz import fuzz, process

from zug_toxfox.modules.ocr import OCR
from zug_toxfox.modules.preprocessing import PreProcessor

UA = "toxfox-ocr-research/1.0 (crp@mailbox.org)"
OUT = Path("data/scraped")
LANG_PREF = ["en", "de"]
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MIN_COVERAGE = 0.50  # >= this fraction of GT ingredients must be visible in the OCR to keep the image


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40).read()


def path_from_front(selected_images: dict) -> str | None:
    front = (selected_images or {}).get("front", {}).get("display", {})
    for url in front.values():
        m = re.search(r"/products/(.+?)/[^/]+$", url)
        if m:
            return m.group(1)
    return None


def clean_gt(text: str) -> list[str]:
    text = re.split(r"(?i)\b(?:may contain|peut contenir|kann enthalten|\+/-|\+\\-)\b", text)[0]
    out = []
    for tok in text.split(","):
        t = re.sub(r"\([^)]*\)", "", tok)  # drop parenthetical sub-notes
        t = re.sub(r"\[[^\]]*\]", "", t)  # drop [nano] etc.
        t = re.sub(r"[*]+", "", t)
        t = re.sub(r"\s+", " ", t).strip().strip(".").strip()
        if len(t) >= 3 and not re.fullmatch(r"[\d.\s%/]+", t):
            out.append(t)
    # de-duplicate, preserve order
    return list(dict.fromkeys(out))


def coverage(gt: list[str], raw_lines: list[str]) -> float:
    if not raw_lines or not gt:
        return 0.0
    hit = sum(
        1 for name in gt
        if len(name) >= 4 and (process.extractOne(name, raw_lines, scorer=fuzz.partial_ratio) or (None, 0))[1] >= 85
    )
    return hit / len(gt)


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth").mkdir(parents=True, exist_ok=True)
    pre, ocr = PreProcessor(), OCR()
    fields = "code,lang,product_name,images,selected_images,ingredients_text_en,ingredients_text_de"
    # Additive: keep already-validated panels, seed seen so we only fetch/OCR NEW candidates.
    existing = {f.stem for f in (OUT / "ground_truth").glob("*.yaml")}
    kept, scanned, seen = len(existing), 0, set(existing)
    print(f"resuming with {kept} existing panels; target {TARGET}")
    for page in range(1, 60):
        if kept >= TARGET:
            break
        try:
            data = json.loads(get(
                f"https://world.openbeautyfacts.org/api/v2/search?fields={fields}"
                f"&page_size=100&page={page}&sort_by=popularity_key"))
        except Exception as e:  # noqa: BLE001
            print("page", page, "fail", e)
            time.sleep(2)
            continue
        prods = data.get("products", [])
        if not prods:
            break
        for p in prods:
            if kept >= TARGET:
                break
            code = str(p.get("code") or "")
            if not code or code in seen:
                continue
            imgs = p.get("images") or {}
            for lang in LANG_PREF:
                key = f"ingredients_{lang}"
                txt = p.get(f"ingredients_text_{lang}") or ""
                if key not in imgs or len(txt) < 30 or "," not in txt:
                    continue
                gt = clean_gt(txt)
                path = path_from_front(p.get("selected_images"))
                if len(gt) < 5 or not path:
                    continue
                seen.add(code)
                scanned += 1
                rev = imgs[key].get("rev")
                img_url = f"https://images.openbeautyfacts.org/images/products/{path}/{key}.{rev}.full.jpg"
                try:
                    blob = get(img_url)
                    arr = cv2.imdecode(__import__("numpy").frombuffer(blob, dtype="uint8"), cv2.IMREAD_COLOR)
                    raw = ocr.process_image(pre.preprocess_image(arr), debug=False)
                except Exception:  # noqa: BLE001
                    break
                cov = coverage([g.lower() for g in gt], raw)
                if cov < MIN_COVERAGE:
                    print(f"  skip {code} [{lang}] cov={cov:.0%} (image not a readable panel)")
                    break
                (OUT / "images" / f"{code}.jpg").write_bytes(blob)
                meta = {"INCI_list": gt, "id": code, "image_name": f"{code}.jpg", "lang": lang,
                        "product_name": p.get("product_name") or "", "raw_ingredients_list": txt,
                        "ocr_coverage": round(cov, 2),
                        "source": f"https://world.openbeautyfacts.org/product/{code}"}
                with open(OUT / "ground_truth" / f"{code}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(meta, f, allow_unicode=True, sort_keys=False)
                kept += 1
                print(f"  [{kept:2}/{TARGET}] {code} [{lang}] cov={cov:.0%} {len(gt):2}ing  {(p.get('product_name') or '')[:30]}")
                break
        time.sleep(1)
    print(f"DONE: kept {kept} valid panels (scanned {scanned} candidates) -> {OUT}")


if __name__ == "__main__":
    main()
