# type: ignore
"""Audit every GT token that the symmetric INCI snapper changes, on both eval sets.
Prints original -> snapped so each can be eyeballed for identity-preservation (fairness check).

    OMP_NUM_THREADS=4 python snap_audit.py
"""
import os
from collections import Counter
from pathlib import Path
import yaml
import score as S

def main():
    S._SNAP = True
    changed = Counter()
    for name, gtdir in S.SETS.items():
        rows = []
        for f in Path(gtdir).glob("*.yaml"):
            d = yaml.safe_load(open(f))
            if name == "scraped" and d.get("raw_ingredients_list"):
                raw = S.clean_gt_v2(d["raw_ingredients_list"])
            else:
                raw = d.get("INCI_list") or []
            for x in raw:
                base = str(x).lower().strip().replace("*", "")
                import re
                base = re.sub(r"\([^)]*\)", "", base)
                base = re.sub(r"\s+", " ", base).strip().strip(".").strip()
                snapped = S.canon(x)
                if snapped != base and snapped not in ("aqua", "parfum"):
                    rows.append((base, snapped))
                    changed[name] += 1
        # dedupe display
        seen = {}
        for b, s in rows:
            seen[(b, s)] = seen.get((b, s), 0) + 1
        print(f"\n===== {name}: {changed[name]} GT-token snaps ({len(seen)} distinct) =====")
        for (b, s), c in sorted(seen.items()):
            tag = "" if c == 1 else f" x{c}"
            print(f"   '{b}'  ->  '{s}'{tag}")

if __name__ == "__main__":
    main()
