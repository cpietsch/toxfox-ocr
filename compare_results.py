# type: ignore
"""Tabulate all benchmark_results/*.json runs against the baseline.

Usage: python compare_results.py
"""
import json
from pathlib import Path

BASE = "baseline_easyocr_minilm"


def main():
    d = Path(__file__).resolve().parent / "benchmark_results"
    runs = {}
    for f in sorted(d.glob("*.json")):
        try:
            runs[f.stem] = json.load(open(f))["summary"]
        except Exception:  # noqa: BLE001
            continue
    if not runs:
        print("no results yet")
        return
    base = runs.get(BASE, {}).get("metrics", {})
    hdr = f"{'run':32} {'exF1':>6} {'exAcc':>6} {'lvF1':>6} {'lvAcc':>6} {'dExF1':>7} {'RSS':>5} {'s/img':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in sorted(runs.items(), key=lambda kv: kv[1].get("metrics", {}).get("exact_f1", 0)):
        m = s.get("metrics", {})
        d_ex = m.get("exact_f1", 0) - base.get("exact_f1", 0) if base else 0
        flag = " *BASE*" if name == BASE else ""
        print(f"{name:32} {m.get('exact_f1',0):6.3f} {m.get('exact_acc',0):6.3f} "
              f"{m.get('lev_f1',0):6.3f} {m.get('lev_acc',0):6.3f} {d_ex:+7.3f} "
              f"{s.get('peak_rss_gb',0):5.2f} {s.get('infer_secs_mean',0):6.2f}{flag}")


if __name__ == "__main__":
    main()
