# type: ignore
"""Probe: can a SMALL instruct LLM (CPU) extract a clean canonical INCI list from noisy OCR text,
beating the heuristic matcher's precision? Measures quality + wall time on a few panels.

    OMP_NUM_THREADS=6 python llm_probe.py Qwen/Qwen2.5-1.5B-Instruct
"""
import os, sys, time, json
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
# server OCR cache (cleaner reads) for a few representative stems
cache = json.load(open("/tmp/cache_rapidserver_curated.json"))
STEMS = ["3600542169950", "4005800162350", "3367729117264"]

PROMPT = (
    "You are an expert at reading cosmetic ingredient labels. Below is raw OCR text from an "
    "ingredient panel, with typos, split words, scrambled order, and non-ingredient text. "
    "Extract ONLY the cosmetic ingredients as canonical INCI names, correcting OCR errors. "
    "Output a comma-separated list, nothing else.\n\nOCR TEXT:\n{txt}\n\nINCI ingredients:"
)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
print(f"[load] {time.time()-t0:.0f}s  model={MID}", flush=True)

for s in STEMS:
    txt = " ".join(cache.get(s, []))[:1500]
    msgs = [{"role": "user", "content": PROMPT.format(txt=txt)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = tok(text, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=400, do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n=== {s}  ({time.time()-t0:.0f}s) ===\n{gen}", flush=True)
print("\nLLM_PROBE_DONE", flush=True)
