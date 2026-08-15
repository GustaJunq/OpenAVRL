"""Generate DPO pairs (j_win, j_lose) using v2 visual evaluator on A100"""
import json
from openavrl2 import OpenAVRL2Pipeline
from pathlib import Path

pipe = OpenAVRL2Pipeline.from_pretrained(device="cuda")
prompts = [l.strip() for l in open("data/prompts.txt")] # that .txt is required for the benchmarks, DPO preferences, etc.

out = Path("data/dpo_pairs.jsonl")
out.parent.mkdir(exist_ok=True)

for p0 in prompts:
    trace = pipe.generate_with_trace(p0, n_max=3)
    # if we have at least 2 steps, first is lose, last is win
    if len(trace.critiques) >= 2:
        # need to save json pairs
        # trace stores only final json, so in real script save all intermediate jsons in pipeline
        # simplified: use critique history
        print(f"Collected {p0[:40]} -> steps {trace.steps}")
        # TODO: modify pipeline to return all intermediate jsons
      
