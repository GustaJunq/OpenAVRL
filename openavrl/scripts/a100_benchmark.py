import time, torch
from openavrl import OpenAVRL2Pipeline

pipe = OpenAVRL2Pipeline.from_pretrained(device="cuda")
torch.cuda.reset_peak_memory_stats()

start = time.time()
img = pipe.generate("Benchmark poster", n_max=3)
elapsed = time.time() - start
print(f"Time: {elapsed:.1f}s, Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
img.save("/tmp/bench.png")
