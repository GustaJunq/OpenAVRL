# OpenAVRL

**Agentic Visual Refinement Loop** — The world's second agentic image generation model (and the first open source one). Image generation guided by a closed agentic visual refinement cycle, combining Qwen3.5 9B with Ideogram 4.0 open weights (9.3B, released June 3, 2026).

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)

---

## What this is

OpenAVRL implements the **AVRL** (Agentic Visual Refinement Loop) architecture. Inspired by GPT Image 2's thinking mode, instead of generating an image in a single pass from a text prompt, the pipeline runs a **closed generate → critique → refine loop**, where the model itself visually evaluates the result and decides whether it needs to be redone.

The system is made up of three modules:

| Module | Role | Base model |
|---|---|---|
| **Upsampler** | Turns the prompt (`p0`) + critique history into a structured JSON spec for the generator | Qwen3.5 9B + `upsampler` LoRA |
| **Generator** | Generates the image from the JSON spec, using Ideogram 4.0's native JSON API. Swappable with Qwen-Image, Flux, SDXL | Ideogram 4.0 (open weights, 9.3B) |
| **Evaluator** | Sees the generated image (multimodal) plus the JSON used, and returns a structured critique with errors localized by bounding box | Qwen3.5 9B + `evaluator` LoRA |

The loop runs for up to `n_max` iterations or until the Evaluator approves the result (`approved: true`). Each error it identifies feeds into the next Upsampler iteration, refining the generation JSON.

### Adapters

You can use 2 separate LoRA adapters (one for Upsampler, one for Evaluator) for best quality, or a single shared adapter for both processes to save VRAM. The single-adapter mode loads Qwen3.5 9B once and reuses the same LoRA, reducing peak VRAM from ~38GB to ~22GB. Ideal for L40S 48GB, Lightning AI Studio, and local inference.

### What is `bbox`?

`bbox` is a bounding box that localizes where an error was found. Format: `[x_min, y_min, x_max, y_max]` normalized to 0-1.

- `x`: horizontal axis (0=left, 1=right)
- `y`: vertical axis (0=top, 1=bottom)

Example: `[0.2, 0.8, 0.5, 0.9]` means the error is in the bottom-left area, from 20% width / 80% height to 50% width / 90% height. The Upsampler uses this to focus the fix on that region in the next iteration. To visualize:

```python
from PIL import ImageDraw
draw = ImageDraw.Draw(image)
w, h = image.size
for err in critique.errors:
    x0, y0, x1, y1 = err.bbox
    draw.rectangle([x0*w, y0*h, x1*w, y1*h], outline="red", width=3)
```

The package also includes training scripts for the LoRAs (SFT) and for a **DPO** (Direct Preference Optimization) stage on the Upsampler, using preference pairs collected automatically from the refinement loop itself.

---

## Project structure

```
openavrl/
├── pipeline.py              # OpenAVRL2Pipeline — orchestrates the full loop (supports share_base_model=True)
├── upsampler.py             # Upsampler: prompt + critiques -> JSON
├── generator.py             # Ideogram 4.0 wrapper (JSON -> image) - swappable
├── evaluator.py             # Multimodal Evaluator: (image, JSON) -> Critique
├── schemas.py                # Pydantic models (Critique, BBoxError, GenerationResult)
├── config/
│   └── a100_train.yml        # Reference config for training/inference on A100/L40S
├── data/
│   └── prompts.txt           # Example prompts for benchmarking and data collection
├── scripts/
│   ├── a100_benchmark.py     # Time and VRAM benchmark
│   └── collect_dpo_data.py   # Preference pair collection for DPO
└── train/
    ├── train_upsampler.py    # SFT for the Upsampler LoRA
    ├── train_evaluator.py    # SFT for the Evaluator LoRA (multimodal)
    └── train_dpo.py          # DPO for the Upsampler over (chosen/rejected) pairs
```

---

## Installation

```bash
git clone https://github.com/GustaJunq/OpenAVRL
cd OpenAVRL
pip install -e .
```

### Requirements

- Python >= 3.10
- CUDA GPU (A100 40GB recommended, L40S 48GB supported; `bfloat16` + `flash_attention_2`)
- Core dependencies: `torch`, `transformers`, `peft`, `diffusers`, `accelerate`, `trl`, `pillow` (installed automatically via `pip install -e .`)

---

## Basic usage

```python
from openavrl import OpenAVRL2Pipeline

# Loads the base model (Qwen3.5 9B) + both LoRA adapters + the Ideogram 4.0 generator
pipe = OpenAVRL2Pipeline.from_pretrained(
    repo_id="SynastriaNetworks/OpenAVRL-2.0",
    device="cuda",
)

# Runs the agentic loop: generate, critique, and refine up to 3 times
image = pipe.generate(
    "Poster for an electronic music festival in São Paulo, neon style",
    n_max=3,
)

image.save("poster.png")
```

### Single adapter mode (VRAM efficient - recommended for L40S)

```python
from openavrl import OpenAVRL2Pipeline

# Loads Qwen3.5 9B once and reuses the same LoRA for both Upsampler and Evaluator
pipe = OpenAVRL2Pipeline.from_pretrained(
    repo_id="SynastriaNetworks/OpenAVRL-2.0",
    device="cuda",
    share_base_model=True,  # single base model + single LoRA
)

image = pipe.generate("Poster for an electronic music festival in São Paulo, neon style", n_max=3)
image.save("poster.png")
```

### Capturing the full trace (for analysis or training data)

`generate_with_trace` returns every critique produced at each step of the loop, along with the final JSON used for generation — useful for debugging Evaluator behavior or feeding preference-data collection.

```python
from openavrl import OpenAVRL2Pipeline

pipe = OpenAVRL2Pipeline.from_pretrained(device="cuda", share_base_model=True)

result = pipe.generate_with_trace("Lo-fi hip hop album cover", n_max=3)

print(f"Converged in {result.steps} step(s)")
for i, critique in enumerate(result.critiques, start=1):
    print(f"Step {i}: approved={critique.approved} score={critique.score}")
    for error in critique.errors:
        print(f"  - {error.type} at {error.bbox}: {error.fix}")

print(result.final_json)
```

### Assembling the pipeline manually (swapping generator to Qwen-Image)

You can instantiate each module separately — useful for swapping the base model, using your own adapters, or changing the generator to Qwen-Image, Flux, SDXL and others:

```python
from openavrl.upsampler import Upsampler
from openavrl.generator import IdeogramGenerator
from openavrl.evaluator import Evaluator
from openavrl.pipeline import OpenAVRL2Pipeline
from diffusers import QwenImagePipeline
import torch

# Custom generator example
class QwenImageGenerator:
    def __init__(self, model_id="Qwen/Qwen-Image"):
        self.pipe = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    def __call__(self, generation_json: dict):
        return self.pipe(prompt=generation_json.get("prompt","")).images[0]

U = Upsampler(model_id="Qwen/Qwen3.5-9B", lora_id="SynastriaNetworks/OpenAVRL-2.0/upsampler", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
E = Evaluator(model_id="Qwen/Qwen3.5-9B", lora_id="SynastriaNetworks/OpenAVRL-2.0/evaluator", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
G = IdeogramGenerator(model_id="ideogram-ai/ideogram-4")
# Or: G = QwenImageGenerator()

pipe = OpenAVRL2Pipeline(U, G, E, share_base_model=True)
image = pipe.generate("Minimalist logo for a coffee shop")
```

---

## Training the adapters

The three training stages expect datasets under `data/*.jsonl` (formats documented in each script) and use `peft` + `trl`.

**1. Upsampler SFT** (`prompt + critiques -> JSON`):

```bash
python -m openavrl.train.train_upsampler
```

**2. Evaluator SFT** (`image + JSON -> critique`, multimodal):

```bash
python -m openavrl.train.train_evaluator
```

**3. Upsampler DPO**, using preference pairs (`chosen` = approved JSON, `rejected` = rejected JSON) collected from the loop itself:

```bash
python openavrl/scripts/collect_dpo_data.py   # generates data/dpo_pairs.jsonl
python -m openavrl.train.train_dpo
```

Reference hyperparameters (LoRA `r=64`, `alpha=128`, batch size, LR, etc.) live in `openavrl/config/a100_train.yml`.
---

## Benchmark

```bash
python openavrl/scripts/a100_benchmark.py
```

Measures total loop time (`n_max=3`) and peak allocated VRAM, saving the resulting image to `/tmp/bench.png`.

For L40S:

```bash
python openavrl/scripts/a100_benchmark.py --device cuda --dtype bfloat16 --n_max 3 --share_base_model
```

---

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).

## Credits

Developed by [SynastrIA Networks](https://github.com/GustaJunq/OpenAVRL), based on the architecture described in the AVRL paper. Built on Ideogram 4.0 open weights (9.3B) and Qwen3.5 9B.
