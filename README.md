# OpenAVRL

**Agentic Visual Refinement Loop** — image generation guided by an agentic visual refinement cycle, combining a multimodal LLM (Qwen3.5 9B) with Ideogram 4.0.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)

---

## What this is

OpenAVRL implements the **AVRL** (Agentic Visual Refinement Loop) architecture: instead of generating an image in a single pass from a text prompt, the pipeline runs a **closed generate → critique → refine loop**, where the model itself visually evaluates the result and decides whether it needs to be redone.

The system is made up of three modules:

| Module | Role | Base model |
|---|---|---|
| **Upsampler** | Turns the prompt (`p0`) + critique history into a structured JSON spec for the generator | Qwen3.5 9B + `upsampler` LoRA |
| **Generator** | Generates the image from the JSON spec, using Ideogram 4.0's native JSON API | Ideogram 4.0 |
| **Evaluator** | Sees the generated image (multimodal) plus the JSON used, and returns a structured critique with errors localized by bounding box | Qwen3.5 9B + `evaluator` LoRA |

The loop runs for up to `n_max` iterations or until the Evaluator approves the result (`approved: true`). Each error it identifies feeds into the next Upsampler iteration, refining the generation JSON.

You can use 2 adapters, one for the Upsampler, and other for evaluating
Or a singular one for making both processes.

The package also includes training scripts for the LoRAs (SFT) and for a **DPO** (Direct Preference Optimization) stage on the Upsampler, using preference pairs collected automatically from the refinement loop itself.

---

## Project structure

```
openavrl/
├── pipeline.py              # OpenAVRL2Pipeline — orchestrates the full loop
├── upsampler.py             # Upsampler: prompt + critiques -> JSON
├── generator.py             # Ideogram 4.0 wrapper (JSON -> image)
├── evaluator.py             # Multimodal Evaluator: (image, JSON) -> Critique
├── schemas.py                # Pydantic models (Critique, BBoxError, GenerationResult)
├── config/
│   └── a100_train.yml        # Reference config for training/inference on A100
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
git clone https://github.com/GustaJunq/OpenAVRL-pkg
cd OpenAVRL-pkg
pip install -e .
```

### Requirements

- Python >= 3.10
- CUDA GPU (the pipeline is designed around an A100 40GB; `bfloat16` + `flash_attention_2`)
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

### Capturing the full trace (for analysis or training data)

`generate_with_trace` returns every critique produced at each step of the loop, along with the final JSON used for generation — useful for debugging Evaluator behavior or feeding preference-data collection.

```python
from openavrl import OpenAVRL2Pipeline

pipe = OpenAVRL2Pipeline.from_pretrained(device="cuda")

result = pipe.generate_with_trace("Lo-fi hip hop album cover", n_max=3)

print(f"Converged in {result.steps} step(s)")
for i, critique in enumerate(result.critiques, start=1):
    print(f"Step {i}: approved={critique.approved} score={critique.score}")
    for error in critique.errors:
        print(f"  - {error.type} at {error.bbox}: {error.fix}")

print(result.final_json)
```

### Assembling the pipeline manually

You can also instantiate each module separately — useful for swapping the base model, using your own adapters, or debugging each stage in isolation:

```python
from openavrl.upsampler import Upsampler
from openavrl.generator import IdeogramGenerator
from openavrl.evaluator import Evaluator
from openavrl.pipeline import OpenAVRL2Pipeline

U = Upsampler(model_id="Qwen/Qwen3.5-9B", lora_id="SynastriaNetworks/OpenAVRL-2.0/upsampler")
E = Evaluator(model_id="Qwen/Qwen3.5-9B", lora_id="SynastriaNetworks/OpenAVRL-2.0/evaluator")
G = IdeogramGenerator(model_id="ideogram-ai/ideogram-4-fp8")

pipe = OpenAVRL2Pipeline(U, G, E)
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

---

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).

## Credits

Developed by [SynastrIA Networks](https://github.com/GustaJunq/OpenAVRL-pkg), based on the architecture described in the AVRL paper.
