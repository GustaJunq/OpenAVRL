# OpenAVRL - Documentation

## Table of Contents
1. [Architecture](#architecture)
2. [Modules](#modules)
3. [Schemas](#schemas)
4. [Installation Details](#installation-details)
5. [Usage Modes](#usage-modes)
6. [Single vs Dual Adapter](#single-vs-dual-adapter)
7. [Bounding Box (bbox) System](#bounding-box-bbox-system)
8. [Swapping Generators](#swapping-generators)
9. [Training](#training)
10. [Benchmarking](#benchmarking)
11. [Configuration](#configuration)
12. [Troubleshooting - L40S / Lightning AI](#troubleshooting)

---

## Architecture

OpenAVRL implements AVRL (Agentic Visual Refinement Loop):

```
p0 (user prompt)
  |
  v
[Upsampler: p0 + critique_history -> JSON_spec]
  |
  v
[Generator: JSON_spec -> image_i]
  |
  v
[Evaluator: (image_i, JSON_spec) -> Critique {approved, score, errors[]}]
  |
  +-- if approved or i == n_max -> return image_i
  +-- else -> append critique to history, loop again
```

This mirrors GPT Image 2's thinking mode, where the model critiques its own output.

The JSON spec used by Ideogram 4.0 is a structured prompt:
```json
{
  "prompt": "detailed description",
  "negative_prompt": "...",
  "style": "neon, 90s rave",
  "aspect_ratio": "16:9",
  "text_regions": [
    {"text": "FESTA 2026", "bbox": [0.2, 0.8, 0.8, 0.9], "style": "bold"}
  ]
}
```

## Modules

### 1. Upsampler (`openavrl/upsampler.py`)
- **Base:** Qwen/Qwen3.5-9B
- **LoRA:** `SynastriaNetworks/OpenAVRL-2.0/upsampler` or combined
- **Input:** `prompt: str`, `critiques: List[Critique]` (from previous iterations)
- **Output:** `dict` JSON spec for generator
- **Training:** SFT + DPO

Responsible for converting vague user prompt + past errors into precise generation instructions.

### 2. Generator (`openavrl/generator.py`)
- **Base:** Ideogram 4.0 (9.3B open weights) - default `ideogram-ai/ideogram-4`
- **Interface:** `__call__(json_spec: dict) -> PIL.Image`
- Swappable: any class implementing same interface works.

Ideogram 4.0 supports native JSON prompting with bbox layout control, transparent backgrounds, and 2K resolution.

### 3. Evaluator (`openavrl/evaluator.py`)
- **Base:** Qwen/Qwen3.5-9B (multimodal vision)
- **LoRA:** `SynastriaNetworks/OpenAVRL-2.0/evaluator` or combined
- **Input:** `image: PIL.Image`, `json_spec: dict`
- **Output:** `Critique`

Multimodal model that sees the image and returns structured errors.

## Schemas

Defined in `openavrl/schemas.py`:

```python
class BBoxError(BaseModel):
    type: str  # e.g. "text_typo", "text_missing", "layout_overlap", "color_error", "object_missing"
    bbox: List[float]  # [x_min, y_min, x_max, y_max] normalized 0-1
    fix: str   # instruction for upsampler: "remove extra 6", "center text"

class Critique(BaseModel):
    approved: bool
    score: float  # 0-1
    reasoning: str  # chain-of-thought
    errors: List[BBoxError]

class GenerationResult(BaseModel):
    image: PIL.Image
    steps: int
    critiques: List[Critique]
    final_json: dict
```

### BBox Coordinate System
- Normalized 0-1, relative to image dimensions
- Origin (0,0) = top-left
- `[x_min, y_min, x_max, y_max]`
- To convert to pixels: `[x_min*W, y_min*H, x_max*W, y_max*H]`

## Installation Details

### Lightning AI Studio + L40S (48GB)

```bash
git clone https://github.com/GustaJunq/OpenAVRL openavrl_repo
cd openavrl_repo
pip install -U pip wheel
pip install flash-attn --no-build-isolation  # required for Qwen3.5 9B on L40S
pip install -e .
pip install accelerate peft trl diffusers transformers
```

Login for Ideogram 4.0 open weights (gated but free):
```bash
huggingface-cli login
```

### Requirements
- Python >=3.10
- CUDA GPU: A100 40GB (reference), L40S 48GB supported
- dtype: `bfloat16` + `flash_attention_2` recommended

## Usage Modes

### Mode A: Simple generate

```python
from openavrl import OpenAVRL2Pipeline
pipe = OpenAVRL2Pipeline.from_pretrained(repo_id="SynastriaNetworks/OpenAVRL-2.0", device="cuda")
image = pipe.generate("Poster for an electronic music festival in São Paulo, neon style", n_max=3)
```

### Mode B: With trace (debugging / data collection)

```python
result = pipe.generate_with_trace(prompt, n_max=3)
print(f"Steps: {result.steps}")
for c in result.critiques:
    print(c.approved, c.score, c.errors)
# result.image, result.final_json
```

### Mode C: Manual assembly

```python
from openavrl.upsampler import Upsampler
from openavrl.evaluator import Evaluator
from openavrl.generator import IdeogramGenerator
from openavrl.pipeline import OpenAVRL2Pipeline

U = Upsampler(model_id="Qwen/Qwen3.5-9B", lora_id="...", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
E = Evaluator(model_id="Qwen/Qwen3.5-9B", lora_id="...", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
G = IdeogramGenerator(model_id="ideogram-ai/ideogram-4")

pipe = OpenAVRL2Pipeline(U, G, E, share_base_model=True)
```

## Single vs Dual Adapter

| Mode | Base Models Loaded | VRAM Peak (n_max=3) | Quality |
|---|---|---|---|
| Dual (2 LoRAs) | Qwen3.5 9B x2 + Ideogram 4.0 9.3B | ~38-42GB | Highest |
| Single (shared LoRA) | Qwen3.5 9B x1 + Ideogram 4.0 9.3B | ~22-26GB | Almost the same as the dual |

**When to use single:**
- L40S 48GB, RTX 4090 24GB (with offload), Lightning AI
- Set `share_base_model=True` in pipeline or `from_pretrained(..., share_base_model=True)`

The commit `Make pipeline load a single base model + attach the SAME LoRA` implements this by sharing the underlying `transformers` model and attaching the same PEFT adapter for both roles.

## Bounding Box (bbox) System

The Evaluator outputs errors localized by bbox. This enables targeted refinement.

**Example critique:**
```json
{
  "approved": false,
  "score": 0.62,
  "errors": [
    {"type": "text_typo", "bbox": [0.21, 0.82, 0.52, 0.91], "fix": "text should be 'FESTA' not 'FESTTA'"},
    {"type": "layout_overlap", "bbox": [0.1, 0.1, 0.9, 0.3], "fix": "logo overlaps headline"}
  ]
}
```

Visualization helper (see README):
```python
from PIL import ImageDraw
def draw_errors(image, critique):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w,h = img.size
    for e in critique.errors:
        x0,y0,x1,y1 = e.bbox
        draw.rectangle([x0*w, y0*h, x1*w, y1*h], outline="red", width=3)
        draw.text((x0*w, max(0,y0*h-12)), f"{e.type}: {e.fix}", fill="red")
    return img
```

## Swapping Generators

To swap Ideogram 4.0 for Qwen-Image, Flux, SDXL:

1. Implement class with `__call__(self, json_spec: dict) -> PIL.Image`
2. Extract `prompt` from `json_spec`
3. Return PIL Image

**Qwen-Image example:**

```python
from diffusers import QwenImagePipeline
import torch

class QwenImageGenerator:
    def __init__(self):
        self.pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
    def __call__(self, json_spec):
        return self.pipe(prompt=json_spec["prompt"]).images[0]
```

**Flux Schnell example:**

```python
from diffusers import FluxPipeline
class FluxGenerator:
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    def __call__(self, json_spec):
        return self.pipe(prompt=json_spec["prompt"]).images[0]
```

Note: Ideogram 4.0 has best text rendering. Qwen-Image has better prompt adherence for complex scenes. Tradeoff.

## Training

Datasets expected under `data/*.jsonl`:

### 1. Upsampler SFT
- Input: `{"prompt": str, "critiques": [...], "target_json": dict}`
- Script: `python -m openavrl.train.train_upsampler`
- LoRA: r=64, alpha=128 (see `config/a100_train.yml`)

### 2. Evaluator SFT (multimodal)
- Input: `{"image_path": str, "json_spec": dict, "target_critique": Critique}`
- Script: `python -m openavrl.train.train_evaluator`

### 3. DPO for Upsampler
- Collect pairs from loop itself:
```bash
python openavrl/scripts/collect_dpo_data.py  # outputs data/dpo_pairs.jsonl
python -m openavrl.train.train_dpo
```
- `chosen` = JSON that got approved, `rejected` = JSON that failed

Hyperparameters reference: `openavrl/config/a100_train.yml`

## Benchmarking

```bash
python openavrl/scripts/a100_benchmark.py --n_max 3
```

Outputs time per iteration and peak VRAM, saves image to `/tmp/bench.png`.

L40S variant:
```bash
python openavrl/scripts/a100_benchmark.py --device cuda --dtype bfloat16 --share_base_model --n_max 3
```

## Configuration

`openavrl/config/a100_train.yml` contains LoRA config, LR, batch size, etc.

Key settings for L40S:
```yaml
torch_dtype: bfloat16
attn_implementation: flash_attention_2
share_base_model: true
low_cpu_mem_usage: true
```

## Troubleshooting

### OOM on L40S
- Use `share_base_model=True`
- Reduce `n_max` from 3 to 2
- Enable offload: `pipe.enable_model_cpu_offload()` or `enable_sequential_cpu_offload()`
- Use `torch.bfloat16` not `float32`

### Flash Attention missing
- Required for Qwen3.5 9B to fit: `pip install flash-attn --no-build-isolation`
- If fails to compile, fallback to `attn_implementation="eager"` but VRAM will increase.

### Text still has typos after 3 loops
- Increase `n_max=5`
- Check Evaluator scores - if stuck at 0.6, Evaluator LoRA may need fine-tuning on your domain