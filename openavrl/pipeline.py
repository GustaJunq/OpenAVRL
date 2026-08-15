import torch
from typing import List
from PIL import Image
from .upsampler import Upsampler
from .generator import IdeogramGenerator
from .evaluator import Evaluator
from .schemas import Critique, GenerationResult

class OpenAVRL2Pipeline:
    def __init__(self, upsampler, generator, evaluator):
        self.U = upsampler
        self.G = generator
        self.E = evaluator

    @classmethod
    def from_pretrained(cls, repo_id="SynastriaNetworks/OpenAVRL-2.0", device="cuda"):
        # repo contains base + two LoRA adapters: adapter_upsampler / adapter_evaluator
        U = Upsampler(model_id="Qwen/Qwen3.5-9B", lora_id=f"{repo_id}/upsampler", device=device)
        # share base to save VRAM: load E from same base model instance
        E = Evaluator(model_id="Qwen/Qwen3.5-9B", lora_id=f"{repo_id}/evaluator", device=device)
        # For true 40GB sharing, you can load both LoRAs on same base via PEFT multi-adapter.
        # Simplified here for clarity.
        G = IdeogramGenerator(model_id="ideogram-ai/ideogram-4-fp8", device=device)
        return cls(U, G, E)

    @torch.inference_mode()
    def generate(self, p0: str, n_max: int = 3, save_steps: bool = False) -> Image.Image:
        critiques: List[Critique] = []
        critique_history = ""
        final_json = {}
        last_image = None

        for i in range(n_max):
            # 1. Upsample / Refine
            final_json = self.U.generate_json(p0, critique_history)

            # 2. Generate with Ideogram 4.0 native JSON
            last_image = self.G(final_json)

            # 3. Multimodal critique (Qwen3.5 sees the image!)
            r = self.E.critique(last_image, final_json)
            critiques.append(r)

            if r.approved:
                print(f"[AVRL2] Approved at step {i+1}/{n_max} score={r.score}")
                break

            critique_history += f"\n- Step {i+1} error: {r.refined_instruction} | errors: {r.errors}"

        return last_image

    def generate_with_trace(self, p0: str, n_max: int = 3) -> GenerationResult:
        # same as generate but returns full trace for DPO data collection
        critiques = []
        history = ""
        final_json = {}
        last_image = None
        for i in range(n_max):
            final_json = self.U.generate_json(p0, history)
            last_image = self.G(final_json)
            r = self.E.critique(last_image, final_json)
            critiques.append(r)
            if r.approved:
                break
            history += f"\n{r.refined_instruction}"

        # save temp
        last_image.save("/tmp/avrl2_last.png")
        return GenerationResult(
            image_path="/tmp/avrl2_last.png",
            final_json=final_json,
            critiques=critiques,
            steps=len(critiques)
        )
      
