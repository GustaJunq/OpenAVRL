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
        # repo contains base + LoRA adapter: we're going to load the base once
        # and attach the SAME adapter instance for both Upsampler and Evaluator to save VRAM
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel

        base_model_id = "Qwen/Qwen3.5-9B"

        # load base processor + model once
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        # Attach the SAME LoRA adapter (from the repo) onto the shared base model.
        # This creates one PeftModel instance containing the adapter; we then pass
        # that exact instance into both Upsampler and Evaluator so they reuse it.
        shared_lora_id = f"{repo_id}/upsampler"  # use the same adapter for both
        shared_adapter_name = "shared_upsampler"
        shared_model = PeftModel.from_pretrained(base_model, shared_lora_id, adapter_name=shared_adapter_name)
        shared_model.set_adapter(shared_adapter_name)

        # create Upsampler and Evaluator reusing the same PeftModel + processor
        U = Upsampler(model=shared_model, processor=processor, lora_id=None, adapter_name="upsampler")
        E = Evaluator(model=shared_model, processor=processor, lora_id=None, adapter_name="evaluator")

        # Generator remains separate
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
