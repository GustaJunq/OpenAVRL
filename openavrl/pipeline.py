import torch
from typing import List
from PIL import Image
from .refiner import Refiner
from .generator import IdeogramGenerator
from .schemas import Critique, GenerationResult


class OpenAVRL2Pipeline:
    """Orchestrates the closed generate -> think/evaluate -> refine loop.

    Since the single Refiner adapter evaluates the *previous* image while
    proposing the *next* JSON (see refiner.py docstring), evaluation lags
    generation by one call. That's why the loop below checks `r.approved`
    for the image already on hand before generating a new one, and why an
    extra evaluation-only pass runs after the loop if the budget is
    exhausted without an approval — otherwise the very last image generated
    would be returned without ever having been judged.
    """

    def __init__(self, refiner: Refiner, generator: IdeogramGenerator):
        self.R = refiner
        self.G = generator

    @classmethod
    def from_pretrained(cls, repo_id: str = "SynastriaNetworks/OpenAVRL-2.0", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel

        base_model_id = "Qwen/Qwen3.5-9B"

        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Single adapter now — no separate upsampler/evaluator LoRAs to juggle.
        adapter_id = f"{repo_id}/refiner"
        model = PeftModel.from_pretrained(base_model, adapter_id, adapter_name="refiner")
        model.set_adapter("refiner")

        R = Refiner(model=model, processor=processor)
        G = IdeogramGenerator(model_id="ideogram-ai/ideogram-4-fp8", device=device)
        return cls(R, G)

    @torch.inference_mode()
    def generate(self, p0: str, n_max: int = 3, save_steps: bool = False) -> Image.Image:
        critique_history = ""
        last_image = None

        for i in range(n_max):
            final_json, r, _thinking = self.R.step(p0, critique_history, image=last_image)

            if last_image is not None:
                if r.approved:
                    print(f"[AVRL2] Approved at step {i}/{n_max} score={r.score}")
                    return last_image
                critique_history += f"\n- Step {i} error: {r.refined_instruction} | errors: {r.errors}"

            last_image = self.G(final_json)

        # Budget exhausted: the image from the final iteration was generated
        # but never evaluated yet (see docstring). One last evaluation-only
        # pass keeps the reported score/approval consistent with what we
        # actually return.
        _, last_critique, _thinking = self.R.step(p0, critique_history, image=last_image)
        print(f"[AVRL2] Budget exhausted after {n_max} steps, approved={last_critique.approved} score={last_critique.score}")
        return last_image

    def generate_with_trace(self, p0: str, n_max: int = 3) -> GenerationResult:
        # same as generate but returns full trace for DPO data collection
        critiques: List[Critique] = []
        history = ""
        last_image = None
        final_json = {}

        for i in range(n_max):
            final_json, r, _thinking = self.R.step(p0, history, image=last_image)
            if last_image is not None:
                critiques.append(r)
                if r.approved:
                    break
                history += f"\n{r.refined_instruction}"
            last_image = self.G(final_json)
        else:
            # loop ran out of iterations without breaking on approval —
            # evaluate the final image too so the trace isn't missing a step
            _, r, _thinking = self.R.step(p0, history, image=last_image)
            critiques.append(r)

        # save temp
        last_image.save("/tmp/avrl2_last.png")
        return GenerationResult(
            image_path="/tmp/avrl2_last.png",
            final_json=final_json,
            critiques=critiques,
            steps=len(critiques),
        )
