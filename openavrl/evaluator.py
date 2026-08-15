import torch, json, re
from transformers import AutoModelForCausalLM, AutoProcessor
from .schemas import Critique, BBoxError

class Evaluator:
    """Qwen3.5 9B Multimodal - adapter 'evaluator' : (image, json) -> Critique"""
    def __init__(self, model_id="Qwen/Qwen3.5-9B", lora_id=None, device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        if lora_id:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_id, adapter_name="evaluator")
            self.model.set_adapter("evaluator")

    def critique(self, image, json_spec: dict) -> Critique:
        system = """You are the Evaluator. You see the generated image and its JSON.
Return JSON only: {"approved": bool, "errors": [{"type": str, "bbox": [x_min,y_min,x_max,y_max], "fix": str}], "refined_instruction": str, "score": float}"""
        inputs = self.processor(
            text=system + f"\nJSON: {json.dumps(json_spec)}",
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return Critique(approved=False, refined_instruction=text[:500], score=0.0)
        data = json.loads(m.group(0))
        errors = [BBoxError(**e) for e in data.get("errors", [])]
        return Critique(
            approved=data.get("approved", False),
            errors=errors,
            refined_instruction=data.get("refined_instruction", ""),
            score=data.get("score", 0.0)
        )
