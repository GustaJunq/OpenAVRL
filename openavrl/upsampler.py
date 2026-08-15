import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class Upsampler:
    """Qwen3.5 9B Multimodal - adapter 'upsampler' : p0 + critique -> JSON"""
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
            self.model = PeftModel.from_pretrained(self.model, lora_id, adapter_name="upsampler")
            self.model.set_adapter("upsampler")

    def generate_json(self, prompt: str, critique_history: str = "") -> dict:
        system = "You are the Upsampler. Output ONLY valid JSON that Ideogram 4.0 expects. No markdown."
        user_msg = f"Base prompt: {prompt}\nPrevious critiques:\n{critique_history}\nGenerate refined JSON."
        inputs = self.processor(text=f"{system}\n{user_msg}", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        # extract last json block
        import json, re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in: {text[:500]}")
        return json.loads(m.group(0))
      
