import torch
from diffusers import AutoPipelineForText2Image

class IdeogramGenerator:
    """Wrapper for Ideogram 4.0 that expects JSON spec natively"""
    def __init__(self, model_id="ideogram-ai/ideogram-4.0", device="cuda"):
        # Ideogram 4.0 uses diffusers-compatible pipeline with json_spec kwarg
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)

    def __call__(self, json_spec: dict):
        # Ideogram 4.0 native JSON input - as per your internal API
        image = self.pipe(json_spec=json_spec, num_inference_steps=28, guidance_scale=4.5).images[0]
        return image
      
