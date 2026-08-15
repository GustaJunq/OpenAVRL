
"""Train Evaluator LoRA multimodal - (image, json) -> Critique JSON"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig
from trl import SFTTrainer

model_id = "Qwen/Qwen3.5-9B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")

# dataset format: {image_path, json_spec, critique_json}
# critique_json = {"approved": bool, "errors": [...], "refined_instruction": str}
ds = load_dataset("json", data_files="data/evaluator.jsonl")

def formatting_func(example):
    return f"<image>\nJSON:{example['json_spec']}\nCritique:{example['critique_json']}"

trainer = SFTTrainer(model, train_dataset=ds["train"], peft_config=lora_config, formatting_func=formatting_func, args={"bf16":True, "per_device_train_batch_size":1})
trainer.train()
trainer.save_model("./openavrl2-lora-E")
