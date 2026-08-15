"""DPO on Upsampler using preference pairs from Evaluator"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

base_id = "Qwen/Qwen3.5-9B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./openavrl2-lora-U")

# dataset: {prompt: p0+history, chosen: j_win, rejected: j_lose}
ds = load_dataset("json", data_files="data/dpo_pairs.jsonl")

config = DPOConfig(beta=0.1, bf16=True, per_device_train_batch_size=1)
trainer = DPOTrainer(model, ref_model=None, args=config, train_dataset=ds["train"])
trainer.train()
trainer.save_model("./openavrl2-lora-U-dpo")
