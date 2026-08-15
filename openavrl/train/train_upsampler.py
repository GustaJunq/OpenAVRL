"""Train Upsampler LoRA r=64 on A100 40GB BF16"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig
from trl import SFTTrainer

model_id = "Qwen/Qwen3.5-9B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2")

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

# dataset format: {prompt: p0 + critiques, completion: json}
ds = load_dataset("json", data_files="data/upsampler.jsonl")

trainer = SFTTrainer(model, train_dataset=ds["train"], peft_config=lora_config, max_seq_length=4096, args={"per_device_train_batch_size":2, "bf16": True})
trainer.train()
trainer.save_model("./openavrl2-lora-U") # You can also save to HF if you want.
