#!/usr/bin/env python3
"""
train_refiner.py

Fine-tune the Refiner adapter (PEFT / LoRA) for the Refiner model in this repo.

This script is intentionally a practical, general starting point and assumes
training data in JSON/JSONL where each example contains at least:
  - prompt: the original p0 prompt (string)
  - target: the expected refined JSON string that the model should output (string)
  - image: optional path to an image file that should be provided to the multimodal
           processor (string, optional)

Usage examples:
  python train_refiner.py --train-file data/train.jsonl --output-dir ./adapter_refiner

Requirements: transformers, datasets, peft, accelerate, torch, pillow
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class DataCollatorForRefiner:
    processor: AutoProcessor
    tokenizer_pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = 4096

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Turn a list of examples into a batch for causal LM training.

        We implement "prompt + target" teacher forcing: the model input is the
        concatenation of `prompt` and `target` and labels mask out the prompt
        tokens (set to -100) so the loss is computed only on the target tokens.

        Each example may include an `image` (local path) which will be loaded
        and passed through the processor together with the text.
        """
        texts = []
        targets = []
        images = []
        for ex in features:
            prompt = ex.get("prompt", "")
            target = ex.get("target", "")
            texts.append(prompt)
            targets.append(target)
            img_path = ex.get("image")
            if img_path:
                images.append(Image.open(img_path).convert("RGB"))
            else:
                images.append(None)

        # Prepare tokenization for prompt+target pairs so we can build labels
        # Use tokenizer from the processor if available
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor has no tokenizer attribute")

        # Tokenize targets separately (we will mask prompt tokens in labels)
        tokenized_prompts = tokenizer(texts, add_special_tokens=False)
        tokenized_targets = tokenizer(targets, add_special_tokens=False)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for p_ids, t_ids in zip(tokenized_prompts["input_ids"], tokenized_targets["input_ids"]):
            input_ids = p_ids + t_ids + [tokenizer.eos_token_id]
            labels = [-100] * len(p_ids) + t_ids + [tokenizer.eos_token_id]

            # Truncate if needed
            if self.max_length and len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length :]
                labels = labels[-self.max_length :]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        # Pad sequences to the same length
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        attention_mask = (batch_input_ids != (tokenizer.pad_token_id or 0)).long()

        out: Dict[str, Any] = {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }

        # If there are images, add pixel values via the processor so the model
        # receives multimodal inputs. The processor can batch images and text
        # together; to be conservative we call it separately and merge results.
        has_images = any(img is not None for img in images)
        if has_images:
            images_only = [img for img in images]
            # The processor expects a list of images and a list of text; provide prompts
            proc = self.processor(text=texts, images=images_only, return_tensors="pt", padding=True)
            # Merge pixel_values if present
            if "pixel_values" in proc:
                out["pixel_values"] = proc["pixel_values"]

            # Some processors also include input_ids/attention_mask for text; we keep
            # the manual construction above because we need the label-masking trick.

        return out


def build_dataset(path: str):
    if path.endswith(".jsonl") or path.endswith(".json"):
        ds = load_dataset("json", data_files=path, split="train")
    else:
        # Try generic dataset loader - user can pass a folder/other formats
        ds = load_dataset(path, split="train")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B", help="base model id")
    parser.add_argument("--train-file", required=True, help="train file (json/jsonl)")
    parser.add_argument("--validation-file", default=None, help="validation file (json/jsonl)")
    parser.add_argument("--output-dir", required=True, help="where to save the trained adapter")
    parser.add_argument("--adapter-name", default="refiner", help="PEFT adapter name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Load model with device_map auto so it can be sharded across GPUs if available
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing (saves memory)")
        model.gradient_checkpointing_enable()

    # Prepare for PEFT / LoRA
    print("Preparing model for PEFT/LoRA")
    try:
        # For some quantized workflows this step is required; it's harmless otherwise
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # prepare_model_for_kbit_training may not be required or may fail if not using k-bit
        pass

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    print("Loading datasets...")
    train_ds = build_dataset(args.train_file)
    eval_ds = build_dataset(args.validation_file) if args.validation_file else None

    # Ensure every example has 'prompt' and 'target' fields
    def _ensure_fields(ex):
        if "prompt" not in ex or "target" not in ex:
            raise ValueError("Each dataset example must contain 'prompt' and 'target' fields")
        return ex

    train_ds = train_ds.map(_ensure_fields)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_ensure_fields)

    data_collator = DataCollatorForRefiner(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        report_to="none",
        save_total_limit=3,
        fp16_full_eval=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=getattr(processor, "tokenizer", None),
    )

    print("Starting training...")
    trainer.train()

    # Save the adapter (PEFT) separately from the base model
    print(f"Saving adapter to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir)

    # Also save a tiny metadata file describing the adapter
    meta = {
        "model_id": args.model_id,
        "adapter_name": args.adapter_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    with open(os.path.join(args.output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
