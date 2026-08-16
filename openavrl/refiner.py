import json
import re
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .schemas import BBoxError, Critique


class Refiner:
    """Qwen3.5 9B Multimodal - single shared adapter.

    Replaces the separate Upsampler ('p0 + critique -> JSON') and Evaluator
    ('image + JSON -> critique') adapters with ONE adapter that does both in a
    single forward pass:

      1. It reasons inside <think>...</think>: given the previous image (if
         any) and the JSON used to generate it, it evaluates whether the
         result is good enough, localizing any errors with a bounding box.
      2. After </think>, its final answer is ONLY the next JSON generation
         spec — never the critique itself.

    Because the image for step N only exists *after* the generator has run on
    step N's JSON, evaluation necessarily lags generation by one call: each
    `step()` evaluates the image passed in (produced by the *previous* JSON)
    and proposes the *next* JSON. On the very first call there is nothing to
    evaluate yet (`image=None`), so it just upsamples the prompt.

    Approval/score/errors are not asked for as a second structured JSON
    anymore — they're parsed out of the reasoning trace via plain markers
    (`APPROVED: true|false`, `SCORE: <float>`, `ERROR | ...`), since the
    final answer slot is reserved for the JSON spec only.
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    _APPROVED_RE = re.compile(r"APPROVED:\s*(true|false)", re.IGNORECASE)
    _SCORE_RE = re.compile(r"SCORE:\s*([0-9.]+)")
    _ERROR_RE = re.compile(
        r"ERROR\s*\|\s*type=(?P<type>[^|]+)\|\s*bbox=\[(?P<bbox>[^\]]+)\]\s*\|\s*fix=(?P<fix>.+)"
    )
    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-9B",
        lora_id: Optional[str] = None,
        device: str = "cuda",
        model=None,
        processor=None,
        adapter_name: str = "refiner",
    ):
        if processor is not None and model is not None:
            self.processor = processor
            self.model = model
        else:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

        if lora_id:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_id, adapter_name=adapter_name)
            self.model.set_adapter(adapter_name)

    def step(
        self, prompt: str, critique_history: str = "", image=None
    ) -> Tuple[dict, Critique, str]:
        """Runs one reasoning + refinement step.

        Args:
            prompt: the original p0 prompt.
            critique_history: accumulated text log of previous errors.
            image: the image produced by the *previous* JSON, or None on the
                first call (nothing to evaluate yet).

        Returns:
            (next_generation_json, critique_of_`image`, raw_thinking_text)
            When `image` is None, the returned Critique is a placeholder
            (`approved=False`, "first pass, not yet evaluated") since there
            was nothing to judge.
        """
        system = (
            "You are the AVRL agent. Inside <think>...</think>, look at the "
            "provided image (if any) against the prompt and prior critique "
            "history, decide if it is good enough, and log any errors as "
            "lines of the form "
            "'ERROR | type=<type> | bbox=[x_min,y_min,x_max,y_max] | fix=<fix>'. "
            "End the reasoning with 'APPROVED: true' or 'APPROVED: false' and "
            "'SCORE: <0-1 float>'. "
            "After </think>, output ONLY the refined JSON spec for the image "
            "generator — no markdown, no commentary outside the JSON."
        )
        user_msg = f"Prompt: {prompt}\nPrevious critiques:\n{critique_history}\n"
        text_input = f"{system}\n{user_msg}"

        if image is not None:
            inputs = self.processor(
                text=text_input, images=image, return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(text=text_input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=3072, temperature=0.7, do_sample=True)
        raw = self.processor.decode(out[0], skip_special_tokens=True)

        thinking, final_answer = self._split_think(raw)

        if image is not None:
            critique = self._parse_critique(thinking)
        else:
            critique = Critique(approved=False, refined_instruction="first pass, not yet evaluated", score=0.0)

        generation_json = self._extract_json(final_answer)
        return generation_json, critique, thinking

    def _split_think(self, text: str) -> Tuple[str, str]:
        if self.THINK_OPEN in text and self.THINK_CLOSE in text:
            start = text.index(self.THINK_OPEN) + len(self.THINK_OPEN)
            end = text.index(self.THINK_CLOSE)
            return text[start:end], text[end + len(self.THINK_CLOSE):]
        # Model didn't emit explicit <think> tags (e.g. before fine-tuning) —
        # fall back to treating the whole thing as both, so JSON extraction
        # below still has a shot at finding the spec.
        return text, text

    def _parse_critique(self, thinking: str) -> Critique:
        approved_match = self._APPROVED_RE.search(thinking)
        score_match = self._SCORE_RE.search(thinking)

        errors = []
        for m in self._ERROR_RE.finditer(thinking):
            try:
                bbox = [float(x.strip()) for x in m.group("bbox").split(",")]
            except ValueError:
                continue
            errors.append(BBoxError(type=m.group("type").strip(), bbox=bbox, fix=m.group("fix").strip()))

        return Critique(
            approved=bool(approved_match and approved_match.group(1).lower() == "true"),
            errors=errors,
            refined_instruction=thinking.strip()[:1000],
            score=float(score_match.group(1)) if score_match else 0.0,
        )

    def _extract_json(self, final_answer: str) -> dict:
        m = self._JSON_RE.search(final_answer)
        if not m:
            raise ValueError(f"No JSON found in final answer: {final_answer[:500]}")
        return json.loads(m.group(0))
          
