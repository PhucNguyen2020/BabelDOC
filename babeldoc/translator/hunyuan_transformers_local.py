"""
Local Tencent HY-MT inference via Hugging Face ``transformers`` (in-process).

Matches the official model card flow: ``AutoTokenizer`` + ``AutoModelForCausalLM``,
``apply_chat_template``, and recommended generation settings (no default system prompt).

Install::

    pip install 'babeldoc[hunyuan-transformers]'

Also install a PyTorch build suitable for your platform (CPU/CUDA), e.g. from
https://pytorch.org/get-started/locally/

FP8 checkpoints: per upstream docs, adjust ``config.json`` (``ignored_layers`` → ``ignore``)
and ``compressed-tensors`` version before loading.
"""

from __future__ import annotations

import logging
import threading

from babeldoc.translator.hunyuan_mt import build_hunyuan_user_prompt
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)


def _chat_template_to_input_ids(tokenized, torch_module):
    """``apply_chat_template(..., return_tensors='pt')`` may return a Tensor or BatchEncoding."""
    if isinstance(tokenized, torch_module.Tensor):
        return tokenized
    if hasattr(tokenized, "input_ids"):
        return tokenized["input_ids"]
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return tokenized["input_ids"]
    raise TypeError(
        f"Unexpected apply_chat_template return type {type(tokenized)!r}; "
        "expected Tensor or mapping with 'input_ids'."
    )


class HunyuanTransformersTranslator(BaseTranslator):
    """HY-MT using ``transformers`` locally; OpenAI HTTP is not used."""

    name = "hunyuan_trf"

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        model_name_or_path: str,
        ignore_cache: bool = False,
        device_map: str | dict | None = "auto",
        torch_dtype: str | None = None,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        try:
            import torch
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Hunyuan HY-MT (transformers) requires optional dependencies. "
                "Install with: pip install 'babeldoc[hunyuan-transformers]' "
                "and a compatible PyTorch build (see pytorch.org)."
            ) from e

        self._torch = torch
        self._model_name_or_path = model_name_or_path
        self.add_cache_impact_parameters("model", model_name_or_path)
        self.add_cache_impact_parameters("hunyuan_transformers_local", True)

        logger.info(
            "Loading HY-MT tokenizer/model from %s (this may download weights)...",
            model_name_or_path,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        load_kw: dict = {"device_map": device_map}
        if torch_dtype:
            td = getattr(torch, torch_dtype, None)
            if td is not None:
                load_kw["torch_dtype"] = td
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kw,
        )
        self._model.eval()
        self._generate_lock = threading.Lock()
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def _generate_text(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        tokenized = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        input_ids = _chat_template_to_input_ids(tokenized, self._torch).to(device)
        input_len = int(input_ids.shape[-1])

        with self._generate_lock:
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.6,
                    top_k=20,
                    repetition_penalty=1.05,
                )

        generated_ids = outputs[0][input_len:]
        self.prompt_token_count.inc(input_len)
        n_new = int(generated_ids.shape[0])
        self.completion_token_count.inc(n_new)
        self.token_count.inc(input_len + n_new)
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        prompt = build_hunyuan_user_prompt(self.lang_in, self.lang_out, text)
        return self._generate_text(prompt)

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None
        return self._generate_text(text)
