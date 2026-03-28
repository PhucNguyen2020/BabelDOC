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

# When --hunyuan-max-new-tokens is omitted, cap each generate() separately. Using the
# entire remaining context (e.g. 100k+ new tokens) in one forward pass can run for
# hours and look "stuck"; long documents still use many sequential generates.
_DEFAULT_PER_CALL_NEW_TOKEN_CEILING = 32768


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
        max_new_tokens: int | None = None,
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
        # Avoid generate() stopping immediately on some checkpoints when pad is unset.
        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
            self._tokenizer, "eos_token_id", None
        ) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
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
        # None = use all remaining positions in the model context per call (minus prompt).
        # A positive int caps generation (still clipped to context room to avoid HF errors).
        self._max_new_tokens_cap = max_new_tokens
        self._generate_lock = threading.Lock()
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

    def _effective_context_length(self) -> int:
        """Upper bound on total sequence length (prompt + new tokens) for this model."""
        m = self._model.config
        ctx = getattr(m, "max_position_embeddings", None)
        if not isinstance(ctx, int) or ctx <= 0:
            ctx = getattr(m, "n_positions", None)
        if not isinstance(ctx, int) or ctx <= 0:
            ctx = 32768
        tok_ml = getattr(self._tokenizer, "model_max_length", None)
        # Tokenizers often use a huge sentinel; only treat as a real cap when reasonable.
        if isinstance(tok_ml, int) and 0 < tok_ml <= 1_000_000:
            ctx = min(ctx, tok_ml)
        return ctx

    def _max_new_tokens_for(self, input_len: int) -> int:
        """Budget for generate(); cannot exceed model context minus prompt."""
        ctx = self._effective_context_length()
        reserve = 64
        room = ctx - input_len - reserve
        if room < 1:
            room = 1
        if self._max_new_tokens_cap is not None:
            return min(self._max_new_tokens_cap, room)
        return min(room, _DEFAULT_PER_CALL_NEW_TOKEN_CEILING)

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
        budget = self._max_new_tokens_for(input_len)

        # Official HY-MT inference params (Tencent README): temperature / top_p / top_k /
        # repetition_penalty. Use them for both plain translation and JSON batch prompts.
        # Greedy decoding often yields immediate EOS or tiny outputs on long structured
        # prompts; sampling matches upstream docs and reduces "Output: 9 token" failures.
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        gen_kw = {
            "max_new_tokens": budget,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
        if pad_id is not None:
            gen_kw["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kw["eos_token_id"] = eos_id

        with self._generate_lock:
            with self._torch.inference_mode():
                outputs = self._model.generate(input_ids, **gen_kw)

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
