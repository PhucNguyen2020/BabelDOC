"""
Tencent Hunyuan HY-MT translation prompts and OpenAI-compatible client tuning.

Use this module with a remote **OpenAI-compatible** server (``/v1/chat/completions``),
e.g. **vLLM** or **SGLang** serving ``tencent/HY-MT1.5-*`` from Hugging Face. CLI:
``--hunyuan-mt``; set ``HY_MT_OPENAI_BASE_URL`` / ``--openai-base-url`` and
``HY_MT_MODEL`` / ``--openai-model`` to match the server's ``--served-model-name``.

For **in-process** Hugging Face ``transformers`` (no HTTP), use
``--hunyuan-transformers`` instead.

Official prompt templates:
https://github.com/Tencent-Hunyuan/HY-MT
"""

from __future__ import annotations

import logging

import openai
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from babeldoc.babeldoc_exception.BabelDOCException import ContentFilterError
from babeldoc.translator.translator import OpenAITranslator

logger = logging.getLogger(__name__)

# HY-MT language table: normalized code -> (Chinese name, English name)
_HY_LANG_NAMES: dict[str, tuple[str, str]] = {
    "zh": ("中文", "Chinese"),
    "en": ("英语", "English"),
    "fr": ("法语", "French"),
    "pt": ("葡萄牙语", "Portuguese"),
    "es": ("西班牙语", "Spanish"),
    "ja": ("日语", "Japanese"),
    "tr": ("土耳其语", "Turkish"),
    "ru": ("俄语", "Russian"),
    "ar": ("阿拉伯语", "Arabic"),
    "ko": ("韩语", "Korean"),
    "th": ("泰语", "Thai"),
    "it": ("意大利语", "Italian"),
    "de": ("德语", "German"),
    "vi": ("越南语", "Vietnamese"),
    "ms": ("马来语", "Malay"),
    "id": ("印尼语", "Indonesian"),
    "tl": ("菲律宾语", "Filipino"),
    "hi": ("印地语", "Hindi"),
    "zh-hant": ("繁体中文", "Traditional Chinese"),
    "pl": ("波兰语", "Polish"),
    "cs": ("捷克语", "Czech"),
    "nl": ("荷兰语", "Dutch"),
    "km": ("高棉语", "Khmer"),
    "my": ("缅甸语", "Burmese"),
    "fa": ("波斯语", "Persian"),
    "gu": ("古吉拉特语", "Gujarati"),
    "ur": ("乌尔都语", "Urdu"),
    "te": ("泰卢固语", "Telugu"),
    "mr": ("马拉地语", "Marathi"),
    "he": ("希伯来语", "Hebrew"),
    "bn": ("孟加拉语", "Bengali"),
    "ta": ("泰米尔语", "Tamil"),
    "uk": ("乌克兰语", "Ukrainian"),
    "bo": ("藏语", "Tibetan"),
    "kk": ("哈萨克语", "Kazakh"),
    "mn": ("蒙古语", "Mongolian"),
    "ug": ("维吾尔语", "Uyghur"),
    "yue": ("粤语", "Cantonese"),
}


def _normalize_hy_lang(code: str) -> str:
    s = code.strip().lower().replace("_", "-")
    if s in ("zh-cn", "zh-hans", "cmn"):
        return "zh"
    if s in ("zh-tw", "zh-hk"):
        return "zh-hant"
    return s


def _target_display_names(lang_out: str) -> tuple[str, str]:
    key = _normalize_hy_lang(lang_out)
    if key in _HY_LANG_NAMES:
        return _HY_LANG_NAMES[key]
    # Fallback: use raw code for English template; repeat for Chinese template
    return (lang_out, lang_out)


def _pair_involves_chinese(lang_in: str, lang_out: str) -> bool:
    for c in (_normalize_hy_lang(lang_in), _normalize_hy_lang(lang_out)):
        if c in ("zh", "zh-hant", "yue"):
            return True
    return False


def build_hunyuan_user_prompt(lang_in: str, lang_out: str, source_text: str) -> str:
    """
    Build the user message body per Tencent HY-MT official templates.
    ZH<=>XX uses the Chinese template; other pairs use the English template.
    """
    target_zh, target_en = _target_display_names(lang_out)
    if _pair_involves_chinese(lang_in, lang_out):
        return (
            f"将以下文本翻译为{target_zh}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{source_text}"
        )
    return (
        f"Translate the following segment into {target_en}, "
        f"without additional explanation.\n\n"
        f"{source_text}"
    )


class HunyuanMTTranslator(OpenAITranslator):
    """
    OpenAI-compatible backend with HY-MT prompt wording and recommended sampling.

    Recommended deployment: vLLM / SGLang with a HF HY-MT checkpoint; use
    --served-model-name (e.g. hunyuan) as --openai-model.
    """

    name = "hunyuan_mt"

    def __init__(
        self,
        *args,
        send_temperature: bool = True,
        **kwargs,
    ):
        super().__init__(*args, send_temperature=False, **kwargs)
        self.send_temperature = send_temperature
        hy_extra = {"top_k": 20, "repetition_penalty": 1.05}
        self.extra_body = {**hy_extra, **self.extra_body}
        if send_temperature:
            self.options = {"temperature": 0.7, "top_p": 0.6}
            self.add_cache_impact_parameters("temperature", 0.7)
            self.add_cache_impact_parameters("top_p", 0.6)
        else:
            self.options = {}
        self.add_cache_impact_parameters("hunyuan_mt_preset", True)

    def prompt(self, text):
        # HY-MT: no separate system prompt; single user turn (see upstream README).
        return [
            {
                "role": "user",
                "content": build_hunyuan_user_prompt(self.lang_in, self.lang_out, text),
            },
        ]

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        IL/JSON paths send fully crafted user prompts; keep content unchanged,
        only apply HY-MT sampling and extra_body.
        """
        if text is None:
            return None

        options = {}
        if self.send_temperature:
            options.update(self.options)
        if self.enable_json_mode_if_requested and rate_limit_params.get(
            "request_json_mode", False
        ):
            options["response_format"] = {"type": "json_object"}

        extra_headers = {}
        if self.send_dashscope_header:
            extra_headers["X-DashScope-DataInspection"] = (
                '{"input": "disable", "output": "disable"}'
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                **options,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                extra_headers=extra_headers,
                extra_body=self.extra_body,
            )
            self.update_token_count(response)
            return response.choices[0].message.content.strip()
        except openai.BadRequestError as e:
            if (
                "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"
                in e.message
            ):
                raise ContentFilterError(e.message) from e
            raise
