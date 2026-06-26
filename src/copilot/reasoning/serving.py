from __future__ import annotations

from ..config import (
    LLM_MODE,
    RATIONALE_GGUF_PATH,
    VLLM_BASE_URL,
    VLLM_MODEL,
)
from ..data.schema import PatientRecord
from ..ranking.ranker import RankedCandidate
from ..safety.filter import SafetyResult
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .rationale import TemplateRationaleGenerator, build_factsheet


class LocalGGUFRationaleGenerator:
    def __init__(self, gguf_path=None, n_ctx: int = 2048, n_threads: int | None = None):
        self.gguf_path = str(gguf_path or RATIONALE_GGUF_PATH)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None
        self._fallback = TemplateRationaleGenerator()

    def _ensure(self) -> bool:
        if self._llm is not None:
            return True
        from pathlib import Path

        if not Path(self.gguf_path).exists():
            return False
        try:
            from llama_cpp import Llama
        except ImportError:
            return False
        self._llm = Llama(
            model_path=self.gguf_path, n_ctx=self.n_ctx,
            n_threads=self.n_threads, verbose=False,
        )
        return True

    def generate(self, top: RankedCandidate, patient: PatientRecord,
                 safety: SafetyResult) -> str:
        if not self._ensure():
            return self._fallback.generate(top, patient, safety)
        fs = build_factsheet(top, patient, safety)
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(fs)},
            ],
            max_tokens=200, temperature=0.0,
        )
        return out["choices"][0]["message"]["content"].strip()


class VLLMRationaleGenerator:
    def __init__(self, base_url=None, model=None):
        self.base_url = base_url or VLLM_BASE_URL
        self.model = model or VLLM_MODEL
        self._client = None
        self._fallback = TemplateRationaleGenerator()

    def _ensure(self) -> bool:
        if self._client is not None:
            return True
        try:
            from openai import OpenAI
        except ImportError:
            return False
        self._client = OpenAI(base_url=self.base_url, api_key="not-needed")
        return True

    def generate(self, top: RankedCandidate, patient: PatientRecord,
                 safety: SafetyResult) -> str:
        if not self._ensure():
            return self._fallback.generate(top, patient, safety)
        fs = build_factsheet(top, patient, safety)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(fs)},
                ],
                max_tokens=200, temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self._fallback.generate(top, patient, safety)


def get_rationale_generator(mode: str | None = None):
    mode = (mode or LLM_MODE).lower()
    if mode == "local":
        return LocalGGUFRationaleGenerator()
    if mode == "vllm":
        return VLLMRationaleGenerator()
    return TemplateRationaleGenerator()
