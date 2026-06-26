from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..config import GUIDELINES_PATH


class GuidelineChunk(BaseModel):
    id: str
    source: str
    year: int
    condition: str
    line: str
    recommends: list[str] = Field(default_factory=list)
    text: str
    citation: str

    def embedding_text(self) -> str:
        return f"[{self.condition} | {self.line}-line] {self.text}"


def load_guidelines() -> list[GuidelineChunk]:
    raw = json.loads(GUIDELINES_PATH.read_text(encoding="utf-8"))
    return [GuidelineChunk(**item) for item in raw]
