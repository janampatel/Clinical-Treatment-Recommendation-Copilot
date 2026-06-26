from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from ..config import DDINTER_DIR

SEVERITY_RANK = {"Minor": 1, "Moderate": 2, "Major": 3}
CITATION = "DDInter 2.0 (Nucleic Acids Research 2025), ddinter2.scbdd.com"

_NOISE = re.compile(
    r"\b(\d+(\.\d+)?\s*(mg|mcg|g|ml|unit|units|%)|oral|tablet|capsule|"
    r"injection|solution|suspension|extended|release|hr|er|xr|sodium|"
    r"hydrochloride|succinate|tartrate|calcium|potassium|maleate)\b", re.IGNORECASE)


def _clean(name: str) -> str:
    name = _NOISE.sub(" ", name)
    name = re.sub(r"[^a-z ]", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


class DDInterDB:
    def __init__(self):
        self._pairs: dict[frozenset, str] = {}
        self._names: dict[str, str] = {}
        self._norm_cache: dict[str, str | None] = {}

    def load(self, directory: Path | None = None) -> "DDInterDB":
        directory = Path(directory or DDINTER_DIR)
        for path in sorted(directory.glob("ddinter_*.csv")):
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    a, b, level = row.get("Drug_A"), row.get("Drug_B"), row.get("Level")
                    if not a or not b or level not in SEVERITY_RANK:
                        continue
                    al, bl = a.strip().lower(), b.strip().lower()
                    self._names.setdefault(al, a.strip())
                    self._names.setdefault(bl, b.strip())
                    key = frozenset((al, bl))
                    prev = self._pairs.get(key)
                    if prev is None or SEVERITY_RANK[level] > SEVERITY_RANK[prev]:
                        self._pairs[key] = level
        return self

    @property
    def n_interactions(self) -> int:
        return len(self._pairs)

    @property
    def n_drugs(self) -> int:
        return len(self._names)

    def normalize(self, drug: str) -> str | None:
        if not drug:
            return None
        if drug in self._norm_cache:
            return self._norm_cache[drug]
        result = self._normalize(drug)
        self._norm_cache[drug] = result
        return result

    def _normalize(self, drug: str) -> str | None:
        if drug.strip().lower() in self._names:
            return self._names[drug.strip().lower()]
        cleaned = _clean(drug)
        if cleaned in self._names:
            return self._names[cleaned]
        best = None
        for name_l in self._names:
            if len(name_l) < 4:
                continue
            if re.search(rf"\b{re.escape(name_l)}\b", cleaned):
                if best is None or len(name_l) > len(best):
                    best = name_l
        return self._names[best] if best else None

    def interaction(self, drug_a: str, drug_b: str) -> str | None:
        na, nb = self.normalize(drug_a), self.normalize(drug_b)
        if not na or not nb or na.lower() == nb.lower():
            return None
        return self._pairs.get(frozenset((na.lower(), nb.lower())))


@lru_cache(maxsize=1)
def get_ddinter() -> DDInterDB:
    return DDInterDB().load()
