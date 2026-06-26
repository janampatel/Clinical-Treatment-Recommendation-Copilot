from __future__ import annotations

import json

from ..config import PROCESSED_COHORT_PATH, SYNTHEA_DIR
from .mock_cohort import load_mock_cohort
from .schema import PatientRecord
from .synthea_loader import load_synthea_patients


def load_processed_cohort() -> list[PatientRecord]:
    raw = json.loads(PROCESSED_COHORT_PATH.read_text(encoding="utf-8"))
    return [PatientRecord(**rec) for rec in raw]


def save_processed_cohort(patients: list[PatientRecord]) -> None:
    PROCESSED_COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_COHORT_PATH.write_text(
        json.dumps([p.model_dump() for p in patients], indent=2), encoding="utf-8")


def load_patients(source: str = "auto") -> list[PatientRecord]:
    if source == "mock":
        return load_mock_cohort()
    if source == "synthea":
        return load_synthea_patients()
    if source == "processed":
        return load_processed_cohort()
    if source == "auto":
        if PROCESSED_COHORT_PATH.exists():
            return load_processed_cohort()
        if any(SYNTHEA_DIR.glob("*.json")):
            return load_synthea_patients()
        return load_mock_cohort()
    raise ValueError(f"unknown source: {source!r}")


def filter_t2d_cohort(patients, require_labs=False):
    cohort = [p for p in patients if p.has_condition("Type 2 Diabetes")]
    if require_labs:
        cohort = [p for p in cohort if p.labs.hba1c is not None]
    return cohort
