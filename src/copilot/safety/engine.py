from __future__ import annotations

import json

from ..config import DDINTER_DIR, INTERACTION_RULES_PATH
from ..data.schema import PatientRecord
from .ddinter import CITATION as DDINTER_CITATION
from .ddinter import DDInterDB, get_ddinter
from .filter import DrugOntology, Flag, SafetyFilter, SafetyResult


class ClinicalSafetyEngine:
    """Contraindication rules (renal/potassium/allergy/duplicate) + DDInter
    drug-drug interactions (Major -> remove, Moderate -> warn)."""

    def __init__(self, ddinter: DDInterDB | None = None, ontology: DrugOntology | None = None):
        self.ontology = ontology or DrugOntology.load()
        self.ddinter = ddinter if ddinter is not None else get_ddinter()
        rules = json.loads(INTERACTION_RULES_PATH.read_text(encoding="utf-8"))
        contraindications = [r for r in rules if r["type"] != "avoid_combo"]
        self._contra = SafetyFilter(rules=contraindications, ontology=self.ontology)

    def _ddinter_flags(self, drug, patient):
        hard, soft = None, []
        for med in patient.medications:
            sev = self.ddinter.interaction(drug, med)
            if sev is None:
                continue
            norm = self.ddinter.normalize(med) or med
            flag = Flag(drug, f"DDINTER-{sev.upper()}",
                        "hard" if sev == "Major" else "soft",
                        f"{sev} drug-drug interaction with current medication {norm}.",
                        DDINTER_CITATION)
            if sev == "Major" and hard is None:
                hard = flag
            elif sev == "Moderate":
                soft.append(flag)
        return hard, soft

    def apply(self, candidates: list[str], patient: PatientRecord) -> SafetyResult:
        base = self._contra.apply(candidates, patient)
        removed = list(base.removed)
        kept = []
        warnings = {k: list(v) for k, v in base.warnings.items()}
        for drug in base.kept:
            hard, soft = self._ddinter_flags(drug, patient)
            if hard is not None:
                removed.append(hard)
                warnings.pop(drug, None)
            else:
                kept.append(drug)
                if soft:
                    warnings.setdefault(drug, []).extend(soft)
        return SafetyResult(kept=kept, removed=removed, warnings=warnings)


def get_safety_engine(ontology: DrugOntology | None = None):
    if any(DDINTER_DIR.glob("ddinter_*.csv")):
        return ClinicalSafetyEngine(ontology=ontology)
    return SafetyFilter(ontology=ontology)
