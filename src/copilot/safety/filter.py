from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import INTERACTION_RULES_PATH
from ..data.schema import PatientRecord

_DRUG_CLASSES_PATH = Path(__file__).resolve().parent / "drug_classes.json"


@dataclass
class Flag:
    drug: str
    rule_id: str
    severity: str
    reason: str
    citation: str


@dataclass
class SafetyResult:
    kept: list[str]
    removed: list[Flag]
    warnings: dict[str, list[Flag]] = field(default_factory=dict)

    def flag_count(self, drug: str) -> int:
        return len(self.warnings.get(drug, []))

    def removed_drugs(self) -> list[str]:
        return [f.drug for f in self.removed]


class DrugOntology:
    def __init__(self, mapping: dict[str, str]):
        self._exact = mapping
        self._lower = {k.lower(): v for k, v in mapping.items()}

    @classmethod
    def load(cls) -> "DrugOntology":
        return cls(json.loads(_DRUG_CLASSES_PATH.read_text(encoding="utf-8")))

    def class_of(self, drug: str) -> str | None:
        if drug in self._exact:
            return self._exact[drug]
        low = drug.lower()
        if low in self._lower:
            return self._lower[low]
        for name, cls in self._lower.items():
            if name in low:
                return cls
        return None


class SafetyFilter:
    def __init__(self, rules=None, ontology=None):
        self.rules = rules if rules is not None else json.loads(
            INTERACTION_RULES_PATH.read_text(encoding="utf-8"))
        self.ontology = ontology or DrugOntology.load()

    def _targets(self, rule, drug):
        if "target_drugs" in rule and drug in rule["target_drugs"]:
            return True
        if "target_class" in rule and self.ontology.class_of(drug) == rule["target_class"]:
            return True
        return False

    def _patient_classes(self, patient):
        classes = set()
        for med in patient.medications:
            c = self.ontology.class_of(med)
            if c:
                classes.add(c)
        return classes

    def _rule_fires(self, rule, drug, patient):
        t = rule["type"]
        labs = patient.labs
        if t == "egfr_min":
            return labs.egfr is not None and labs.egfr < rule["min_egfr"]
        if t == "potassium_max":
            return labs.potassium is not None and labs.potassium > rule["max_potassium"]
        if t == "allergy":
            terms = [a.lower() for a in rule["allergy_terms"]]
            allergies = " ".join(patient.allergies).lower()
            return any(term in allergies for term in terms)
        if t == "avoid_combo":
            return rule["conflicts_with_class"] in self._patient_classes(patient)
        if t == "duplicate_class":
            dc = self.ontology.class_of(drug)
            return dc is not None and dc in self._patient_classes(patient)
        return False

    def _applies_to_drug(self, rule, drug):
        if rule["type"] == "duplicate_class":
            return True
        return self._targets(rule, drug)

    def apply(self, candidates: list[str], patient: PatientRecord) -> SafetyResult:
        kept, removed, warnings = [], [], {}
        for drug in candidates:
            hard_flag = None
            soft_flags = []
            for rule in self.rules:
                if not self._applies_to_drug(rule, drug):
                    continue
                if not self._rule_fires(rule, drug, patient):
                    continue
                flag = Flag(drug, rule["id"], rule["severity"], rule["reason"], rule["citation"])
                if rule["severity"] == "hard" and hard_flag is None:
                    hard_flag = flag
                elif rule["severity"] == "soft":
                    soft_flags.append(flag)
            if hard_flag is not None:
                removed.append(hard_flag)
            else:
                kept.append(drug)
                if soft_flags:
                    warnings[drug] = soft_flags
        return SafetyResult(kept=kept, removed=removed, warnings=warnings)
