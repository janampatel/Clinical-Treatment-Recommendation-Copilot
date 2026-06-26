from __future__ import annotations

from dataclasses import dataclass

from ..candidates.generator import Candidate
from ..data.schema import PatientRecord
from ..safety.filter import DrugOntology, SafetyResult

FEATURE_NAMES = [
    "guideline_relevance",
    "similar_patient_success",
    "interaction_safety_score",
    "comorbidity_compatibility",
    "lab_value_compatibility",
]

_COMORBIDITY_PREF = {
    "sglt2": {"Chronic Kidney Disease", "Hypertension"},
    "glp1": set(),
    "acei": {"Chronic Kidney Disease", "Hypertension"},
    "arb": {"Chronic Kidney Disease", "Hypertension"},
    "ccb": {"Hypertension"},
    "thiazide": {"Hypertension"},
    "statin": {"Hyperlipidemia"},
    "ns_mra": {"Chronic Kidney Disease"},
}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class FeatureVector:
    values: list[float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


class FeatureBuilder:
    def __init__(self, ontology: DrugOntology | None = None):
        self.ontology = ontology or DrugOntology.load()

    def guideline_relevance(self, c: Candidate) -> float:
        base = 0.7 if c.is_first_line else (0.4 if c.guideline_support else 0.0)
        return _clamp(base + min(0.3, 0.1 * len(c.guideline_support)))

    def similar_patient_success(self, c: Candidate) -> float:
        rate = c.similar_patient_success_rate
        return _clamp(rate) if rate is not None else 0.0

    def interaction_safety_score(self, drug: str, safety: SafetyResult) -> float:
        return 1.0 / (1.0 + safety.flag_count(drug))

    def comorbidity_compatibility(self, drug: str, patient: PatientRecord) -> float:
        pref = _COMORBIDITY_PREF.get(self.ontology.class_of(drug), set())
        score = 0.5
        for comorb in pref:
            if patient.has_condition(comorb):
                score += 0.25
        return _clamp(score)

    def lab_value_compatibility(self, drug: str, patient: PatientRecord) -> float:
        cls = self.ontology.class_of(drug)
        labs = patient.labs
        score = 0.7
        if cls == "biguanide" and labs.egfr is not None:
            score = _clamp((labs.egfr - 30) / 30.0)
        if cls == "insulin" and labs.hba1c is not None and labs.hba1c >= 10:
            score += 0.3
        if cls == "glp1" and labs.hba1c is not None and labs.hba1c >= 9:
            score += 0.2
        if cls == "sglt2" and labs.egfr is not None:
            score = _clamp(0.5 + (labs.egfr - 20) / 80.0)
        return _clamp(score)

    def build(self, candidate: Candidate, patient: PatientRecord,
              safety: SafetyResult) -> FeatureVector:
        return FeatureVector([
            self.guideline_relevance(candidate),
            self.similar_patient_success(candidate),
            self.interaction_safety_score(candidate.drug, safety),
            self.comorbidity_compatibility(candidate.drug, patient),
            self.lab_value_compatibility(candidate.drug, patient),
        ])
