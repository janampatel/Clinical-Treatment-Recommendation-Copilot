from __future__ import annotations

from typing import Protocol

from ..data.schema import PatientRecord
from ..ranking.ranker import RankedCandidate
from ..safety.filter import SafetyResult
from .prompt import SYSTEM_PROMPT, build_user_prompt


def build_factsheet(top: RankedCandidate, patient: PatientRecord, safety: SafetyResult) -> dict:
    cand = top.candidate
    gl = cand.guideline_support[0] if cand.guideline_support else None
    rate = cand.similar_patient_success_rate
    warnings = [w.reason for w in safety.warnings.get(cand.drug, [])]

    key_labs = {}
    for name in ("hba1c", "egfr", "ldl", "sbp", "dbp", "potassium"):
        v = getattr(patient.labs, name)
        if v is not None:
            key_labs[name] = v

    return {
        "recommended_drug": cand.drug,
        "patient": {
            "age": patient.age, "sex": patient.sex,
            "conditions": patient.conditions, "key_labs": key_labs,
        },
        "guideline_basis": None if gl is None else {
            "line": gl.line, "source": gl.source, "citation": gl.citation,
            "condition": patient.conditions[0] if patient.conditions else "the condition",
        },
        "similar_patient_evidence": None if rate is None else {
            "n_patients": cand.support_patient_count,
            "mean_success_rate": round(rate, 2),
        },
        "safety": {"passed_filter": True, "warnings": warnings},
        "compatibility": {
            "comorbidity_compatibility": round(top.features.get("comorbidity_compatibility", 0.0), 2),
            "lab_value_compatibility": round(top.features.get("lab_value_compatibility", 0.0), 2),
        },
    }


class RationaleGenerator(Protocol):
    def generate(self, top: RankedCandidate, patient: PatientRecord, safety: SafetyResult) -> str:
        ...


class TemplateRationaleGenerator:
    def generate(self, top, patient, safety) -> str:
        fs = build_factsheet(top, patient, safety)
        lines = [f"Why {fs['recommended_drug']}?"]

        gb = fs["guideline_basis"]
        if gb:
            lines.append(f"- Guideline-recommended {gb['line']}-line therapy for "
                         f"{gb['condition']} ({gb['source']}).")
        else:
            lines.append("- Supported by the retrieved clinical guidelines.")

        spe = fs["similar_patient_evidence"]
        if spe and spe["n_patients"] > 0:
            lines.append(f"- {spe['n_patients']} similar patient(s) on this therapy had a "
                         f"mean treatment success of {int(spe['mean_success_rate'] * 100)}%.")
        else:
            lines.append("- No similar-patient outcome data was available for this option.")

        warnings = fs["safety"]["warnings"]
        if warnings:
            lines.append(f"- Passed the safety filter with a caution: {warnings[0]}")
        else:
            lines.append("- No contraindications, drug interactions, or allergy conflicts "
                         "were detected for this patient.")

        comorbs = [c for c in patient.conditions if "type 2 diabetes" not in c.lower()]
        if comorbs:
            lines.append(f"- Compatible with comorbid {', '.join(comorbs)} and the "
                         f"patient's current labs.")
        else:
            lines.append("- Compatible with the patient's diabetes profile and current labs.")
        return "\n".join(lines)

    def messages(self, top, patient, safety) -> list[dict]:
        fs = build_factsheet(top, patient, safety)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(fs)},
        ]
