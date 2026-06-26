from __future__ import annotations

from .engine import PipelineResult


def _candidate_row(ranked) -> dict:
    c = ranked.candidate
    return {
        "drug": c.drug,
        "score": round(ranked.score, 4),
        "is_first_line": c.is_first_line,
        "features": {k: round(v, 3) for k, v in ranked.features.items()},
        "guideline_sources": [
            {"id": g.guideline_id, "line": g.line, "source": g.source,
             "citation": g.citation}
            for g in c.guideline_support
        ],
        "similar_patient_success_rate": (
            round(c.similar_patient_success_rate, 3)
            if c.similar_patient_success_rate is not None else None
        ),
        "support_patient_count": c.support_patient_count,
        "citations": c.citations,
    }


def result_to_dict(result: PipelineResult) -> dict:
    rec = None
    if result.top is not None:
        rec = {
            "drug": result.top.candidate.drug,
            "score": round(result.top.score, 4),
            "features": {k: round(v, 3) for k, v in result.top.features.items()},
            "citations": result.top.candidate.citations,
            "rationale": result.rationale,
        }

    return {
        "patient": {
            "patient_id": result.patient.patient_id,
            "age": result.patient.age,
            "sex": result.patient.sex,
            "conditions": result.patient.conditions,
            "medications": result.patient.medications,
            "allergies": result.patient.allergies,
            "labs": result.patient.labs.model_dump(exclude_none=True),
        },
        "grounded": result.grounded,
        "recommendation": rec,
        "ranked": [_candidate_row(r) for r in result.ranked],
        "removed": [
            {"drug": f.drug, "rule_id": f.rule_id, "severity": f.severity,
             "reason": f.reason, "citation": f.citation}
            for f in result.safety.removed
        ],
        "candidates_before_safety": [c.drug for c in result.candidates_before_safety],
        "similar_patients": [
            {"patient_id": h.patient.patient_id, "similarity": round(h.score, 4),
             "age": h.patient.age, "sex": h.patient.sex,
             "conditions": h.patient.conditions}
            for h in result.similar_patients
        ],
        "guideline_hits": [
            {"id": h.chunk.id, "source": h.chunk.source, "condition": h.chunk.condition,
             "line": h.chunk.line, "citation": h.chunk.citation, "text": h.chunk.text}
            for h in result.guideline_hits[:6]
        ],
        "log": result.messages,
    }
