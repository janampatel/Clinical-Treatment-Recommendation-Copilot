from __future__ import annotations

from dataclasses import dataclass, field

from ..data.schema import PatientRecord
from ..retrieval.vector_store import GuidelineHit, PatientHit
from ..safety.filter import DrugOntology


@dataclass
class GuidelineSupport:
    guideline_id: str
    line: str
    source: str
    citation: str


@dataclass
class PatientSupport:
    patient_id: str
    success: float
    similarity: float


@dataclass
class Candidate:
    drug: str
    guideline_support: list[GuidelineSupport] = field(default_factory=list)
    patient_support: list[PatientSupport] = field(default_factory=list)

    @property
    def is_first_line(self) -> bool:
        return any(g.line == "first" for g in self.guideline_support)

    @property
    def citations(self) -> list[str]:
        seen, out = set(), []
        for g in self.guideline_support:
            if g.citation not in seen:
                seen.add(g.citation)
                out.append(g.citation)
        return out

    @property
    def similar_patient_success_rate(self) -> float | None:
        if not self.patient_support:
            return None
        return sum(p.success for p in self.patient_support) / len(self.patient_support)

    @property
    def support_patient_count(self) -> int:
        return len(self.patient_support)


class CandidateGenerator:
    def __init__(self, ontology: DrugOntology | None = None):
        self.ontology = ontology or DrugOntology.load()

    def _guideline_agent(self, patient, guideline_hits):
        candidates: dict[str, Candidate] = {}
        for hit in guideline_hits:
            chunk = hit.chunk
            if chunk.line not in ("first", "second"):
                continue
            if not patient.has_condition(chunk.condition):
                continue
            for drug in chunk.recommends:
                cand = candidates.setdefault(drug, Candidate(drug=drug))
                cand.guideline_support.append(GuidelineSupport(
                    chunk.id, chunk.line, chunk.source, chunk.citation))
        return candidates

    def _retrieval_agent(self, candidates, patient_hits):
        for cand in candidates.values():
            cand_class = self.ontology.class_of(cand.drug)
            for ph in patient_hits:
                for outcome in ph.patient.treatment_history:
                    same_class = (cand_class is not None
                                  and self.ontology.class_of(outcome.drug) == cand_class)
                    if outcome.drug == cand.drug or same_class:
                        cand.patient_support.append(PatientSupport(
                            ph.patient.patient_id, outcome.success, ph.score))
                        break

    def generate(self, patient: PatientRecord, guideline_hits, patient_hits) -> list[Candidate]:
        candidates = self._guideline_agent(patient, guideline_hits)
        self._retrieval_agent(candidates, patient_hits)
        return sorted(candidates.values(),
                      key=lambda c: (c.is_first_line, c.support_patient_count), reverse=True)
