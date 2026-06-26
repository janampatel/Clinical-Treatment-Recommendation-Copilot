from __future__ import annotations

from dataclasses import dataclass, field

from .candidates.generator import Candidate, CandidateGenerator
from .config import TOP_K_GUIDELINES, TOP_K_PATIENTS
from .data.schema import PatientRecord
from .guidelines.loader import load_guidelines
from .ranking.features import FeatureBuilder
from .ranking.ranker import RankedCandidate, TreatmentRanker
from .reasoning.rationale import RationaleGenerator
from .reasoning.serving import get_rationale_generator
from .retrieval.vector_store import GuidelineHit, PatientHit, RetrievalIndex
from .safety.engine import get_safety_engine
from .safety.filter import DrugOntology, SafetyResult


@dataclass
class PipelineResult:
    patient: PatientRecord
    similar_patients: list[PatientHit]
    guideline_hits: list[GuidelineHit]
    candidates_before_safety: list[Candidate]
    safety: SafetyResult
    ranked: list[RankedCandidate]
    top: RankedCandidate | None
    rationale: str | None
    grounded: bool
    messages: list[str] = field(default_factory=list)


class CopilotEngine:
    def __init__(self, index, ranker=None, rationale_generator=None,
                 use_full_guideline_corpus=True):
        self.index = index
        self.ontology = DrugOntology.load()
        self.generator = CandidateGenerator(ontology=self.ontology)
        self.safety = get_safety_engine(ontology=self.ontology)
        self.builder = FeatureBuilder(ontology=self.ontology)
        self.ranker = ranker or TreatmentRanker().load()
        self.rationale_generator = rationale_generator or get_rationale_generator()
        self.use_full_guideline_corpus = use_full_guideline_corpus
        self._all_guidelines = load_guidelines()

    def _guideline_hits(self, patient) -> list[GuidelineHit]:
        if self.use_full_guideline_corpus:
            return [GuidelineHit(chunk=c, score=1.0) for c in self._all_guidelines]
        return self.index.search_guidelines(patient, k=TOP_K_GUIDELINES)

    def recommend(self, patient: PatientRecord, explain: bool = True) -> PipelineResult:
        msgs = []
        similar = self.index.search_patients(patient, k=TOP_K_PATIENTS)
        g_hits = self._guideline_hits(patient)

        candidates = self.generator.generate(patient, g_hits, similar)
        msgs.append(f"Stage 2: {len(candidates)} candidate(s) assembled.")

        safety = self.safety.apply([c.drug for c in candidates], patient)
        kept = [c for c in candidates if c.drug in safety.kept]
        msgs.append(f"Stage 3: {len(safety.removed)} removed, {len(kept)} kept.")

        feats = [self.builder.build(c, patient, safety) for c in kept]
        ranked = self.ranker.rank(kept, feats)
        top = ranked[0] if ranked else None

        grounded = top is not None and bool(top.candidate.citations)
        rationale = None
        if explain and grounded:
            rationale = self.rationale_generator.generate(top, patient, safety)

        return PipelineResult(
            patient=patient, similar_patients=similar, guideline_hits=g_hits,
            candidates_before_safety=candidates, safety=safety, ranked=ranked,
            top=top if grounded else None, rationale=rationale,
            grounded=grounded, messages=msgs)


def build_engine(source=None, embedder=None, embedder_kind=None) -> CopilotEngine:
    from .config import DATA_SOURCE, EMBEDDER_KIND
    from .retrieval.retriever import build_index

    index = build_index(source=source or DATA_SOURCE, embedder=embedder,
                        embedder_kind=embedder_kind or EMBEDDER_KIND)
    return CopilotEngine(index=index)
