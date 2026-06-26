from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..candidates.generator import Candidate, CandidateGenerator
from ..data.loader import filter_t2d_cohort, load_patients
from ..guidelines.loader import load_guidelines
from ..retrieval.embeddings import Embedder, HashEmbedder
from ..retrieval.vector_store import GuidelineHit, RetrievalIndex
from ..safety.filter import DrugOntology, SafetyFilter
from .features import FEATURE_NAMES, _COMORBIDITY_PREF, FeatureBuilder


@dataclass
class TrainingData:
    X: np.ndarray
    y: list[int]
    groups: list[int]
    feature_names: list[str]


def gold_label(patient, candidate: Candidate, ontology: DrugOntology) -> int:
    label = 0
    if candidate.is_first_line:
        label += 2
    elif candidate.guideline_support:
        label += 1

    cand_class = ontology.class_of(candidate.drug)
    best_self = None
    for o in patient.treatment_history:
        if o.drug == candidate.drug or ontology.class_of(o.drug) == cand_class:
            best_self = o.success if best_self is None else max(best_self, o.success)
    if best_self is not None:
        if best_self >= 0.85:
            label += 2
        elif best_self >= 0.70:
            label += 1

    if any(patient.has_condition(c) for c in _COMORBIDITY_PREF.get(cand_class, set())):
        label += 1
    return min(label, 4)


def build_training_data(source: str = "mock", embedder: Embedder | None = None) -> TrainingData:
    embedder = embedder or HashEmbedder()
    cohort = filter_t2d_cohort(load_patients(source))
    guidelines = load_guidelines()

    index = RetrievalIndex(embedder=embedder)
    index.build(cohort, guidelines)

    ontology = DrugOntology.load()
    generator = CandidateGenerator(ontology=ontology)
    safety = SafetyFilter(ontology=ontology)
    builder = FeatureBuilder(ontology=ontology)
    all_g_hits = [GuidelineHit(chunk=c, score=1.0) for c in guidelines]

    rows, labels, groups = [], [], []
    for patient in cohort:
        p_hits = index.search_patients(patient, k=5)
        candidates = generator.generate(patient, all_g_hits, p_hits)
        if not candidates:
            continue
        safety_res = safety.apply([c.drug for c in candidates], patient)
        kept = [c for c in candidates if c.drug in safety_res.kept]
        if len(kept) < 2:
            continue
        for cand in kept:
            rows.append(builder.build(cand, patient, safety_res).values)
            labels.append(gold_label(patient, cand, ontology))
        groups.append(len(kept))

    return TrainingData(np.array(rows, dtype=float), labels, groups, list(FEATURE_NAMES))
