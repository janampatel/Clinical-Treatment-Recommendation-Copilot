from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from ..config import (
    GUIDELINE_COLLECTION,
    PATIENT_COLLECTION,
    TOP_K_GUIDELINES,
    TOP_K_PATIENTS,
)
from ..data.schema import PatientRecord
from ..guidelines.loader import GuidelineChunk
from .embeddings import get_embedder


@dataclass
class PatientHit:
    patient: PatientRecord
    score: float


@dataclass
class GuidelineHit:
    chunk: GuidelineChunk
    score: float


class RetrievalIndex:
    def __init__(self, embedder=None, location=":memory:", url=None, api_key=None):
        self.embedder = embedder or get_embedder("auto")
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        else:
            self.client = QdrantClient(location=location)
        self._patients_by_id: dict[str, PatientRecord] = {}

    def _recreate(self, name, dim):
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def build(self, patients, guidelines):
        dim = self.embedder.dim
        self._recreate(PATIENT_COLLECTION, dim)
        self._recreate(GUIDELINE_COLLECTION, dim)

        self._patients_by_id = {p.patient_id: p for p in patients}
        p_vectors = self.embedder.encode([p.to_clinical_text() for p in patients])
        self.client.upsert(
            collection_name=PATIENT_COLLECTION,
            points=[models.PointStruct(id=i, vector=v, payload=p.model_dump())
                    for i, (p, v) in enumerate(zip(patients, p_vectors))])

        g_vectors = self.embedder.encode([g.embedding_text() for g in guidelines])
        self.client.upsert(
            collection_name=GUIDELINE_COLLECTION,
            points=[models.PointStruct(id=i, vector=v, payload=g.model_dump())
                    for i, (g, v) in enumerate(zip(guidelines, g_vectors))])

    def search_patients(self, query, k=TOP_K_PATIENTS) -> list[PatientHit]:
        qv = self.embedder.encode([query.to_clinical_text()])[0]
        res = self.client.query_points(
            collection_name=PATIENT_COLLECTION, query=qv, limit=k + 1).points
        hits = []
        for r in res:
            rec = PatientRecord(**r.payload)
            if rec.patient_id == query.patient_id:
                continue
            hits.append(PatientHit(patient=rec, score=r.score))
            if len(hits) >= k:
                break
        return hits

    def search_guidelines(self, query, k=TOP_K_GUIDELINES) -> list[GuidelineHit]:
        qv = self.embedder.encode([query.to_clinical_text()])[0]
        res = self.client.query_points(
            collection_name=GUIDELINE_COLLECTION, query=qv, limit=k).points
        return [GuidelineHit(chunk=GuidelineChunk(**r.payload), score=r.score) for r in res]
