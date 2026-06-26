from __future__ import annotations

import os

from ..config import QDRANT_API_KEY, QDRANT_URL
from ..data.loader import filter_t2d_cohort, load_patients
from ..guidelines.loader import load_guidelines
from .embeddings import get_embedder
from .vector_store import RetrievalIndex


def build_index(source="auto", embedder_kind="auto", embedder=None,
                location=":memory:", use_cloud=True) -> RetrievalIndex:
    patients = filter_t2d_cohort(load_patients(source))
    guidelines = load_guidelines()
    force_memory = os.environ.get("COPILOT_VECTOR_STORE", "").lower() == "memory"
    url = QDRANT_URL if (use_cloud and QDRANT_URL and not force_memory) else None
    idx = RetrievalIndex(
        embedder=embedder or get_embedder(embedder_kind),
        location=location, url=url, api_key=QDRANT_API_KEY if url else None)
    idx.build(patients, guidelines)
    return idx
