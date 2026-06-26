from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from ..data.loader import filter_t2d_cohort, load_patients
from ..data.mock_cohort import DEMO_QUERY_CKD, DEMO_QUERY_PATIENT
from ..data.schema import PatientRecord
from ..engine import build_engine
from ..graph.pipeline import CopilotGraph
from ..presenter import result_to_dict

app = FastAPI(title="Clinical Treatment Recommendation Copilot", version="1.0")


@lru_cache(maxsize=1)
def get_graph() -> CopilotGraph:
    embedder_kind = os.environ.get("COPILOT_EMBEDDER", "auto")
    source = os.environ.get("COPILOT_SOURCE", "auto")
    engine = build_engine(source=source, embedder_kind=embedder_kind)
    return CopilotGraph(engine)


@app.get("/health")
def health() -> dict:
    g = get_graph()
    return {
        "status": "ok",
        "embedder": type(g.engine.index.embedder).__name__,
        "ranker_trained": g.engine.ranker.is_trained,
        "rationale_generator": type(g.engine.rationale_generator).__name__,
    }


@app.get("/demo-patients")
def demo_patients() -> dict:
    cohort = filter_t2d_cohort(load_patients("auto"))
    return {
        "demo": [DEMO_QUERY_PATIENT.model_dump(), DEMO_QUERY_CKD.model_dump()],
        "cohort": [p.model_dump() for p in cohort],
    }


@app.post("/recommend")
def recommend(patient: PatientRecord, explain: bool = True) -> dict:
    result = get_graph().invoke(patient, explain=explain)
    return result_to_dict(result)
