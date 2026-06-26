"""LangGraph orchestration: retrieve -> candidates -> safety -> rank ->
grounding guard -> explain/suppress."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..data.schema import PatientRecord
from ..engine import CopilotEngine, PipelineResult


class GraphState(TypedDict, total=False):
    patient: PatientRecord
    similar: list
    guideline_hits: list
    candidates: list
    safety: Any
    ranked: list
    top: Any
    grounded: bool
    rationale: str | None
    explain: bool
    log: Annotated[list[str], operator.add]


def build_graph(engine: CopilotEngine):

    def retrieve(state: GraphState) -> dict:
        patient = state["patient"]
        similar = engine.index.search_patients(patient)
        g_hits = engine._guideline_hits(patient)
        return {
            "similar": similar,
            "guideline_hits": g_hits,
            "log": [f"Stage 1: {len(similar)} similar patients, {len(g_hits)} guideline chunks."],
        }

    def generate_candidates(state: GraphState) -> dict:
        cands = engine.generator.generate(
            state["patient"], state["guideline_hits"], state["similar"]
        )
        return {"candidates": cands, "log": [f"Stage 2: {len(cands)} candidates."]}

    def safety_filter(state: GraphState) -> dict:
        cands = state["candidates"]
        res = engine.safety.apply([c.drug for c in cands], state["patient"])
        return {"safety": res, "log": [f"Stage 3: {len(res.removed)} removed, {len(res.kept)} kept."]}

    def rank(state: GraphState) -> dict:
        patient, safety = state["patient"], state["safety"]
        kept = [c for c in state["candidates"] if c.drug in safety.kept]
        feats = [engine.builder.build(c, patient, safety) for c in kept]
        ranked = engine.ranker.rank(kept, feats)
        top = ranked[0] if ranked else None
        grounded = top is not None and bool(top.candidate.citations)
        return {
            "ranked": ranked, "top": top, "grounded": grounded,
            "log": [f"Stage 4: ranked {len(ranked)}; grounded={grounded}."],
        }

    def grounding_guard(state: GraphState) -> str:
        return "explain" if state.get("grounded") else "suppress"

    def explain(state: GraphState) -> dict:
        if not state.get("explain", True):
            return {"rationale": None}
        text = engine.rationale_generator.generate(
            state["top"], state["patient"], state["safety"]
        )
        return {"rationale": text, "log": ["Stage 5: rationale generated (explain-only)."]}

    def suppress(state: GraphState) -> dict:
        return {
            "top": None, "rationale": None,
            "log": ["Grounding guard: no citation/safe candidate — recommendation suppressed."],
        }

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate_candidates", generate_candidates)
    g.add_node("safety_filter", safety_filter)
    g.add_node("rank", rank)
    g.add_node("explain", explain)
    g.add_node("suppress", suppress)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate_candidates")
    g.add_edge("generate_candidates", "safety_filter")
    g.add_edge("safety_filter", "rank")
    g.add_conditional_edges("rank", grounding_guard,
                            {"explain": "explain", "suppress": "suppress"})
    g.add_edge("explain", END)
    g.add_edge("suppress", END)
    return g.compile()


class CopilotGraph:
    def __init__(self, engine: CopilotEngine):
        self.engine = engine
        self.graph = build_graph(engine)

    def invoke(self, patient: PatientRecord, explain: bool = True) -> PipelineResult:
        final = self.graph.invoke({"patient": patient, "explain": explain, "log": []})
        return PipelineResult(
            patient=patient,
            similar_patients=final.get("similar", []),
            guideline_hits=final.get("guideline_hits", []),
            candidates_before_safety=final.get("candidates", []),
            safety=final["safety"],
            ranked=final.get("ranked", []),
            top=final.get("top"),
            rationale=final.get("rationale"),
            grounded=final.get("grounded", False),
            messages=final.get("log", []),
        )
