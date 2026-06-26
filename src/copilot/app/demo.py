"""Streamlit demo UI."""
from __future__ import annotations

import sys
from pathlib import Path

# make `copilot` importable when run via `streamlit run src/copilot/app/demo.py`
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from copilot.data.loader import filter_t2d_cohort, load_patients
from copilot.data.mock_cohort import DEMO_QUERY_CKD, DEMO_QUERY_PATIENT
from copilot.data.schema import LabPanel, PatientRecord
from copilot.engine import build_engine
from copilot.graph.pipeline import CopilotGraph
from copilot.presenter import result_to_dict

st.set_page_config(page_title="Clinical Treatment Copilot", layout="wide")

CURATED_MEDS = ["Lisinopril", "Losartan", "Amlodipine", "Hydrochlorothiazide",
                "Metoprolol", "Atorvastatin", "Rosuvastatin", "Metformin",
                "Insulin glargine", "Empagliflozin", "Semaglutide", "Sitagliptin"]
CURATED_ALLERGIES = ["Sulfa", "Penicillin", "Aspirin"]


@st.cache_resource(show_spinner="Building pipeline (retrieval + ranker)…")
def get_graph() -> CopilotGraph:
    # source/embedder come from env (COPILOT_SOURCE / COPILOT_EMBEDDER)
    return CopilotGraph(build_engine())


def _short_label(p: PatientRecord) -> str:
    # collapse repeated CKD stages, keep the label readable
    conds, seen = [], set()
    for c in p.conditions:
        key = "CKD" if "kidney" in c.lower() else c
        if key not in seen:
            seen.add(key)
            conds.append("CKD" if key == "CKD" else c)
    return f"Cohort {p.patient_id[:8]} — {', '.join(conds[:3])}"


def patient_picker() -> PatientRecord:
    st.sidebar.header("Patient")
    presets = {
        "★ Demo: T2D + Hypertension (treatment-naive)": DEMO_QUERY_PATIENT,
        "★ Demo: T2D + HTN + CKD-4 + Sulfa allergy": DEMO_QUERY_CKD,
    }
    for p in filter_t2d_cohort(load_patients("auto")):
        presets[_short_label(p)] = p

    choice = st.sidebar.selectbox("Start from", list(presets.keys()))
    base = presets[choice]

    st.sidebar.caption("Adjust labs to see the safety filter / ranking react:")
    hba1c = st.sidebar.slider("HbA1c (%)", 5.0, 13.0, float(base.labs.hba1c or 8.0), 0.1)
    egfr = st.sidebar.slider("eGFR", 8.0, 120.0, float(base.labs.egfr or 80.0), 1.0)
    potassium = st.sidebar.slider("Potassium (mEq/L)", 3.5, 6.0,
                                  float(base.labs.potassium or 4.3), 0.1)
    sbp = st.sidebar.slider("Systolic BP", 110.0, 180.0, float(base.labs.sbp or 140.0), 1.0)

    # Options must include whatever the patient already has, or Streamlit errors.
    med_options = sorted(set(CURATED_MEDS) | set(base.medications))
    meds = st.sidebar.multiselect("Current medications", med_options,
                                  default=base.medications)
    allergy_options = sorted(set(CURATED_ALLERGIES) | set(base.allergies))
    allergies = st.sidebar.multiselect("Allergies", allergy_options,
                                       default=base.allergies)

    return base.model_copy(update={
        "medications": meds, "allergies": allergies,
        "labs": LabPanel(hba1c=hba1c, egfr=egfr, potassium=potassium, sbp=sbp,
                         dbp=base.labs.dbp, ldl=base.labs.ldl, hdl=base.labs.hdl,
                         creatinine=base.labs.creatinine),
    })


def main():
    st.title("🩺 Clinical Treatment Recommendation Copilot")
    st.caption("Deterministic ranking · LLM explains only · every recommendation "
               "is citation-backed and safety-filtered.")

    graph = get_graph()
    patient = patient_picker()
    data = result_to_dict(graph.invoke(patient))

    # --- patient summary ---
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.subheader("Patient")
        st.write(f"**{patient.age} y/o {patient.sex}** — "
                 f"{', '.join(patient.conditions) or 'no conditions'}")
        st.write(f"**Meds:** {', '.join(patient.medications) or 'none'}")
        st.write(f"**Allergies:** {', '.join(patient.allergies) or 'none'}")
    with c2:
        st.subheader("Key labs")
        labs = data["patient"]["labs"]
        st.table(pd.DataFrame([labs]).T.rename(columns={0: "value"})
                 if labs else pd.DataFrame())
    with c3:
        st.subheader("Grounded?")
        if data["grounded"]:
            st.success("✅ Yes")
        else:
            st.error("❌ Suppressed")

    # --- recommendation ---
    st.divider()
    rec = data["recommendation"]
    if rec:
        st.subheader(f"⭐ Recommended: {rec['drug']}")
        rc1, rc2 = st.columns([3, 2])
        with rc1:
            st.markdown("**Rationale (LLM — explanation only):**")
            st.code(rec["rationale"], language=None)
            st.markdown("**Citations:**")
            for cit in rec["citations"]:
                st.caption(f"• {cit}")
        with rc2:
            st.markdown("**Feature breakdown (drives the ranking):**")
            st.bar_chart(pd.Series(rec["features"]))
    else:
        st.error("No grounded, safe recommendation — pipeline suppressed output.")

    # --- demo-flow detail ---
    st.divider()
    t1, t2, t3, t4 = st.tabs(
        ["1️⃣ Retrieval", "2️⃣–3️⃣ Candidates & Safety", "4️⃣ Ranking", "🔎 Pipeline log"])

    with t1:
        st.markdown("**Top similar patients**")
        sim = data["similar_patients"]
        st.dataframe(pd.DataFrame(sim) if sim else pd.DataFrame(), width="stretch")
        st.markdown("**Retrieved guideline excerpts**")
        for g in data["guideline_hits"]:
            with st.expander(f"{g['source']} — {g['condition']} ({g['line']}-line)"):
                st.write(g["text"])
                st.caption(g["citation"])

    with t2:
        st.markdown("**Candidates assembled (Stage 2):** "
                    + (", ".join(data["candidates_before_safety"]) or "none"))
        st.markdown("**❌ Removed by Drug Safety Filter (Stage 3):**")
        if data["removed"]:
            df = pd.DataFrame(data["removed"])[["drug", "severity", "reason", "citation"]]
            st.dataframe(df, width="stretch")
        else:
            st.info("Nothing removed for this patient.")

    with t3:
        rows = []
        for r in data["ranked"]:
            row = {"drug": r["drug"], "score": r["score"], "first_line": r["is_first_line"]}
            row.update(r["features"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(), width="stretch")

    with t4:
        for m in data["log"]:
            st.text(m)


if __name__ == "__main__":
    main()
