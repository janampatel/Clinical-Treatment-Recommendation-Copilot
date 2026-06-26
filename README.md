# Clinical Treatment Recommendation Copilot

A treatment-recommendation system for Type 2 Diabetes and its common comorbidities
(hypertension, hyperlipidemia, chronic kidney disease).

**Core design principle — the ML decides, the LLM only explains.** A deterministic
pipeline ranks the treatment options; a language model is confined to writing the
rationale and can never change the ranking. Every recommendation is backed by a
guideline citation and has passed a rule-based drug-safety filter, so the output
is auditable, not a black box.

## Pipeline

| Stage | What it does |
|-------|--------------|
| 1. Retrieval | Embed the patient and query Qdrant for similar patients + relevant guideline chunks |
| 2. Candidate generation | Assemble drug candidates from guidelines and similar-patient outcomes |
| 3. Safety filter | Remove contraindications, drug-drug interactions, and allergy conflicts |
| 4. Ranking | A LightGBM ranker scores the survivors on 5 clinical features |
| 5. Explanation | Generate a grounded, cited rationale for the top recommendation |

A **grounding guard** (LangGraph) suppresses any recommendation that lacks a
citation or fails the safety check. FastAPI backend, Streamlit UI.

## Real data, real safety

- **Patients** — synthetic but clinically realistic records from
  [Synthea](https://github.com/synthetichealth/synthea), parsed from FHIR into a
  compact cohort (`data/processed/cohort.json`).
- **Drug interactions** — [DDInter 2.0](https://ddinter2.scbdd.com/), a public
  database of ~300k severity-graded interactions, drives the safety filter.
- **Guidelines** — curated excerpts from ADA, JNC8, ACC/AHA, and KDIGO, each with
  a citation.

## Quick start

```powershell
pip install -r requirements.txt
.\run_local.ps1
```

Opens http://localhost:8501. Pick a patient, then drag **eGFR below 30** and watch
metformin get removed by the safety filter — the contraindication is caught and
explained, live.

By default the rationale uses an instant template generator and an in-memory
vector store (no model downloads, nothing external). The clinical embedder
(PubMedBERT) and the fine-tuned LLM are optional upgrades, see below.

## API

```powershell
$env:PYTHONPATH = "src"
uvicorn copilot.api.main:app --port 8000   # docs at http://localhost:8000/docs
```

`POST /recommend` returns the full auditable breakdown: ranked candidates with
per-feature scores, the removed drugs with cited reasons, the similar patients,
the guideline hits, and the rationale.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## Optional upgrades

- **Real clinical embeddings** — `pip install sentence-transformers` and set
  `COPILOT_EMBEDDER=pubmedbert`.
- **Fine-tuned explanation model** — train a LoRA adapter
  (`notebooks/lora_finetune.ipynb`), export a CPU-servable GGUF
  (`notebooks/merge_to_gguf.ipynb`), then set `LLM_MODE=local`. If the model is
  absent, the pipeline falls back to the template generator automatically.

## Deploy (Google Cloud Run)

```powershell
.\deploy\deploy.ps1 -ProjectId <your-project-id>
```

Builds the image in Cloud Build (no local Docker) and deploys the UI and API.

## Layout

```
src/copilot/
  data/         schema, loaders, cohort
  guidelines/   guideline corpus + loader
  retrieval/    embeddings + Qdrant
  candidates/   candidate generation
  safety/       DDInter + rule-based filter
  ranking/      LightGBM ranker + features
  reasoning/    rationale generators
  graph/        LangGraph pipeline + grounding guard
  api/          FastAPI
  app/          Streamlit
```