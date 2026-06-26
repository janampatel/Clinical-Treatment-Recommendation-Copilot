# One-command local demo — guaranteed, no external dependencies.
#   .\run_local.ps1
# Opens the Streamlit app at http://localhost:8501
#
# Uses: real Synthea cohort (107 patients) + real DDInter safety DB + trained
# LightGBM ranker + template rationale. In-memory vector store, hash embedder
# (instant, no model downloads). Nothing depends on the cloud.

$env:PYTHONPATH   = "src"
$env:COPILOT_VECTOR_STORE = "memory"   # force in-memory Qdrant (no cloud dependency)
$env:COPILOT_SOURCE   = "processed"
$env:COPILOT_EMBEDDER = "hash"   # instant; no torch / model download
$env:LLM_MODE     = "template"   # instant grounded rationale

Write-Host "Starting Clinical Copilot at http://localhost:8501 ..." -ForegroundColor Green
streamlit run src/copilot/app/demo.py --server.port=8501
