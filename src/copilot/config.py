import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv():
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = REPO_ROOT / "data"
SYNTHEA_DIR = DATA_DIR / "synthea"
QDRANT_PATH = DATA_DIR / "qdrant"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
GUIDELINES_PATH = REPO_ROOT / "src" / "copilot" / "guidelines" / "guidelines.json"
INTERACTION_RULES_PATH = REPO_ROOT / "src" / "copilot" / "safety" / "interaction_rules.json"
DDINTER_DIR = DATA_DIR / "clinical" / "ddinter"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_COHORT_PATH = PROCESSED_DIR / "cohort.json"

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM = 768
PATIENT_COLLECTION = "patient_cases"
GUIDELINE_COLLECTION = "guideline_chunks"
TOP_K_PATIENTS = 5
TOP_K_GUIDELINES = 5

RANKER_MODEL_PATH = ARTIFACTS_DIR / "lightgbm_ranker.txt"

REASONING_BASE_MODEL = os.environ.get("RATIONALE_BASE_MODEL", "Qwen/Qwen3-4B-Instruct")
LORA_ADAPTER_PATH = ARTIFACTS_DIR / "lora_adapter"
LLM_MODE = os.environ.get("LLM_MODE", "template")
RATIONALE_GGUF_PATH = Path(os.environ.get(
    "RATIONALE_GGUF_PATH", str(ARTIFACTS_DIR / "rationale-model" / "qwen-rationale-q4.gguf")))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen-rationale")

EMBEDDER_KIND = os.environ.get("COPILOT_EMBEDDER", "auto")
DATA_SOURCE = os.environ.get("COPILOT_SOURCE", "auto")
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()

PRIMARY_CONDITION = "Type 2 Diabetes"
SUPPORTED_CONDITIONS = (
    "Type 2 Diabetes", "Hypertension", "Hyperlipidemia", "Chronic Kidney Disease",
)
