"""Generate the LoRA fine-tuning dataset (chat JSONL) from the cohort."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import REPO_ROOT
from ..data.loader import filter_t2d_cohort, load_patients
from ..data.schema import LabPanel, PatientRecord
from ..engine import CopilotEngine
from ..retrieval.embeddings import HashEmbedder
from ..retrieval.retriever import build_index
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .rationale import TemplateRationaleGenerator, build_factsheet

OUTPUT_PATH = REPO_ROOT / "data" / "lora" / "train.jsonl"

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _jitter(p: PatientRecord, seed: int) -> PatientRecord:
    import random

    rng = random.Random(seed)

    def clamp(v, lo, hi):
        return None if v is None else max(lo, min(hi, round(v, 1)))

    labs = p.labs
    new_labs = LabPanel(
        hba1c=clamp((labs.hba1c or 8.0) + rng.uniform(-0.6, 0.6), 6.0, 12.0),
        egfr=clamp((labs.egfr or 80) + rng.uniform(-12, 12), 12, 120),
        ldl=clamp((labs.ldl or 130) + rng.uniform(-20, 20), 60, 220),
        hdl=labs.hdl,
        sbp=clamp((labs.sbp or 140) + rng.uniform(-10, 10), 110, 175),
        dbp=labs.dbp,
        potassium=clamp((labs.potassium or 4.3) + rng.uniform(-0.3, 0.5), 3.5, 5.8),
        creatinine=labs.creatinine,
    )
    return p.model_copy(update={
        "patient_id": f"{p.patient_id}-v{seed}",
        "age": max(30, min(90, p.age + rng.randint(-5, 5))),
        "labs": new_labs,
    })


def augment_cohort(cohort: list[PatientRecord], variants_per_patient: int) -> list[PatientRecord]:
    out: list[PatientRecord] = list(cohort)
    for p in cohort:
        for k in range(variants_per_patient):
            out.append(_jitter(p, seed=hash((p.patient_id, k)) % 100000))
    return out


def _call_anthropic(messages: list[dict], model: str = _ANTHROPIC_MODEL) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = next(m["content"] for m in messages if m["role"] == "system")
    user_turns = [{"role": m["role"], "content": m["content"]}
                  for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model, max_tokens=300, system=system, messages=user_turns,
    )
    return resp.content[0].text.strip()


def generate_dataset(
    provider: str = "template",
    variants_per_patient: int = 9,
    top_k: int = 3,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    cohort = filter_t2d_cohort(load_patients("auto"))
    patients = augment_cohort(cohort, variants_per_patient)

    index = build_index(source="auto", embedder=HashEmbedder())
    engine = CopilotEngine(index=index)
    template = TemplateRationaleGenerator()

    examples: list[dict] = []
    for patient in patients:
        result = engine.recommend(patient, explain=False)
        for ranked in result.ranked[:top_k]:
            if not ranked.candidate.citations:
                continue
            factsheet = build_factsheet(ranked, patient, result.safety)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(factsheet)},
            ]
            if provider == "anthropic":
                target = _call_anthropic(messages)
            else:
                target = template.generate(ranked, patient, result.safety)
            examples.append({"messages": messages + [
                {"role": "assistant", "content": target}]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return output_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate Stage 5 LoRA dataset")
    ap.add_argument("--provider", choices=["template", "anthropic"], default="template")
    ap.add_argument("--variants", type=int, default=9)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    path = generate_dataset(
        provider=args.provider, variants_per_patient=args.variants, top_k=args.top_k
    )
    n = sum(1 for _ in path.open(encoding="utf-8"))
    print(f"Wrote {n} examples to {path}")
