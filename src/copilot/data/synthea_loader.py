from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ..config import SYNTHEA_DIR
from .schema import LabPanel, PatientRecord, TreatmentOutcome

LOINC_TO_LAB = {
    "4548-4": "hba1c", "4549-2": "hba1c",
    "33914-3": "egfr", "48642-3": "egfr", "48643-1": "egfr",
    "98979-8": "egfr", "62238-1": "egfr",
    "18262-6": "ldl", "13457-7": "ldl", "2089-1": "ldl",
    "2085-9": "hdl", "8480-6": "sbp", "8462-4": "dbp",
    "2823-3": "potassium", "6298-4": "potassium", "2160-0": "creatinine",
}

CONDITION_MAP = {
    "diabetes mellitus type 2": "Type 2 Diabetes",
    "diabetes": "Type 2 Diabetes",
    "hypertensive": "Hypertension",
    "hypertension": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "chronic kidney disease stage 1": "Chronic Kidney Disease stage 1",
    "chronic kidney disease stage 2": "Chronic Kidney Disease stage 2",
    "chronic kidney disease stage 3": "Chronic Kidney Disease stage 3",
    "chronic kidney disease stage 4": "Chronic Kidney Disease stage 4",
    "chronic kidney disease stage 5": "Chronic Kidney Disease stage 5",
    "chronic kidney disease": "Chronic Kidney Disease",
}


def _age_from_birthdate(birth: str) -> int:
    b = datetime.fromisoformat(birth).date() if "T" not in birth else \
        datetime.fromisoformat(birth.replace("Z", "+00:00")).date()
    today = date.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def _map_condition(display: str) -> Optional[str]:
    d = display.lower()
    for key, canon in CONDITION_MAP.items():
        if key in d:
            return canon
    return None


def parse_bundle(bundle: dict) -> Optional[PatientRecord]:
    patient_id = age = sex = None
    conditions: list[str] = []
    medications: list[str] = []
    allergies: list[str] = []
    labs = LabPanel()

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            patient_id = res.get("id")
            if res.get("birthDate"):
                age = _age_from_birthdate(res["birthDate"])
            g = (res.get("gender") or "").lower()
            sex = "M" if g == "male" else "F" if g == "female" else "U"

        elif rtype == "Condition":
            for coding in res.get("code", {}).get("coding", []):
                canon = _map_condition(coding.get("display", ""))
                if canon and canon not in conditions:
                    conditions.append(canon)

        elif rtype == "MedicationRequest":
            disp = res.get("medicationCodeableConcept", {}).get("text")
            if not disp:
                codings = res.get("medicationCodeableConcept", {}).get("coding", [])
                disp = codings[0].get("display") if codings else None
            if disp and res.get("status") in (None, "active"):
                medications.append(disp)

        elif rtype == "AllergyIntolerance":
            disp = res.get("code", {}).get("text") or \
                next((c.get("display") for c in res.get("code", {}).get("coding", [])), None)
            if disp:
                allergies.append(disp)

        elif rtype == "Observation":
            for coding in res.get("code", {}).get("coding", []):
                field = LOINC_TO_LAB.get(coding.get("code"))
                if field and "valueQuantity" in res and getattr(labs, field) is None:
                    val = res["valueQuantity"].get("value")
                    if val is not None:
                        setattr(labs, field, round(float(val), 1))
            for comp in res.get("component", []):
                for coding in comp.get("code", {}).get("coding", []):
                    field = LOINC_TO_LAB.get(coding.get("code"))
                    if field and "valueQuantity" in comp and getattr(labs, field) is None:
                        setattr(labs, field, round(float(comp["valueQuantity"]["value"]), 1))

    if patient_id is None or age is None:
        return None

    return PatientRecord(
        patient_id=patient_id, age=age, sex=sex or "U",
        conditions=conditions, medications=medications, allergies=allergies, labs=labs,
        treatment_history=_derive_treatment_history(medications, labs),
    )


def _derive_treatment_history(medications, labs):
    diabetes_drugs = {
        "metformin", "insulin", "glargine", "empagliflozin", "dapagliflozin",
        "semaglutide", "liraglutide", "sitagliptin", "glipizide", "glyburide",
        "pioglitazone", "canagliflozin",
    }
    hba1c = labs.hba1c
    success = 0.85 if hba1c is None else max(0.3, min(0.95, 1.0 - (hba1c - 6.5) * 0.1))
    history = []
    for med in medications:
        if any(d in med.lower() for d in diabetes_drugs):
            history.append(TreatmentOutcome(drug=med, success=round(success, 2)))
    return history


def load_synthea_patients(directory: Path | None = None) -> list[PatientRecord]:
    directory = directory or SYNTHEA_DIR
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if bundle.get("resourceType") != "Bundle":
            continue
        rec = parse_bundle(bundle)
        if rec is not None:
            records.append(rec)
    return records
