from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LabPanel(BaseModel):
    hba1c: Optional[float] = None
    egfr: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    potassium: Optional[float] = None
    creatinine: Optional[float] = None


class TreatmentOutcome(BaseModel):
    drug: str
    success: float = Field(ge=0.0, le=1.0)


class PatientRecord(BaseModel):
    patient_id: str
    age: int
    sex: str
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    labs: LabPanel = Field(default_factory=LabPanel)
    treatment_history: list[TreatmentOutcome] = Field(default_factory=list)

    def has_condition(self, name: str) -> bool:
        name = name.lower()
        return any(name in c.lower() for c in self.conditions)

    def to_clinical_text(self) -> str:
        labs = self.labs
        lab_bits = []
        if labs.hba1c is not None:
            lab_bits.append(f"HbA1c {labs.hba1c}%")
        if labs.egfr is not None:
            lab_bits.append(f"eGFR {labs.egfr} mL/min/1.73m2")
        if labs.ldl is not None:
            lab_bits.append(f"LDL {labs.ldl} mg/dL")
        if labs.sbp is not None and labs.dbp is not None:
            lab_bits.append(f"BP {labs.sbp}/{labs.dbp} mmHg")
        if labs.potassium is not None:
            lab_bits.append(f"potassium {labs.potassium} mEq/L")
        if labs.creatinine is not None:
            lab_bits.append(f"creatinine {labs.creatinine} mg/dL")

        parts = [
            f"{self.age}-year-old {'male' if self.sex == 'M' else 'female'} patient.",
            f"Conditions: {', '.join(self.conditions) or 'none recorded'}.",
            f"Current medications: {', '.join(self.medications) or 'none'}.",
            f"Allergies: {', '.join(self.allergies) or 'none known'}.",
        ]
        if lab_bits:
            parts.append("Labs: " + ", ".join(lab_bits) + ".")
        return " ".join(parts)
