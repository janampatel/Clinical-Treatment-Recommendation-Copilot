"""Small hand-built T2D cohort for tests and as a fallback when no Synthea
data is present."""
from __future__ import annotations

from .schema import LabPanel, PatientRecord, TreatmentOutcome


def _p(**kw) -> PatientRecord:
    return PatientRecord(**kw)


MOCK_COHORT: list[PatientRecord] = [
    _p(
        patient_id="C001", age=58, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["Lisinopril"],
        allergies=[],
        labs=LabPanel(hba1c=8.4, egfr=88, ldl=130, hdl=40, sbp=148, dbp=92,
                      potassium=4.2, creatinine=0.9),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.9),
                           TreatmentOutcome(drug="Empagliflozin", success=0.85)],
    ),
    _p(
        patient_id="C002", age=63, sex="F",
        conditions=["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"],
        medications=["Amlodipine", "Atorvastatin"],
        allergies=[],
        labs=LabPanel(hba1c=9.1, egfr=72, ldl=160, hdl=38, sbp=152, dbp=95,
                      potassium=4.5, creatinine=1.0),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.8),
                           TreatmentOutcome(drug="Semaglutide", success=0.9)],
    ),
    _p(
        patient_id="C003", age=71, sex="M",
        conditions=["Type 2 Diabetes", "Chronic Kidney Disease stage 4"],
        medications=["Insulin glargine"],
        allergies=[],
        labs=LabPanel(hba1c=8.0, egfr=25, ldl=110, hdl=45, sbp=138, dbp=80,
                      potassium=5.1, creatinine=2.6),
        treatment_history=[TreatmentOutcome(drug="Insulin glargine", success=0.75)],
    ),
    _p(
        patient_id="C004", age=49, sex="F",
        conditions=["Type 2 Diabetes"],
        medications=[],
        allergies=["Sulfa"],
        labs=LabPanel(hba1c=7.6, egfr=95, ldl=120, hdl=50, sbp=124, dbp=78,
                      potassium=4.0, creatinine=0.7),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.95)],
    ),
    _p(
        patient_id="C005", age=66, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension", "Chronic Kidney Disease stage 3"],
        medications=["Losartan"],
        allergies=[],
        labs=LabPanel(hba1c=8.8, egfr=48, ldl=140, hdl=42, sbp=144, dbp=88,
                      potassium=4.8, creatinine=1.6),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.7),
                           TreatmentOutcome(drug="Dapagliflozin", success=0.88)],
    ),
    _p(
        patient_id="C006", age=54, sex="F",
        conditions=["Type 2 Diabetes", "Hyperlipidemia"],
        medications=["Rosuvastatin"],
        allergies=[],
        labs=LabPanel(hba1c=10.2, egfr=90, ldl=190, hdl=35, sbp=130, dbp=82,
                      potassium=4.1, creatinine=0.8),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.6),
                           TreatmentOutcome(drug="Insulin glargine", success=0.85)],
    ),
    _p(
        patient_id="C007", age=60, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["Hydrochlorothiazide"],
        allergies=[],
        labs=LabPanel(hba1c=7.9, egfr=80, ldl=125, hdl=44, sbp=146, dbp=90,
                      potassium=3.9, creatinine=1.0),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.85),
                           TreatmentOutcome(drug="Sitagliptin", success=0.7)],
    ),
    _p(
        patient_id="C008", age=68, sex="F",
        conditions=["Type 2 Diabetes", "Hypertension", "Chronic Kidney Disease stage 3"],
        medications=["Amlodipine", "Atorvastatin"],
        allergies=[],
        labs=LabPanel(hba1c=8.5, egfr=52, ldl=150, hdl=40, sbp=150, dbp=92,
                      potassium=4.6, creatinine=1.4),
        treatment_history=[TreatmentOutcome(drug="Empagliflozin", success=0.9),
                           TreatmentOutcome(drug="Metformin", success=0.75)],
    ),
    _p(
        patient_id="C009", age=45, sex="M",
        conditions=["Type 2 Diabetes"],
        medications=[],
        allergies=[],
        labs=LabPanel(hba1c=7.2, egfr=100, ldl=115, hdl=48, sbp=122, dbp=76,
                      potassium=4.0, creatinine=0.8),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.92)],
    ),
    _p(
        patient_id="C010", age=73, sex="F",
        conditions=["Type 2 Diabetes", "Hypertension", "Hyperlipidemia",
                    "Chronic Kidney Disease stage 4"],
        medications=["Losartan", "Atorvastatin", "Insulin glargine"],
        allergies=[],
        labs=LabPanel(hba1c=8.3, egfr=22, ldl=145, hdl=39, sbp=140, dbp=84,
                      potassium=5.3, creatinine=2.9),
        treatment_history=[TreatmentOutcome(drug="Insulin glargine", success=0.8)],
    ),
    _p(
        patient_id="C011", age=57, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["Lisinopril", "Amlodipine"],
        allergies=[],
        labs=LabPanel(hba1c=8.7, egfr=85, ldl=135, hdl=41, sbp=150, dbp=94,
                      potassium=4.4, creatinine=0.9),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.88),
                           TreatmentOutcome(drug="Semaglutide", success=0.91)],
    ),
    _p(
        patient_id="C012", age=62, sex="F",
        conditions=["Type 2 Diabetes", "Hyperlipidemia"],
        medications=["Atorvastatin"],
        allergies=["Penicillin"],
        labs=LabPanel(hba1c=9.5, egfr=78, ldl=175, hdl=37, sbp=128, dbp=80,
                      potassium=4.2, creatinine=1.0),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.7),
                           TreatmentOutcome(drug="Dapagliflozin", success=0.86)],
    ),
    _p(
        patient_id="C013", age=50, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["Metoprolol"],
        allergies=[],
        labs=LabPanel(hba1c=8.1, egfr=92, ldl=128, hdl=43, sbp=142, dbp=88,
                      potassium=4.3, creatinine=0.9),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.9),
                           TreatmentOutcome(drug="Empagliflozin", success=0.84)],
    ),
    _p(
        patient_id="C014", age=69, sex="F",
        conditions=["Type 2 Diabetes", "Chronic Kidney Disease stage 3"],
        medications=["Empagliflozin"],
        allergies=[],
        labs=LabPanel(hba1c=7.8, egfr=55, ldl=132, hdl=46, sbp=134, dbp=80,
                      potassium=4.7, creatinine=1.3),
        treatment_history=[TreatmentOutcome(drug="Empagliflozin", success=0.89),
                           TreatmentOutcome(drug="Metformin", success=0.78)],
    ),
    _p(
        patient_id="C015", age=55, sex="M",
        conditions=["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"],
        medications=["Lisinopril", "Rosuvastatin"],
        allergies=[],
        labs=LabPanel(hba1c=8.9, egfr=83, ldl=165, hdl=38, sbp=149, dbp=93,
                      potassium=4.5, creatinine=1.0),
        treatment_history=[TreatmentOutcome(drug="Metformin", success=0.86),
                           TreatmentOutcome(drug="Semaglutide", success=0.9)],
    ),
]


# Treatment-naive T2D + hypertension demo patient.
DEMO_QUERY_PATIENT = _p(
    patient_id="QUERY-DEMO", age=61, sex="M",
    conditions=["Type 2 Diabetes", "Hypertension"],
    medications=["Lisinopril"],
    allergies=[],
    labs=LabPanel(hba1c=8.6, egfr=82, ldl=145, hdl=40, sbp=150, dbp=92,
                  potassium=4.4, creatinine=1.0),
    treatment_history=[],
)


# Demo patient that triggers contraindications (low eGFR, sulfa allergy).
DEMO_QUERY_CKD = _p(
    patient_id="QUERY-CKD", age=70, sex="F",
    conditions=["Type 2 Diabetes", "Hypertension", "Chronic Kidney Disease stage 4"],
    medications=["Losartan"],
    allergies=["Sulfa"],
    labs=LabPanel(hba1c=8.8, egfr=24, ldl=150, hdl=38, sbp=146, dbp=88,
                  potassium=5.2, creatinine=2.7),
    treatment_history=[],
)


def load_mock_cohort() -> list[PatientRecord]:
    return [p.model_copy(deep=True) for p in MOCK_COHORT]
