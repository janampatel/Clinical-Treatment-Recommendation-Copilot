"""Parse raw Synthea FHIR into a compact T2D cohort JSON (data/processed/cohort.json)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.data.loader import (  # noqa: E402
    filter_t2d_cohort,
    load_synthea_patients,
    save_processed_cohort,
)


def main() -> None:
    patients = load_synthea_patients()
    cohort = filter_t2d_cohort(patients)
    save_processed_cohort(cohort)
    from copilot.config import PROCESSED_COHORT_PATH

    size_kb = PROCESSED_COHORT_PATH.stat().st_size / 1024
    print(f"Parsed {len(patients)} Synthea patients -> {len(cohort)} T2D cohort")
    print(f"Wrote {PROCESSED_COHORT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
