"""Inspect the downloaded UCI dataset before training or feature selection."""

import json
from pathlib import Path

import pandas as pd

from backend.preprocessing import EXCLUDED_COLUMNS, FEATURE_NAMES, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "cervical_data.csv"
REPORT_PATH = ROOT / "reports" / "dataset_inspection.json"
TARGET_CANDIDATES = ["Hinselmann", "Schiller", "Citology", "Biopsy"]


def inspect_dataset():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Download the UCI Cervical Cancer "
            "Risk Factors CSV and place it at dataset/cervical_data.csv."
        )

    data = pd.read_csv(DATASET_PATH, na_values=["?"])
    invalid_counts = {}
    for column in data.columns:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_counts[column] = int((converted.isna() & data[column].notna()).sum())
        data[column] = converted

    columns = {}
    for column in data.columns:
        series = data[column]
        columns[column] = {
            "missing": int(series.isna().sum()),
            "unique_non_missing": int(series.nunique(dropna=True)),
            "unique_values": sorted(series.dropna().unique().tolist()),
            "invalid_values_coerced_to_missing": invalid_counts[column],
        }

    target_counts = {
        target: {
            str(label): int(count)
            for label, count in data[target].value_counts(dropna=False).sort_index().items()
        }
        for target in TARGET_CANDIDATES
        if target in data.columns
    }

    report = {
        "dataset": "UCI Cervical Cancer (Risk Factors)",
        "source": "https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors",
        "rows": int(len(data)),
        "column_count": int(len(data.columns)),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "unique_selected_input_profiles": int(data[FEATURE_NAMES].drop_duplicates().shape[0]),
        "columns": columns,
        "target_candidate_distributions": target_counts,
        "selected_target": TARGET_COLUMN,
        "selected_features": FEATURE_NAMES,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "selection_rationale": (
            "Biopsy is the selected binary outcome. Hinselmann, Schiller, and Citology "
            "are other screening outcomes and are excluded. Dx:Cancer, Dx:CIN, Dx:HPV, "
            "and Dx are diagnosis indicators that could leak information from the same "
            "clinical assessment, so they are excluded. Two STD diagnosis timing fields "
            "are excluded because 787 of 858 values are missing. Two constant-zero fields "
            "are excluded because they carry no predictive information. Intimate "
            "reproductive and detailed sexual-history variables are excluded. The final "
            "inputs are Age, the dataset's smoking-status field, and its aggregate STD "
            "history field. Smoking duration and pack-year fields are excluded to keep "
            "the questionnaire simple. "
            "Source rows are preserved; identical selected-input profiles are assigned to "
            "the same stratified group during hold-out and cross-validation to avoid profile leakage."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset: {DATASET_PATH}")
    print(f"Records: {len(data)} | Columns: {len(data.columns)}")
    print("\nColumn inspection (missing | unique non-missing | unique values):")
    for name, details in columns.items():
        print(
            f"- {name}: {details['missing']} | {details['unique_non_missing']} | "
            f"{details['unique_values']}"
        )
    print("\nCandidate target distributions:")
    for target, counts in target_counts.items():
        print(f"- {target}: {counts}")
    print(f"\nSelected target: {TARGET_COLUMN}")
    print(f"Selected input features: {len(FEATURE_NAMES)}")
    print(f"Inspection report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    inspect_dataset()
