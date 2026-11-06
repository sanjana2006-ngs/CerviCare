"""Shared feature definitions and preprocessing for training and prediction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Biopsy"
EXCLUDED_COLUMNS = {
    "Hinselmann",
    "Schiller",
    "Citology",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "STDs:cervical condylomatosis",
    "STDs:AIDS",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "STDs: Number of diagnosis",
    "Smokes (years)",
    "Smokes (packs/year)",
}

FEATURE_SCHEMA = [
    {"name": "Age", "label": "Age", "question": "What is your age?", "group": "background", "type": "number", "min": 13, "max": 84, "step": 1, "allow_unknown": False},
    {"name": "Smokes", "label": "Do you smoke?", "question": "Do you currently smoke?", "group": "lifestyle", "type": "boolean", "allow_unknown": True, "help": "Choose Don't know if you are unsure; unknown answers are handled by the model's fitted imputer."},
    {"name": "STDs", "label": "History of sexually transmitted infection", "question": "Have you ever been diagnosed with a sexually transmitted infection (STI)?", "group": "medical-history", "type": "boolean", "allow_unknown": True, "help": "This sensitive field is included because it is one of the model inputs. Choose Don't know if unsure; it is not treated as No."},
]

FEATURE_NAMES = [feature["name"] for feature in FEATURE_SCHEMA]
NUMERIC_FEATURES = [
    feature["name"] for feature in FEATURE_SCHEMA if feature["type"] == "number"
]
BINARY_FEATURES = [
    feature["name"] for feature in FEATURE_SCHEMA if feature["type"] == "boolean"
]


def build_preprocessor():
    """Build shared numeric and binary imputers and scaling transformations."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
