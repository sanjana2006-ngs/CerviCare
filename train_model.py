"""Train CerviCare models and create metrics, feature rankings, and charts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib

from backend.preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    TARGET_COLUMN,
    build_preprocessor,
)
from inspect_dataset import inspect_dataset

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "cervical_data.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
CHART_DIR = ROOT / "charts"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_pipeline(estimator):
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def evaluate(estimator, features, target):
    predictions = estimator.predict(features)
    result = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1_score": float(f1_score(target, predictions, zero_division=0)),
    }
    if hasattr(estimator, "predict_proba"):
        classes = list(estimator.classes_)
        if 1 in classes and target.nunique() == 2:
            positive_probabilities = estimator.predict_proba(features)[:, classes.index(1)]
            result["roc_auc"] = float(roc_auc_score(target, positive_probabilities))
            result["average_precision"] = float(
                average_precision_score(target, positive_probabilities)
            )
    return result


def save_charts(data, confusion, rankings, comparison, target, positive_probabilities):
    sns.set_theme(style="whitegrid", context="notebook")
    palette = ["#b55c79", "#63334f"]

    class_counts = data[TARGET_COLUMN].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        x=["Negative biopsy result", "Positive biopsy result"],
        y=[int(class_counts.get(0, 0)), int(class_counts.get(1, 0))],
        color=palette[0],
        ax=axis,
    )
    axis.set(title="Biopsy Outcome Distribution", xlabel="Recorded outcome", ylabel="Records")
    axis.tick_params(axis="x", rotation=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    top_features = rankings[:10][::-1]
    fig, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=pd.DataFrame(top_features),
        x="importance",
        y="label",
        color=palette[1],
        ax=axis,
    )
    axis.set(
        title="Random Forest Feature Importance",
        xlabel="Mean decrease in impurity (relative importance)",
        ylabel="Questionnaire feature",
    )
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="RdPu",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axis,
    )
    axis.set(title="Random Forest Confusion Matrix", xlabel="Predicted outcome", ylabel="Actual outcome")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    comparison_frame = pd.DataFrame(comparison).set_index("model")
    fig, axis = plt.subplots(figsize=(9, 5))
    comparison_frame[["accuracy", "precision", "recall", "f1_score"]].plot(
        kind="bar",
        ax=axis,
        color=["#d9a8b7", "#a75f7c", "#79506c", "#46344c"],
    )
    axis.set(title="Model Comparison on the Held-Out Test Set", xlabel="", ylabel="Score")
    axis.set_ylim(0, 1)
    axis.legend(title="Metric", loc="upper right", ncol=2)
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        target, positive_probabilities
    )
    roc_auc = roc_auc_score(target, positive_probabilities)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=palette[1],
        linewidth=2,
        label=f"Random Forest (AUC = {roc_auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Chance")
    axis.set(
        title="ROC Curve on the Held-Out Test Set",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(target, positive_probabilities)
    average_precision = average_precision_score(target, positive_probabilities)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        recall,
        precision,
        color=palette[1],
        linewidth=2,
        label=f"Random Forest (AP = {average_precision:.3f})",
    )
    axis.axhline(
        float(target.mean()),
        linestyle="--",
        color="#999999",
        label="Positive-class prevalence",
    )
    axis.set(
        title="Precision-Recall Curve on the Held-Out Test Set",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "precision_recall_curve.png", dpi=180)
    plt.close(fig)


def train():
    print("=" * 48)
    print("       CERVICARE MODEL TRAINING")
    print("=" * 48)

    inspect_dataset()
    raw_data = pd.read_csv(DATASET_PATH, na_values=["?"])
    data = raw_data.copy()
    invalid_values = {}
    for column in data.columns:
        converted = pd.to_numeric(raw_data[column], errors="coerce")
        invalid_mask = raw_data[column].notna() & converted.isna()
        count = int(invalid_mask.sum())
        if count:
            invalid_values[column] = count
        data[column] = converted
    if invalid_values:
        raise ValueError(
            "Dataset contains non-numeric values beyond the documented '?' missing "
            f"marker: {invalid_values}"
        )
    required_columns = set(FEATURE_NAMES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values; cannot train safely.")
    observed_targets = set(data[TARGET_COLUMN].unique())
    if observed_targets != {0, 1}:
        raise ValueError(
            f"Expected binary Biopsy labels 0 and 1, found {sorted(observed_targets)}."
        )

    inputs = data[FEATURE_NAMES]
    target = data[TARGET_COLUMN].astype(int)
    for feature in FEATURE_SCHEMA:
        values = inputs[feature["name"]].dropna()
        if feature["type"] == "boolean":
            invalid = ~values.isin([0, 1])
        else:
            invalid = (values < feature["min"]) | (values > feature["max"])
            if feature["step"] == 1:
                invalid |= values.mod(1).ne(0)
        if invalid.any():
            raise ValueError(
                f"Dataset contains invalid values for {feature['name']}: "
                f"{sorted(values[invalid].unique().tolist())}"
            )
    class_counts = target.value_counts().sort_index()
    positive_fraction = float(target.mean())
    if positive_fraction >= 0.4:
        raise ValueError(
            "The class distribution no longer supports the configured balanced-class strategy."
        )

    print(f"\nDataset loaded: {len(data)} records; source rows are preserved.")
    print(f"Target: {TARGET_COLUMN} | Classes: {class_counts.to_dict()}")
    print(f"Input features: {len(FEATURE_NAMES)}")
    print("Identical questionnaire profiles are kept together during evaluation.")
    print("Positive class is rare; class_weight='balanced' is used.")

    groups = pd.util.hash_pandas_object(inputs, index=False).to_numpy()
    grouped_split = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(grouped_split.split(inputs, target, groups))
    X_train, X_test = inputs.iloc[train_indices], inputs.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]
    training_groups = groups[train_indices]

    model_specs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    forest_pipeline = make_pipeline(model_specs["Random Forest"])

    cross_validation = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(
        clone(forest_pipeline),
        X_train,
        y_train,
        cv=cross_validation,
        groups=training_groups,
        scoring="f1",
        n_jobs=1,
    )

    comparisons = []
    fitted_models = {}
    for name, estimator in model_specs.items():
        pipeline = forest_pipeline if name == "Random Forest" else make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        comparisons.append({"model": name, **evaluate(pipeline, X_test, y_test)})

    fitted_forest = fitted_models["Random Forest"]
    metrics = evaluate(fitted_forest, X_test, y_test)
    predictions = fitted_forest.predict(X_test)
    positive_class_index = list(fitted_forest.classes_).index(1)
    positive_probabilities = fitted_forest.predict_proba(X_test)[:, positive_class_index]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Negative biopsy result", "Positive biopsy result"],
        output_dict=True,
        zero_division=0,
    )

    feature_labels = {item["name"]: item["label"] for item in FEATURE_SCHEMA}
    importances = fitted_forest.named_steps["classifier"].feature_importances_
    transformed_feature_names = fitted_forest.named_steps[
        "preprocessing"
    ].get_feature_names_out()
    rankings = [
        {
            "rank": rank,
            "name": name,
            "label": feature_labels[name],
            "importance": float(importance),
        }
        for rank, (name, importance) in enumerate(
            sorted(
                zip(transformed_feature_names, importances),
                key=lambda pair: pair[1],
                reverse=True,
            ),
            start=1,
        )
    ]

    metadata = {
        "model": "Random Forest Classifier",
        "dataset": "UCI Cervical Cancer (Risk Factors)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors",
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "records": int(len(data)),
        "features": len(FEATURE_NAMES),
        "original_features": int(len(raw_data.columns)),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_distribution": {
            "negative": int(class_counts.get(0, 0)),
            "positive": int(class_counts.get(1, 0)),
        },
        "exact_duplicate_rows_preserved": int(data.duplicated().sum()),
        "repeated_input_profile_rows_preserved": int(pd.Series(groups).duplicated().sum()),
        "unique_input_profiles": int(pd.Series(groups).nunique()),
        "split_strategy": "StratifiedGroupKFold; identical selected input profiles stay together",
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "class_weight": "balanced",
    }
    metrics_report = {
        **metrics,
        "model": "Random Forest Classifier",
        "target": TARGET_COLUMN,
        "positive_label": 1,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "cross_validation": {
            "metric": "f1",
            "folds": 5,
            "mean": float(cv_scores.mean()),
            "standard_deviation": float(cv_scores.std()),
        },
        "model_comparison": comparisons,
        "metadata": metadata,
        "limitations": [
            "The dataset contains only 55 positive biopsy outcomes among 858 records.",
            "The held-out test set has few positive cases, so its metrics are uncertain.",
            "Results describe this historical dataset and do not establish clinical performance.",
            (
                f"On this held-out split, the Random Forest missed {int(matrix[1][0])} "
                f"of {int(matrix[1].sum())} positive biopsy records (false negatives)."
            ),
            (
                f"The Random Forest produced {int(matrix[0][1])} false-positive predictions "
                f"among records with a negative biopsy label."
            ),
        ],
    }

    for directory in (MODEL_DIR, REPORT_DIR, CHART_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_forest, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(
        fitted_forest.named_steps["preprocessing"],
        MODEL_DIR / "preprocessing_pipeline.pkl",
    )
    (MODEL_DIR / "feature_importance.json").write_text(
        json.dumps({"features": rankings}, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_report, indent=2),
        encoding="utf-8",
    )
    save_charts(data, matrix, rankings, comparisons, y_test, positive_probabilities)

    print("\nTraining complete.")
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1 Score : {metrics['f1_score']:.2%}")
    print(f"5-fold training F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("\nTop features:")
    for item in rankings[:5]:
        print(f"{item['rank']}. {item['label']} ({item['importance']:.4f})")
    print("\nModel and reports saved.")
    print("=" * 48)


if __name__ == "__main__":
    train()
