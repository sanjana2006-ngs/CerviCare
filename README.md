# CerviCare

**Design and Development of a Working Prototype for Cervical Cancer Risk Classification Using a Random Forest Classifier with Feature Ranking**

## Abstract

CerviCare is a student-built machine-learning prototype for educational screening-support classification. It uses the UCI Cervical Cancer (Risk Factors) dataset, a saved Scikit-learn preprocessing and Random Forest pipeline, and a Flask web application. The project reports actual held-out evaluation metrics and model-derived feature rankings. It is not a diagnostic system and does not make clinical recommendations.

## Problem Statement

This mini project explores whether structured demographic and historical risk-factor data can be used by a machine-learning classifier to identify patterns associated with a recorded biopsy outcome. It also demonstrates responsible preprocessing, evaluation, and communication of model limitations.

## Objectives

- Inspect the source dataset and document the target and leakage safeguards.
- Preprocess missing values using a fitted Scikit-learn pipeline.
- Train and evaluate a Random Forest classifier on a stratified hold-out split.
- Rank input features using the trained model's feature importances.
- Provide a small Flask API and responsive, privacy-minded user interface.
- Communicate that a model classification is not a diagnosis.

## Proposed Solution

The browser presents a three-step questionnaire using only the selected dataset input fields. The Flask API validates the current submission and passes it to the same saved preprocessing-and-classification pipeline used during training. It returns the actual predicted Biopsy class without presenting an uncalibrated probability as personal cancer risk. Questionnaire responses are not saved by the server; the current result and its inputs are kept temporarily in the browser tab for the result and printable report.

## Dataset

- **Name:** UCI Cervical Cancer (Risk Factors)
- **Source:** [UCI Machine Learning Repository, dataset 383](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
- **Citation:** Fernandes, K., Cardoso, J., & Fernandes, J. (2017). *Cervical Cancer (Risk Factors)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z310
- **License:** CC BY 4.0; attribute the dataset to its authors and UCI when redistributing it.
- **Original shape:** 858 records and 36 columns.
- **Local file:** `dataset/cervical_data.csv` (the official UCI CSV).

The UCI data was downloaded from the official dataset page. If the file is absent, download its CSV and put it at exactly `dataset/cervical_data.csv`. Do not replace it with synthetic or fabricated records.

## Methodology

### Dataset inspection and target selection

Run `python inspect_dataset.py` before training. The inspection report lists every column, missing-value count, observed unique values, and the distributions of the four candidate outcomes.

`Biopsy` is the selected binary target. The dataset also contains `Hinselmann`, `Schiller`, and `Citology` outcomes; all three are excluded from model input. The diagnosis indicators `Dx:Cancer`, `Dx:CIN`, `Dx:HPV`, and `Dx` are also excluded because they may disclose information from the same clinical assessment as the selected target. This avoids using a test outcome or diagnosis flag as a questionnaire predictor. To keep the public questionnaire comfortable and privacy-minded, detailed sexual-history and reproductive variables are excluded from the trained feature set.

The two STD diagnosis timing columns are excluded because each has 787 missing values out of 858. The constant-zero `STDs:cervical condylomatosis` and `STDs:AIDS` columns have no predictive information. The final three questionnaire inputs are `Age`, `Smokes`, and the aggregate `STDs` history field, all present in the UCI dataset. No replacement questions or values are invented. The rationale and inspection output are recorded in `reports/dataset_inspection.json`.

### Requested questionnaire topics checked against the dataset

- **Age**, **smoking status**, and an aggregate **STI history** indicator exist in the dataset and are model inputs. The STI history question offers “Don't know”; this is sent as missing and imputed by the fitted pipeline, never converted to “No”.
- The dataset contains no HPV vaccination, HPV testing, sexually-active, or age-at-first-period feature. Partner count and age at first intercourse are not equivalent to a yes/no sexual-activity response. `STDs:HPV` records an HPV-related condition, not vaccination or testing; `Dx:HPV` is a diagnosis indicator and is excluded to avoid target leakage. These items are not added as model features or inferred from other fields.
- Detailed sexual, reproductive, smoking-duration, and pack-year questions are not part of the updated simple questionnaire.

The questionnaire asks only for the three model inputs: age, smoking status, and STI history. "Don't know" is available for binary answers and is passed as missing, then imputed by the saved pipeline using the training-fitted most-frequent value. No identity or contact information is requested.

The source contains repeated rows and repeated questionnaire profiles. Since the dataset provides no patient identifiers to establish that these are data-entry errors, training preserves all 858 records (803 negative and 55 positive Biopsy labels). Instead, a stratified group split keeps identical selected-input profiles within the same partition, preventing the same profile from appearing in both training and test data. A fixed random state and a 5-fold `StratifiedGroupKFold` are used; the cross-validation F1 score is calculated only on the training partition. Because the positive outcome is uncommon, the Random Forest uses `class_weight="balanced"`.

### Preprocessing

All selected inputs are numeric in the source data. The preprocessing `ColumnTransformer` applies median imputation and scaling to Age, and most-frequent imputation and scaling to the binary smoking and aggregate STI-history fields. Each transformation is fitted only on its training fold or training split. The fitted preprocessor is saved and reused for prediction; no independent preprocessing is applied to questionnaire submissions.

### Random Forest

The principal model is a Random Forest classifier with 400 trees, `min_samples_leaf=2`, `class_weight="balanced"`, and `random_state=42`. Logistic Regression and a Decision Tree are included as simple comparison baselines, evaluated on the same held-out test split.

### Feature Ranking

Feature rankings are extracted from the fitted Random Forest's `feature_importances_`, matched to readable questionnaire labels, and sorted in descending order. Up to ten of the model's actual input features are displayed in the web application. Impurity-based importance describes how the fitted model uses inputs; it is not evidence of medical causation.

### Model Evaluation

Accuracy, precision, recall, F1-score, ROC-AUC, average precision, confusion matrix, classification report, model comparison, ROC and precision-recall curves, and a training-only cross-validation summary are calculated from actual model runs and saved to `reports/model_metrics.json` and `charts/`. The dashboard reads these generated values. Do not interpret the accuracy alone as model quality: there are few positive cases, and the held-out test set contains only a small number of them.

## System Architecture

```text
UCI CSV
  -> dataset inspection
  -> missing-value handling
  -> stratified, input-profile-grouped train/test split
  -> fitted preprocessing + Random Forest
  -> evaluation, feature rankings, charts, saved artifacts
  -> Flask API
  -> Vanilla JavaScript questionnaire and dashboard
```

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn and Joblib
- Matplotlib and Seaborn
- Flask
- HTML5, CSS3, and Vanilla JavaScript

## Installation

From the project root in Windows PowerShell:

```powershell
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The supplied dataset is already at `dataset/cervical_data.csv`. If you need to obtain it again, download the CSV from the [official UCI dataset page](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) and place it at that exact path.

## Running the Project

Inspect the dataset and its fields:

```powershell
python inspect_dataset.py
```

Train the model and regenerate all model artifacts and charts:

```powershell
python train_model.py
```

Start the local application:

```powershell
python backend/app.py
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001). The Flask application loads saved artifacts; it does not retrain when it starts. If a saved model is missing, the prediction endpoint returns an explicit error instructing you to train it.

## Project Structure

```text
dataset/cervical_data.csv          Official UCI dataset
model/                             Generated model and feature ranking
charts/                            Generated evaluation charts
reports/                           Dataset inspection and model metrics
backend/app.py                     Flask pages and API
backend/preprocessing.py           Shared feature schema and pipeline
frontend/                          HTML pages, CSS, and JavaScript
inspect_dataset.py                 Dataset inspection utility
train_model.py                     Training and evaluation script
tests/test_backend.py              Backend/API tests
requirements.txt                   Python dependencies
```

Generated model files and the dataset are excluded from version control by `.gitignore`; they remain in the local project directory and can be recreated or retrieved using the steps above.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Confirms that the local Flask server responds. |
| `POST` | `/api/predict` | Validates the questionnaire and returns the predicted Biopsy class. |
| `GET` | `/api/metrics` | Returns actual saved evaluation metrics and metadata. |
| `GET` | `/api/features` | Returns the calculated top-ten ranking and questionnaire schema. |

The prediction request body is `{ "responses": { ... } }`, with the three feature names provided by `/api/features`: `Age`, `Smokes`, and `STDs`. Binary values may be `null` when the user is unsure; the saved imputer handles them. Unknown fields, omitted fields, invalid types, and out-of-range values are rejected. The response includes the numeric `prediction` (`0` or `1`) and wording that identifies it as a predicted negative or positive Biopsy class. It does not expose a user-facing probability.

## Limitations

- This is a small historical dataset collected in one location; it does not establish performance for a current or general population.
- The positive Biopsy class is rare (55 of 858 source records; repeated rows are preserved), making positive-class metric estimates uncertain.
- The model does not provide a calibrated individual risk estimate.
- Random Forest feature importance is not causal or clinical interpretation.
- This prototype has not undergone clinical validation, regulatory review, or deployment review.
- The questionnaire includes an STI-history question. Do not enter real patient or identifying information.

## Future Scope

- Evaluate the model on larger, representative, externally validated data.
- Study calibration and threshold selection with clinically qualified collaborators.
- Review usability, privacy, accessibility, and fairness before any real-world use.
- Add clinician-reviewed explanations and translated questionnaire labels.

## Medical Disclaimer

> **Medical Disclaimer:** Predictions are experimental model outputs from a historical dataset. CerviCare cannot diagnose cervical cancer: a negative prediction does not rule out disease and a positive prediction does not confirm it. A healthcare professional should interpret screening concerns. The application does not replace recommended medical screening.

## References

1. Fernandes, K., Cardoso, J., & Fernandes, J. (2017). *Cervical Cancer (Risk Factors)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z310
2. Scikit-learn documentation: [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
3. Scikit-learn documentation: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Testing

Run the backend tests from the project root:

```powershell
python -m unittest discover -s tests -v
```
