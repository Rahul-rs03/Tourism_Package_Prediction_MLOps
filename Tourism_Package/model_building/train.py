import os
import pandas as pd
import joblib

from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub.utils import RepositoryNotFoundError


import xgboost as xgb

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("Tourism_Package_Prediction")


# -----------------------------
# Load processed data from HF
# -----------------------------
print("Loading processed datasets from Hugging Face...")

DATASET_REPO = "rahulsuren12/tourism-package-prediction"


Xtrain = load_dataset(DATASET_REPO, data_files="processed/Xtrain.csv")["train"].to_pandas()
Xtest  = load_dataset(DATASET_REPO, data_files="processed/Xtest.csv")["train"].to_pandas()

ytrain = load_dataset(DATASET_REPO, data_files="processed/ytrain.csv")["train"].to_pandas().squeeze()
ytest  = load_dataset(DATASET_REPO, data_files="processed/ytest.csv")["train"].to_pandas().squeeze()

print("Data loaded successfully.")


# -------------------------
# Feature groups (Tourism dataset)
# -------------------------
numeric_features = [
    "Age",
    "MonthlyIncome",
    "NumberOfTrips",
    "DurationOfPitch",
    "PreferredPropertyStar",
    "NumberOfPersonVisiting"
]

categorical_features = [
    "Gender",
    "Occupation",
    "MaritalStatus",
    "CityTier",
    "ProductPitched"
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# -------------------------
# Preprocessing pipeline
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop"
)

# -------------------------
# Base XGBoost model
# -------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

# -------------------------
# Hyperparameter grid
# -------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
}

# -------------------------
# Full model pipeline
# -------------------------
model_pipeline = make_pipeline(preprocessor,xgb_model)

# -----------------------------
# Training + MLflow Logging
# -----------------------------

with mlflow.start_run(run_name="XGBoost_GridSearch_Tourism"):

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

# -------------------------
# Save model
# -------------------------
    model_path = "best_tourism_package_model.joblib"
    joblib.dump(best_model, model_path)

# -------------------------
# Log the model artifact
# -------------------------
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

# -----------------------------
# Upload model to HF Model Hub
# -----------------------------
print("Uploading model to Hugging Face Model Hub...")

api = HfApi()

MODEL_REPO = "rahulsuren12/tourism-package-model"
repo_type = "model"

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type=repo_type)
    print("Model repo already exists.")
except RepositoryNotFoundError:
    print("Model repo not found. Creating new repo...")
    create_repo(repo_id=MODEL_REPO, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj="best_tourism_package_model.joblib",
    path_in_repo="best_tourism_package_model.joblib",
    repo_id=MODEL_REPO,
    repo_type=repo_type,
)

print("Model uploaded successfully to Hugging Face Model Hub.")
