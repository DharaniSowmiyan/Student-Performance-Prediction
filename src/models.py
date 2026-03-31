import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ─── FEATURE COLUMN GROUPS ────────────────────────────────────────────────────
def get_feature_columns(features: pd.DataFrame):
    """
    Returns three lists of column names:
        static_cols   — aggregate click / interaction counts
        sequence_cols — binary pattern flags (pat_*)
        hybrid_cols   — both combined
    """
    exclude = {"id_student", "performance_group", "performance_label"}

    static_cols = [
        c for c in features.columns
        if c not in exclude and not c.startswith("pat_")
    ]
    sequence_cols = [
        c for c in features.columns
        if c.startswith("pat_")
    ]
    hybrid_cols = static_cols + sequence_cols

    return static_cols, sequence_cols, hybrid_cols


# ─── BUILD MODELS ─────────────────────────────────────────────────────────────
def build_models():
    """
    Returns a dict of model name → sklearn Pipeline.
    StandardScaler is included so LR, SVM, and neural models get scaled input.
    class_weight='balanced' / scale_pos_weight handles the 69/31 class imbalance.
    Models included:
        - LogisticRegression  : linear baseline
        - RandomForest        : ensemble baseline
        - XGBoost             : gradient boosting (scale_pos_weight for imbalance)
        - LightGBM            : fast gradient boosting
        - SVM                 : RBF kernel, strong on high-dim binary features
    """
    # Ratio of negative to positive samples for XGBoost imbalance handling
    _neg_pos_ratio = 69 / 31  # approx class distribution

    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=_neg_pos_ratio,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
    }


# ─── EVALUATE WITH CROSS-VALIDATION ──────────────────────────────────────────
def evaluate_model_cv(model, X, y, cv=5):
    """
    Runs stratified k-fold cross-validation.
    Returns mean accuracy, precision, recall, f1 across folds.
    """
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    cv_results = cross_validate(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=False,
    )

    return {
        "accuracy":  round(cv_results["test_accuracy"].mean(),           4),
        "precision": round(cv_results["test_precision_weighted"].mean(), 4),
        "recall":    round(cv_results["test_recall_weighted"].mean(),     4),
        "f1":        round(cv_results["test_f1_weighted"].mean(),         4),
    }


# ─── RUN ALL EXPERIMENTS ──────────────────────────────────────────────────────
def run_all_experiments(features: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """
    Trains every model × every feature set combination using cross-validation.

    Returns a DataFrame with one row per (model, feature_set) combination.
    """
    y = (features["performance_group"] == "High").astype(int)

    static_cols, sequence_cols, hybrid_cols = get_feature_columns(features)

    feature_sets = {
        "static":   static_cols,
        "sequence": sequence_cols,
        "hybrid":   hybrid_cols,
    }

    models = build_models()

    rows = []
    for model_name, model in models.items():
        for feat_name, cols in feature_sets.items():
            if not cols:
                print(f"  Skipping {model_name} / {feat_name} — no columns")
                continue

            print(f"  {model_name:22s} | {feat_name:10s} ...", end=" ", flush=True)
            X = features[cols]

            metrics = evaluate_model_cv(model, X, y, cv=cv)
            rows.append({
                "model":      model_name,
                "features":   feat_name,
                **metrics,
            })
            print(f"F1={metrics['f1']:.4f}")

    results = pd.DataFrame(rows)
    results = results.sort_values("f1", ascending=False).reset_index(drop=True)
    return results


# ─── TRAIN & SAVE BEST MODEL ──────────────────────────────────────────────────
def train_and_save_best_model(
    features: pd.DataFrame,
    results: pd.DataFrame,
    save_path: str,
) -> dict:
    """
    Trains the best model (highest F1) on the full dataset and saves it.

    Saves:
        results/best_model.pkl  — the fitted pipeline
        results/best_model_info.csv — metadata about which model was chosen

    Returns a dict with model name, feature set, and metrics.
    """
    best_row = results.iloc[0]
    model_name = best_row["model"]
    feat_name  = best_row["features"]

    print(f"\nBest model: {model_name}  |  features: {feat_name}  |  F1={best_row['f1']:.4f}")

    static_cols, sequence_cols, hybrid_cols = get_feature_columns(features)
    feature_sets = {
        "static":   static_cols,
        "sequence": sequence_cols,
        "hybrid":   hybrid_cols,
    }

    cols = feature_sets[feat_name]
    X    = features[cols]
    y    = (features["performance_group"] == "High").astype(int)

    models = build_models()
    best_pipeline = models[model_name]
    best_pipeline.fit(X, y)

    model_data = {
        "pipeline":     best_pipeline,
        "model_name":   model_name,
        "feature_set":  feat_name,
        "feature_cols": cols,
        "metrics":      best_row.to_dict(),
    }

    with open(save_path + "best_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    pd.DataFrame([best_row]).to_csv(save_path + "best_model_info.csv", index=False)

    print(f"Saved to {save_path}best_model.pkl")
    return model_data


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
def get_confusion_matrix(
    features: pd.DataFrame,
    model_name: str,
    feat_name: str,
) -> np.ndarray:
    """
    Fits the chosen model on 80% of data and returns confusion matrix on 20%.
    Used for visualisation only.
    """
    from sklearn.model_selection import train_test_split

    static_cols, sequence_cols, hybrid_cols = get_feature_columns(features)
    feature_sets = {
        "static":   static_cols,
        "sequence": sequence_cols,
        "hybrid":   hybrid_cols,
    }

    cols = feature_sets[feat_name]
    X    = features[cols]
    y    = (features["performance_group"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = build_models()
    model  = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return confusion_matrix(y_test, y_pred)
