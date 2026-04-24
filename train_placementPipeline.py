import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

RANDOM_STATE = 42

# ============================================================================
# PERUBAHAN UTAMA vs train_churnPipeline.py:
# 1. Notebook2 memiliki DUA task: klasifikasi (placement_status) dan regresi
#    (salary_lpa) — jadi file ini menyediakan dua fungsi train terpisah.
# 2. Kolom ordinal & nominal disesuaikan dengan dataset placement (bukan lagi
#    year/stress_level/internet_quality/gender/course seperti versi churn).
# 3. Numerical pipeline memakai median + StandardScaler (sesuai notebook 6.1)
#    — lebih robust terhadap outlier daripada mean imputer versi churn.
# 4. Model default dipilih Gradient Boosting karena di notebook (bab 6-7)
#    konsisten menjadi pemenang pada metrik F1-macro (klasifikasi) & MAE (regresi).
# 5. Feature engineering (academic_avg, practical_experience, skill_score,
#    wellbeing_score, is_no_backlog) diekstrak ke fungsi terpisah agar bisa
#    dipakai ulang oleh pipeline orchestrator sebelum train-test-split.
# ============================================================================


def add_feature_engineering(df):
    """Turunan fitur sesuai bagian 4.5 notebook2."""
    df = df.copy()
    df["academic_avg"] = (df["tenth_percentage"] + df["twelfth_percentage"]
                          + df["cgpa"] * 10) / 3
    df["practical_experience"] = (df["projects_completed"]
                                   + df["internships_completed"]
                                   + df["hackathons_participated"])
    df["skill_score"] = (df["coding_skill_rating"]
                          + df["communication_skill_rating"]
                          + df["aptitude_skill_rating"]) / 3
    df["wellbeing_score"] = df["sleep_hours"] - 0.5 * df["stress_level"]
    df["is_no_backlog"] = (df["backlogs"] == 0).astype(int)
    return df


def _build_preprocessor(x_train):
    # Kolom kategorikal ordinal & nominal didefinisikan eksplisit (lihat 6.1 notebook2).
    ordinal_cat = ["family_income_level", "city_tier", "extracurricular_involvement"]
    nominal_cat = ["gender", "branch", "part_time_job", "internet_access"]
    # Sisanya = numerik (termasuk fitur turunan hasil feature engineering)
    numeric_features = [c for c in x_train.columns
                        if c not in ordinal_cat + nominal_cat]

    # Numeric: imputasi median (tahan outlier) + StandardScaler (untuk model linear).
    numeric_preprocess = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("num_scaler",  StandardScaler()),
    ])

    # Ordinal: urutan kategori mengikuti konvensi domain di notebook 6.1.
    ordinal_preprocess = Pipeline([
        ("ord_imputer", SimpleImputer(strategy="most_frequent")),
        ("ord_encoder", OrdinalEncoder(
            categories=[
                ["Low", "Medium", "High"],        # family_income_level
                ["Tier 3", "Tier 2", "Tier 1"],   # city_tier (Tier 1 = kota besar)
                ["Low", "Medium", "High"],        # extracurricular_involvement
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # Nominal: one-hot dengan drop='first' (hindari dummy trap di model linear).
    nominal_preprocess = Pipeline([
        ("nom_imputer", SimpleImputer(strategy="most_frequent")),
        ("nom_encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("ord", ordinal_preprocess, ordinal_cat),
        ("nom", nominal_preprocess, nominal_cat),
    ], remainder="drop")


def train_classifier(x_train, y_train):
    """Latih model klasifikasi placement_status -> kembalikan run_id MLflow."""
    preprocess = _build_preprocessor(x_train)

    placement_clf = Pipeline([
        ("preprocessing", preprocess),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE)),
    ])

    # PERUBAHAN: nama eksperimen MLflow disesuaikan dengan studi kasus notebook2.
    mlflow.set_experiment("Student Placement - Classification")

    with mlflow.start_run(run_name="placement_classifier") as run:
        # TAMBAHAN: log lengkap hyperparameter + meta data split untuk reproducibility.
        mlflow.log_param("model", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_train_samples", len(x_train))
        mlflow.log_param("n_features", x_train.shape[1])

        placement_clf.fit(x_train, y_train)

        # Persistence #1: simpan sebagai .pkl (Pickle via joblib).
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(placement_clf, "artifacts/placement_classifier.pkl")
        # Persistence #2: log ke MLflow + daftarkan ke Model Registry
        # (memenuhi instruksi "atau langsung melalui MLflow Model Registry").
        mlflow.sklearn.log_model(
            placement_clf,
            artifact_path="model",
            registered_model_name="placement_classifier",
        )

    return run.info.run_id


def train_regressor(x_train, y_train):
    """Latih model regresi salary_lpa (subset Placed) -> kembalikan run_id MLflow."""
    preprocess = _build_preprocessor(x_train)

    salary_reg = Pipeline([
        ("preprocessing", preprocess),
        ("regressor", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE)),
    ])

    mlflow.set_experiment("Student Placement - Regression")

    with mlflow.start_run(run_name="salary_regressor") as run:
        # TAMBAHAN: log lengkap hyperparameter + meta data split.
        mlflow.log_param("model", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_train_samples", len(x_train))
        mlflow.log_param("n_features", x_train.shape[1])

        salary_reg.fit(x_train, y_train)

        # Persistence #1: .pkl (Pickle via joblib)
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(salary_reg, "artifacts/salary_regressor.pkl")
        # Persistence #2: MLflow artifact + Model Registry.
        mlflow.sklearn.log_model(
            salary_reg,
            artifact_path="model",
            registered_model_name="salary_regressor",
        )

    return run.info.run_id
