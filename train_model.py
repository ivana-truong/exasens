from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# Maybe consider SMOTENC if there's difficulty identifying COPD
# from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any


def build_pipeline() -> Pipeline:
    # Identify numerical vs categorical cols
    num_cols = ["Imaginary Part - avg", "Real Part - avg"]
    category_cols = ["Gender", "Age", "Smoking"]

    # Preprocess by handling NaN values
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # No missing NaN vals to handle
    # For robustness if data could grow in the future, may want to handle NaN
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, category_cols),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            # use L2 regularization by setting L1 ratio to 0
            # l1_ratio=0,
            class_weight="balanced",
            random_state=42,
        ))
    ])

    return model_pipeline


def test_parameters(exasens_df: pd.DataFrame, model_pipeline: Pipeline) -> tuple[Pipeline, dict[str, Any]]:
    # test parameters
    param_to_values = {
        "clf__C": [0.01, 0.1, 0.5, 1.0, 10.0, 11.0, 12.0, 15.0],
        # 0.0 was always the best parameter, so removed this test
        # "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    param_search = GridSearchCV(
        model_pipeline,
        param_grid=param_to_values,
        scoring="roc_auc",
        refit=True,
        return_train_score=True
    )

    x = exasens_df.loc[:, ["Imaginary Part - avg", "Real Part - avg", "Gender", "Age", "Smoking"]]
    y = exasens_df.loc[:, "diagnosis_number"]
    param_search.fit(x, y)

    print("Best params:", param_search.best_params_)
    print("Best CV ROC-AUC:", param_search.best_score_)

    return (param_search.best_estimator_, param_search.cv_results_)


def viz_param_search(search_results: dict[str, Any]) -> None:
    results = pd.DataFrame(
        search_results,
        columns=[
            "param_clf__C",
            "mean_test_score",
            "mean_train_score",
            "std_test_score"
        ]
    )

    plt.semilogx(
        results["param_clf__C"],
        results["mean_test_score"],
        marker="o",
    )

    plt.xlabel("C (log scale)")
    plt.ylabel("Mean ROC-AUC")
    plt.title("Effect of Regularization Strength")
    plt.savefig("plots/model_performance/C_parameter.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.semilogx(
        results["param_clf__C"],
        results["std_test_score"],
        marker="o",
    )

    plt.xlabel("C (log scale)")
    plt.ylabel("Standard Deviation of ROC-AUC Scores")
    plt.title("Model Stability Across Folds")
    plt.savefig("plots/model_performance/model_stability.png", dpi=200, bbox_inches="tight")
    plt.close()


def performance_metrics(model: Pipeline, exasens_df: pd.DataFrame) -> None:
    x = exasens_df.loc[:, ["Imaginary Part - avg", "Real Part - avg", "Gender", "Age", "Smoking"]]
    y = exasens_df.loc[:, "diagnosis_number"]

    predictions = cross_val_predict(
        estimator=model,
        X=x,
        y=y,
        method="predict_proba",
    )

    copd_predictions = predictions[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, copd_predictions)

    threshold = 0.90
    y_pred = (copd_predictions >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    metrics = {
        "ROC-AUC": roc_auc_score(y, proba_oof),
        "Sensitivity (Recall)": recall_score(y, y_pred),
        "Precision (PPV)": precision_score(y, y_pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, y_pred),
    }

    metrics