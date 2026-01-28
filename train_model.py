from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    average_precision_score
)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any


def build_pipeline() -> Pipeline:
    """Build preprocess and model pipeline

    Returns:
        Pipeline: model pipeline
    """
    # Identify numerical vs categorical cols
    num_cols = ["Imaginary Part - avg", "Real Part - avg", "Age"]
    category_cols = ["Gender", "Smoking"]

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
            l1_ratio=0,
            # If testing Elasticnet, need to use different solver
            # solver="saga",
            class_weight="balanced",
            random_state=42,
        ))
    ])

    return model_pipeline


def test_parameters(x: pd.DataFrame, y: pd.Series, model_pipeline: Pipeline) -> tuple[Pipeline, dict[str, Any]]:
    """Test different C values using cross validation

    Args:
        x (pd.DataFrame): test x value
        y (pd.Series): test y values
        model_pipeline (Pipeline): model pipeline that preprocesses and fits data

    Returns:
        tuple[Pipeline, dict[str, Any]]: returns C value with highest ROC-AUC and a test result dictionary
    """
    # test parameters
    param_to_values = {
        "clf__C": [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 11.0, 12.0, 15.0],
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

    param_search.fit(x, y)

    # print(param_search.cv_results_)
    # print("Best params:", param_search.best_params_)
    # print("Highest ROC-AUC:", param_search.best_score_)

    return (param_search.best_estimator_, param_search.cv_results_)


def viz_param_search(search_results: dict[str, Any]) -> None:
    """Visualize ROC-AUC with different C values and model stability across folds

    Args:
        search_results (dict[str, Any]): test results dictionary
    """
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


def performance_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plot precision and recall and create table with various performance metrics.

    Args:
        model (Pipeline): model with best parameters
        x_test (pd.DataFrame): x values for test data
        y_test (pd.Series): y values for test data
    """
    predictions = cross_val_predict(
        estimator=model,
        X=x_test,
        y=y_test,
        method="predict_proba",
    )

    copd_predictions = predictions[:, 1]

    fig, ax = plt.subplots()

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        copd_predictions,
        pos_label=1
    )
    # print(f"recall: {recall}")
    # print(f"precision: {precision}")
    # print(thresholds)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"LR Model (AP = {format(average_precision_score(y_test, copd_predictions, pos_label=1), ".2f")})")
    ax.plot([0, 1.0], [0.2, 0.2], color="black", linestyle="dashed", label="Chance (AP = 0.20)")
    ax.set_title("Precision-Recall curve")
    ax.set_ylabel("Precision (Tp / (Tp + Fp))")
    ax.set_xlabel("Recall (Tp / (Tp + Fn))")
    ax.set_ylim(top=1.0)
    plt.legend()
    plt.savefig("plots/model_performance/precision_recall_curve.png", dpi=500, bbox_inches="tight")

    # Find thresholds that match certain recall and precision
    target_thresholds = []
    for i in range(len(thresholds)):
        if recall[i] >= 0.80 and precision[i] >= 0.65:
            target_thresholds.append(thresholds[i])
        else:
            continue
    # print(target_thresholds)

    # Set last matching threshold as threshold to compare scores against
    threshold = target_thresholds[-1]
    y_pred = (copd_predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # exasens_df["prediction"] = y_pred
    # exasens_df.to_csv("./exasens/exasens_predictions.csv", index=False)

    metrics = {
        "Accuracy": (tp + tn) / (tp + tn + fn + fp),
        "Sensitivity/Recall": tp / (tp + fn),
        "Precision": tp / (tp + fp),
        "Specificity": float(tn / (tn + fp)),
        "F1": f1_score(y_test, y_pred),
    }
    # print(metrics)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.table(
        cellText=[[format(val, ".2f")] for val in metrics.values()],
        rowLabels=list(metrics.keys()),
        colWidths=[0.2, 0.8],
        loc="center",
    )
    plt.savefig("plots/model_performance/metrics_table.png", dpi=1000, bbox_inches="tight")
    plt.close()
