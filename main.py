from eda import read_exasens_csv, permittivity_boxplot, gender_table, age_hist, add_outlier_bound_in_df
from train_model import build_pipeline, test_parameters, viz_param_search, performance_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Load in dataset as dict where key is a title and value is the whole dataset or a subset
    subset_to_df = read_exasens_csv("exasens/Exasens.csv")

    # Visualize data
    permittivity_boxplot(subset_to_df)
    gender_table(subset_to_df)
    age_hist(subset_to_df)

    # Mark permittivity outliers for each subset
    for key, diagnosis_df in subset_to_df.items():
        if key == "All Samples":
            continue
        subset_to_df[key] = add_outlier_bound_in_df(df=diagnosis_df)

    # Encode diagnosis as integers for training
    # Categorized as 1 for COPD and 0 for not COPD
    diagnosis_to_integer = {
        "HC": 0,
        "Asthma": 0,
        "Infected": 0,
        "COPD": 1
    }
    for key, diagnosis_df in subset_to_df.items():
        if key == "All Samples":
            continue
        subset_to_df[key]["diagnosis_number"] = diagnosis_to_integer[key]

    # Merge subsets into one dataset and export
    exasens_prepped = pd.concat([subset_to_df["HC"], subset_to_df["Asthma"], subset_to_df["Infected"], subset_to_df["COPD"]], axis="rows")
    exasens_prepped.to_csv("./exasens/exasens_prepped.csv", index=False)

    # exasens_prepped = pd.read_csv("exasens/exasens_prepped.csv")

    # Remove outlier values so they will be imputed with median values
    exasens_df_no_outliers = exasens_prepped.copy()
    exasens_df_no_outliers["Imaginary Part - avg"] = np.where(
        # Condition
        exasens_prepped["imaginary_outlier"] == True,
        # if True
        np.nan,
        # Else
        exasens_df_no_outliers["Imaginary Part - avg"],
    )
    exasens_df_no_outliers["Real Part - avg"] = np.where(
        # Condition
        exasens_prepped["real_outlier"] == True,
        # if True
        np.nan,
        # Else
        exasens_df_no_outliers["Real Part - avg"],
    )

    x = exasens_df_no_outliers.loc[:, ["Imaginary Part - avg", "Real Part - avg", "Gender", "Age", "Smoking"]]
    y = exasens_df_no_outliers.loc[:, "diagnosis_number"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # Build preprocess pipeline and test optimal parameters
    model_pipeline = build_pipeline()
    best_model_pipeline, test_results = test_parameters(x_train, y_train, model_pipeline=model_pipeline)

    # Visualize model performance
    viz_param_search(search_results=test_results)
    performance_metrics(model=best_model_pipeline, x_test=x_test, y_test=y_test)
