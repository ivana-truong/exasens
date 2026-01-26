from eda import read_exasens_csv, permittivity_boxplot, gender_table, age_hist, add_outlier_bound_in_df
from train_model import build_pipeline, test_parameters, viz_param_search
import pandas as pd

if __name__ == "__main__":

    # Load in dataset as dict where key is a title and value is the whole dataset or a subset
    subset_to_df = read_exasens_csv("exasens/Exasens.csv")

    # Exploratory Data Analysis
    # print(subset_to_df["All Samples"]["Diagnosis"].value_counts())

    # # Visualize data
    # permittivity_boxplot(subset_to_df)
    # gender_table(subset_to_df)
    # age_hist(subset_to_df)

    # # Mark permittivity outliers for each subset
    # for key, diagnosis_df in subset_to_df.items():
    #     if key == "All Samples":
    #         continue
    #     subset_to_df[key] = add_outlier_bound_in_df(df=diagnosis_df)

    # # Optional: Write outliers dfs to csv in `exasens` folder
    # for key, diagnosis_df in subset_to_df.items():
    #     if key == "All Samples":
    #         continue
    #     subset_to_df[key].to_csv(f"exasens/{key}_outliers.csv")

    # Encode diagnosis as integers for training
    # Categorized as 1 for COPD and 0 for not COPD
    # diagnosis_to_integer = {
    #     "HC": 0,
    #     "Asthma": 0,
    #     "Infected": 0,
    #     "COPD": 1
    # }
    # for key, diagnosis_df in subset_to_df.items():
    #     if key == "All Samples":
    #         continue
    #     subset_to_df[key]["diagnosis_number"] = diagnosis_to_integer[key]

    # # Merge subsets into one dataset and export
    # exasens_prepped = pd.concat([subset_to_df["HC"], subset_to_df["Asthma"], subset_to_df["Infected"], subset_to_df["COPD"]], axis="rows")
    # exasens_prepped.to_csv("./exasens/exasens_prepped.csv", index=False)

    exasens_prepped = pd.read_csv("exasens/exasens_prepped.csv")

    model_pipeline = build_pipeline()
    # best_model_pipeline, test_results = test_parameters(exasens_df=exasens_prepped, model_pipeline=model_pipeline)

    # Removing outliers decreased ROC-AUC?
    exasens_df_no_outliers = exasens_prepped[exasens_prepped["is_outlier"] == False]
    best_model_pipeline, test_results = test_parameters(exasens_df=exasens_df_no_outliers, model_pipeline=model_pipeline)
    viz_param_search(search_results=test_results)

    # TODO: what to do about gender imbalance in data?
