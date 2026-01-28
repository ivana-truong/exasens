import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_exasens_csv(filepath: str) -> dict[str, pd.DataFrame]:
    """Read in dataframe, skipping empty rows and data key"""

    exasens_df = pd.read_csv(
        "exasens/Exasens.csv",
        # Using "Float64" because it is nullable
        dtype={
            "Diagnosis": str,
            "ID": str,
            "Imaginary Part": "Float64",
            "Unnamed: 3": "Float64",
            "Real Part": "Float64",
            "Unnamed: 5": "Float64",
            "Gender": "Float64",
            "Age": "Float64",
            "Smoking": "Float64",
        },
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        # Skip empty and additional label row
        skiprows=[1, 2],
    )

    # Rename columns more clearly
    exasens_df = exasens_df.rename(
        mapper={"Imaginary Part": "Imaginary Part - min", "Unnamed: 3": "Imaginary Part - avg", "Real Part": "Real Part - min", "Unnamed: 5": "Real Part - avg"}, axis=1
    )

    # df for each diagnosis
    hc_df = exasens_df[exasens_df["Diagnosis"] == "HC"]
    asthma_df = exasens_df[exasens_df["Diagnosis"] == "Asthma"]
    infected_df = exasens_df[exasens_df["Diagnosis"] == "Infected"]
    copd_df = exasens_df[exasens_df["Diagnosis"] == "COPD"]

    return {"All Samples": exasens_df, "HC": hc_df, "Asthma": asthma_df, "Infected": infected_df, "COPD": copd_df}


def permittivity_boxplot(subset_to_df: dict[str, pd.DataFrame]) -> None:
    """Create boxplots for all subsets of exasens data for both imaginary and real parts of saliva permittivity"""
    # Saliva Permittivity - Imaginary Part (avg)
    fig, ax = plt.subplots()
    plt.tight_layout()

    imaginary_data = []
    real_data = []
    labels = []
    for diagnosis_df in [subset_to_df["HC"], subset_to_df["Asthma"], subset_to_df["Infected"], subset_to_df["COPD"]]:
        drop_nan_df = diagnosis_df.dropna(subset=["Imaginary Part - avg", "Real Part - avg"])
        imaginary_data.append(list(drop_nan_df["Imaginary Part - avg"]))
        real_data.append(list(drop_nan_df["Real Part - avg"]))
        labels.append(drop_nan_df["Diagnosis"].iloc[0])

    bp = ax.boxplot(x=imaginary_data, tick_labels=labels)
    # print([f"({flier.get_xdata()}, {flier.get_ydata()})" for flier in bp["fliers"]])
    ax.set_title("Saliva Permittivity - Imaginary")
    ax.set_ylabel("Saliva Permittivity Imaginary Part - Avg")
    ax.table(cellText=[[len(data)] for data in imaginary_data], rowLabels=labels, colLabels=["Sample count"], bbox=[0, -0.33, 0.2, 0.2])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("plots/eda/permittivity_imaginary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Saliva Permittivity - Real Part (avg)
    fig, ax = plt.subplots()
    ax.boxplot(x=real_data, tick_labels=labels)
    ax.set_title("Saliva Permittivity - Real")
    ax.set_ylabel("Saliva Permittivity Real Part - Avg")
    ax.table(cellText=[[len(data)] for data in real_data], rowLabels=labels, colLabels=["Sample count"], bbox=[0, -0.33, 0.2, 0.2])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("plots/eda/permittivity_real.png", dpi=200, bbox_inches="tight")
    plt.close()


def gender_table(subset_to_df: dict[str, pd.DataFrame]) -> None:
    """Create table for whole dataset and all subsets of sample counts for each gender and proportion of male/female"""

    # Look at gender in dataset
    # A lot more females in overall dataset, more males in COPD
    # prevalence of COPD by gender: https://doi.org/10.2147/COPD.S146390
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")

    gender_data = []
    subset_labels = []
    for subset_name, df in subset_to_df.items():
        counts = df["Gender"].value_counts()
        gender_data.append([counts.loc[1.0] + counts.loc[0.0], counts.loc[1.0], counts.loc[0.0], format(counts.loc[1.0] / counts.loc[0.0], ".4f")])
        subset_labels.append(subset_name)

    ax.table(
        cellText=gender_data,
        rowLabels=subset_labels,
        # WARNING: This is also hard-coded and should match how `gender_data` is appended to.
        # Male = 1.0, Female = 0.0
        colLabels=["Total", "Male", "Female", "Male/Female"],
        loc="center",
    )

    plt.savefig("plots/eda/gender_counts.png", dpi=1000, bbox_inches="tight")
    plt.close()


def age_hist(subset_to_df: dict[str, pd.DataFrame]) -> None:
    """Histogram of ages in whole dataset and each subset

    Args:
        subset_to_df (pd.DataFrame): dict where keys are names of dataset or subset and values are the corresponding data
    """
    fig, axs = plt.subplots(nrows=2, ncols=4)
    fig.suptitle("Age Distribution", fontsize=14)
    fig.tight_layout()

    # Create layout for one large hist on top of five smaller hists
    gs = axs[0, 1].get_gridspec()
    # remove the underlying Axes
    for ax in axs[0, 0:]:
        ax.remove()

    large_hist_ax = fig.add_subplot(gs[0, 0:])

    large_hist_ax.hist(x=subset_to_df["All Samples"]["Age"])
    large_hist_ax.axvline(
        subset_to_df["All Samples"]["Age"].mean(),
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {format(subset_to_df["All Samples"]["Age"].mean(), '.2f')}"
    )
    large_hist_ax.axvline(
        subset_to_df["All Samples"]["Age"].median(),
        color="black",
        linestyle="solid",
        linewidth=1,
        label=f"Median: {format(subset_to_df["All Samples"]["Age"].median(), '.2f')}"
    )
    large_hist_ax.set_title("All Samples")
    large_hist_ax.legend()
    axs[1, 0].hist(x=subset_to_df["HC"]["Age"])
    axs[1, 0].set_title("HC")
    axs[1, 1].hist(x=subset_to_df["Asthma"]["Age"])
    axs[1, 1].set_title("Asthma")
    axs[1, 2].hist(x=subset_to_df["Infected"]["Age"])
    axs[1, 2].set_title("Infected")
    axs[1, 3].hist(x=subset_to_df["COPD"]["Age"])
    axs[1, 3].set_title("COPD")
    axs[1, 3].axvline(
        subset_to_df["COPD"]["Age"].mean(),
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=format(f"Mean: {format(subset_to_df["COPD"]["Age"].mean(), '.2f')}")
    )
    axs[1, 3].axvline(
        subset_to_df["COPD"]["Age"].median(),
        color="black", linestyle="solid",
        linewidth=1,
        label=format(f"Median: {format(subset_to_df["COPD"]["Age"].median(), '.2f')}")
        )
    axs[1, 3].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
    )

    plt.subplots_adjust(hspace=0.4)
    plt.savefig("plots/eda/age_dist.png", dpi=200, bbox_inches="tight")
    plt.close()


def calc_outlier_bounds(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Calculate outlier thresholds based on interquartile range (IQR)

    Args:
        df (pd.DataFrame): Dataframe to calculate outlier threshold for
        column (str): which column to calculate outlier threshold of

    Returns:
        dict[str, float]: dict where key identifies upper or lower bounds and values are the outlier thresholds
    """
    # drop NaN values
    permittivity_dropna = df[column].dropna()
    # print(f"{df} {column}: {len(df[column]) - len(permittivity_dropna)} NaN values out of {len(df[column])} values")
    Q1_threshold = np.percentile(permittivity_dropna, 25)
    Q3_threshold = np.percentile(permittivity_dropna, 75)
    interquantile_range = Q3_threshold - Q1_threshold

    lower_outlier_bound = Q1_threshold - (1.5 * interquantile_range)
    upper_outlier_bound = Q3_threshold + (1.5 * interquantile_range)

    # print(f"lower bound {lower_outlier_bound}: \n {lower_outliers}")
    # print(f"upper bound {upper_outlier_bound}: \n {upper_outliers}")

    return {"lower_bound": lower_outlier_bound, "upper_bound": upper_outlier_bound}


def is_outlier(real_value: float, imaginary_value: float, real_outlier_bounds: dict[str, float], imaginary_outlier_bounds: dict[str, float]) -> dict[str, bool]:
    """Compare a saliva permittivity measurement again outlier thresholds.

    Args:
        real_value (float): real saliva permittivity value
        imaginary_value (float): imaginary saliva permittivity value
        real_outlier_bounds (dict[str, float]): dict of real outlier thresholds
        imaginary_outlier_bounds (dict[str, float]): dict of imaginary outlier thresholds

    Returns:
        dict[str, bool]: dict where keys are "real" and "imaginary" and values are booleans indicating whether the value is an outlier
    """
    if (real_value > real_outlier_bounds["upper_bound"]) or (real_value < real_outlier_bounds["lower_bound"]):
        real_outlier = True
    else:
        real_outlier = False

    if (imaginary_value > imaginary_outlier_bounds["upper_bound"]) or (imaginary_value < imaginary_outlier_bounds["lower_bound"]):
        imaginary_outlier = True
    else:
        imaginary_outlier = False

    return {"real": real_outlier, "imaginary": imaginary_outlier}


def add_outlier_bound_in_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Find outliers in dataframe and add columns identifying real and imaginary outliers

    Args:
        df (pd.DataFrame): DataFrame to calculate outliers of

    Returns:
        pd.DataFrame: `df` with additional columns identifying real and imaginary outliers
    """
    real_outlier_bounds: dict[str, float] = calc_outlier_bounds(df=df, column="Real Part - avg")
    imaginary_outlier_bounds: dict[str, float] = calc_outlier_bounds(df=df, column="Imaginary Part - avg")
    # print(f"real outliers bounds: {real_outlier_bounds}")
    # print(f"imaginary outliers bounds: {imaginary_outlier_bounds}")

    # Mark as outlier if real or imaginary permittivity outlier
    real_outliers: list[bool] = []
    imaginary_outliers: list[bool] = []
    for row_num in range(len(df)):
        # Assuming here that both will be missing if one is missing
        if pd.isna(df["Real Part - avg"].iloc[row_num]) and pd.isna(df["Imaginary Part - avg"].iloc[row_num]):
            real_outliers.append(False)
            imaginary_outliers.append(False)
        else:
            col_to_outlier_bool = is_outlier(
                real_outlier_bounds=real_outlier_bounds,
                imaginary_outlier_bounds=imaginary_outlier_bounds,
                real_value=df["Real Part - avg"].iloc[row_num],
                imaginary_value=df["Imaginary Part - avg"].iloc[row_num]
            )
            # If either real or imaginary outlier, append to respective lists
            real_outliers.append(col_to_outlier_bool["real"])
            imaginary_outliers.append(col_to_outlier_bool["imaginary"])

    df["real_outlier"] = real_outliers
    df["imaginary_outlier"] = imaginary_outliers

    return df
