import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    """ 1) Read in dataframe, skipping empty rows and data key """

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
            "Gender":"Float64",
            "Age": "Float64",
            "Smoking": "Float64",
        },
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        # Skip empty and additional label row
        skiprows=[1, 2],
    )

    # Rename columns more clearly
    exasens_df = exasens_df.rename(
        mapper={
            "Imaginary Part": "Imaginary Part - min",
            "Unnamed: 3": "Imaginary Part - avg",
            "Real Part": "Real Part - min",
            "Unnamed: 5": "Real Part - avg"
        },
        axis=1
    )

    # df for each diagnosis
    hc_df = exasens_df[exasens_df["Diagnosis"] == "HC"]
    asthma_df = exasens_df[exasens_df["Diagnosis"] == "Asthma"]
    infected_df = exasens_df[exasens_df["Diagnosis"] == "Infected"]
    copd_df = exasens_df[exasens_df["Diagnosis"] == "COPD"]


    """Exploratory Data analysis"""

    # exasens_df["Diagnosis"].value_counts()

    # Visualize data and identify outliers

    # Saliva Permittivity - Imaginary Part (avg)
    fig, ax = plt.subplots()
    imaginary_data = []
    real_data = []
    labels = []
    for diagnosis_df in [hc_df, asthma_df, infected_df, copd_df]:
        drop_nan_df = diagnosis_df.dropna(subset=["Imaginary Part - avg"])
        # NOTE: lots of missing values for "Imaginary Part - avg"
        # healthy: 160 -> 40 values

        imaginary_data.append(list(drop_nan_df["Imaginary Part - avg"]))
        real_data.append(list(drop_nan_df["Real Part - avg"]))
        labels.append(drop_nan_df["Diagnosis"].iloc[0])


    box_plot = ax.boxplot(x = imaginary_data, tick_labels=labels)
    ax.set_ylabel("Saliva Permittivity Imaginary Part - Avg")
    plt.savefig("plots/permittivity_imaginary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Saliva Permittivity - Real Part (avg)
    fig, ax = plt.subplots()
    box_plot = ax.boxplot(x = real_data, tick_labels=labels)
    ax.set_ylabel("Saliva Permittivity Real Part - Avg")
    plt.savefig("plots/permittivity_real.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Look at gender in dataset
    # A lot more females in overall dataset, more males in COPD
    # prevalence of COPD by gender: https://doi.org/10.2147/COPD.S146390
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')

    gender_data = []
    for df in [exasens_df, hc_df, asthma_df, infected_df, copd_df]:
        counts = df["Gender"].value_counts()
        gender_data.append([counts.loc[1.0], counts.loc[0.0], format(counts.loc[1.0]/counts.loc[0.0], ".4f")])


    ax.table(
        cellText=gender_data,
        # WARNING: This is hard-coded and should match loop above
        rowLabels=["All samples", "HC", "Asthma", "Infected", "COPD"],
        # WARNING: This is also hard-coded and should match how `gender_data` is appended to. 
        # Male = 1.0, Female = 0.0
        colLabels=["Male", "Female", "Male/Female"],
        loc="center"
    )
    
    plt.savefig("plots/gender_counts.png", dpi=200, bbox_inches="tight")
    plt.close()


    # Age
    fig, axs = plt.subplots(nrows=2, ncols=4)
    fig.suptitle(f'Age Distribution', fontsize = 14)
    fig.tight_layout()

    # Create layout for one large hist on top of five smaller hists
    gs = axs[0, 1].get_gridspec()
    # remove the underlying Axes
    for ax in axs[0, 0:]:
        ax.remove()
    
    large_hist_ax = fig.add_subplot(gs[0,0:])

    large_hist_ax.hist(x=exasens_df["Age"])
    large_hist_ax.axvline(exasens_df["Age"].mean(), color='black', linestyle='dashed', linewidth=1, label=f"Mean: {format(exasens_df["Age"].mean(), '.2f')}")
    large_hist_ax.axvline(exasens_df["Age"].median(), color='black', linestyle='solid', linewidth=1, label=f"Median: {format(exasens_df["Age"].median(), '.2f')}")
    large_hist_ax.set_title("All Samples")
    large_hist_ax.legend()
    axs[1, 0].hist(x=hc_df["Age"])
    axs[1, 0].set_title("HC")
    axs[1, 1].hist(x=asthma_df["Age"])
    axs[1, 1].set_title("Asthma")
    axs[1, 2].hist(x=infected_df["Age"])
    axs[1, 2].set_title("Infected")
    axs[1,3].hist(x=copd_df["Age"])
    axs[1,3].set_title("COPD")
    axs[1,3].axvline(copd_df["Age"].mean(), color='black', linestyle='dashed', linewidth=1, label=format(f"Mean: {format(copd_df["Age"].mean(), '.2f')}"))
    axs[1,3].axvline(copd_df["Age"].median(), color='black', linestyle='solid', linewidth=1, label=format(f"Median: {format(copd_df["Age"].median(), '.2f')}"))
    axs[1,3].legend(fontsize= 7, handlelength = 0.5, loc="upper left")
    
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("plots/age_dist.png", dpi=200, bbox_inches="tight")
    plt.close()
