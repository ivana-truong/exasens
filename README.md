# Identifying COPD from Exasens Dataset using Logistic Regression
Using the [exasens dataset](https://doi.org/10.24432/C5M03M) from Research Center Borstel, BioMaterialBank Nord (Borstel, Germany) to identify COPD using logistic regression with L2 regularization.

To download dependencies:
```bash
pip install -r requirements.txt
```

To create all plots and train model run:
```bash
python main.py
```

# Files

Root Directory
| File  | Summary |
| ------------- | ------------- |
| main.py  | Run functions in `eda.py` and `train_model.py`  |
| eda.py  | Functions for adding additional columns to exasens data and exploratory data analysis tables/plots. Saves new dataframe to exasens_prepped.csv |
| train_model.py  | Functions for preproccessing exasens data and training/selecting parameters/assessing a logistic regression model  |

Exasens Directory
| File  | Summary |
| ------------- | ------------- |
| exasens/Exasens.csv | Dataset downloaded from [UC Irvine ML Repo](https://doi.org/10.24432/C5M03M) |
| exasens/exasens_prepped.csv | CSV with additional columns identifying outliers and converting diagnosis to integers |

Plots Directory
| Directory  | Summary |
| ------------- | ------------- |
| plots/eda | Exploratory data analysis plots |
| plots/model_performance | Plots for model performance metrics  |