import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load the data
expr = pd.read_csv("../../data/expresion_matrix.csv")
sens = pd.read_csv("../../data/sensitivity_matrix_Activity_Area.csv", index_col=0)

# Normalize X
X = np.array(expr)
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

rep = 20
lr_combined_result_df = pd.DataFrame()
svm_combined_result_df = pd.DataFrame()
rf_combined_result_df = pd.DataFrame()


def process_iteration(X, y, model_class, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    model = RandomForestRegressor(n_estimators=10) if model_class == RandomForestRegressor else model_class()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mse = np.mean((y_test - y_hat) ** 2)
    cor = np.corrcoef(y_test, y_hat)[0, 1]
    return mse, cor


for med in sens.columns:
    print(f"Processing: {med}")  # Debugging statement
    y = np.array(sens[med])
    print(f"Initial y size: {y.size}")  # Debugging statement

    X_filtered = X_norm[~np.isnan(y)]
    y_filtered = y[~np.isnan(y)]

    print(f"Filtered X size: {X_filtered.shape}, Filtered y size: {y_filtered.size}")  # Debugging statement

    results = {
        "lr": [],
        "svm": [],
        "rf": []
    }

    for model_class, key in zip([LinearRegression, SVR, RandomForestRegressor], ["lr", "svm", "rf"]):
        print(f"Training model: {key}")  # Debugging statement
        results[key] = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_iteration)(X_filtered, y_filtered, model_class, i) for i in tqdm(range(rep))
        )
        print(f"Done ML: {key}")  # Debugging statement

    # Combine results into DataFrames
    lr_med_df = pd.DataFrame(results["lr"], columns=[f"{med}_mse", f"{med}_cor"])
    svm_med_df = pd.DataFrame(results["svm"], columns=[f"{med}_mse", f"{med}_cor"])
    rf_med_df = pd.DataFrame(results["rf"], columns=[f"{med}_mse", f"{med}_cor"])

    lr_combined_result_df = pd.concat([lr_combined_result_df, lr_med_df], axis=1)
    svm_combined_result_df = pd.concat([svm_combined_result_df, svm_med_df], axis=1)
    rf_combined_result_df = pd.concat([rf_combined_result_df, rf_med_df], axis=1)

# Save the combined result DataFrame to a CSV file
lr_combined_result_df.to_csv('result_data_lr.csv', index=False)
svm_combined_result_df.to_csv('result_data_svm.csv', index=False)
rf_combined_result_df.to_csv('result_data_rf.csv', index=False)


# Calculate row means
def calculate_av_per_drug(combined_result_df, sens_columns):
    av_per_drug = combined_result_df.mean()
    av_per_drug_mse = av_per_drug.iloc[::2]
    av_per_drug_mse.index = sens_columns
    av_per_drug_cor = av_per_drug.iloc[1::2]
    av_per_drug_cor.index = sens_columns
    return av_per_drug_mse, av_per_drug_cor


lr_av_per_drug_mse, lr_av_per_drug_cor = calculate_av_per_drug(lr_combined_result_df, sens.columns)
svm_av_per_drug_mse, svm_av_per_drug_cor = calculate_av_per_drug(svm_combined_result_df, sens.columns)
rf_av_per_drug_mse, rf_av_per_drug_cor = calculate_av_per_drug(rf_combined_result_df, sens.columns)

# Plot correlations per drug for LR, SVM, and RF
fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size
ax.plot(lr_av_per_drug_cor, label='Linear Regression')
ax.plot(svm_av_per_drug_cor, label='SVM')
ax.plot(rf_av_per_drug_cor, label='Random Forest')
plt.xticks(rotation=45, ha='right')  # Tilt labels
plt.legend()
plt.tight_layout()  # Adjust layout to prevent labels from being cut off
plt.savefig('correlations_per_drug.png', dpi=300)  # Increase resolution for presentation

# Plot average MSE and COR values
data = {
    'Model': ['LR', 'SVM', 'RF'],
    'MSE': [lr_av_per_drug_mse.mean(), svm_av_per_drug_mse.mean(), rf_av_per_drug_mse.mean()],
    'COR': [lr_av_per_drug_cor.mean(), svm_av_per_drug_cor.mean(), rf_av_per_drug_cor.mean()]
}

df = pd.DataFrame(data)

# Set up the bar plot
fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size

# Plot MSE
df.plot(kind='bar', x='Model', y='MSE', color=['blue', 'green', 'red'], ax=ax, position=0.5, width=0.4)
# Plot COR with a secondary y-axis
df.plot(kind='bar', x='Model', y='COR', color=['skyblue', 'lightgreen', 'lightcoral'], ax=ax, position=-0.5, width=0.4,
        secondary_y=True)

# Customize the plot
ax.set_title('Comparison of MSE and COR for LR, SVM, and RF')
ax.set_ylabel('MSE')
ax.right_ax.set_ylabel('COR')
ax.set_xticklabels(df['Model'], rotation=45, ha='right')  # Tilt labels
plt.tight_layout()  # Adjust layout to prevent labels from being cut off
plt.savefig('mse_cor_comparison.png', dpi=300)  # Increase resolution for presentation
