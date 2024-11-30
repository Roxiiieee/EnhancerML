import dask.dataframe as dd
import numpy as np
from joblib import load
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import gc

# Load the saved LightGBM model
model_path = 'lightgbm_model_balanced_smote_best_7.joblib'
best_lgb_model = load(model_path)
print("LightGBM model loaded successfully!")

# Define the data types for each column to avoid mismatched dtypes
dtypes = {
    'seqnames': 'object',
    'start': 'int64',
    'end': 'int64',
    'width': 'int64',
    'strand': 'object',
    'H3K18ac_1': 'float64',
    'H3K14ac_1': 'float64',
    'H2AFZ_1': 'float64',
    'H3K4me1_1': 'float64',
    'H3K4me2_1': 'float64',
    'H3K4me3_1': 'float64',
    'ATAC_signalValues': 'float64',
    'STARR_seq_binary': 'int64'
}

# Load the new data file using Dask with specified dtypes
# Using Dask to handle large datasets efficiently and avoid memory issues
new_data = dd.read_csv('DMR_method_norm_tiled_NAs.csv', dtype=dtypes)

# Specify the features to use
features_to_use = ['H3K18ac_1', 'H3K14ac_1', 'H2AFZ_1', 'H3K4me1_1', 'H3K4me2_1', 'H3K4me3_1', 'ATAC_signalValues']

# Separate features and target variable
X_new = new_data[features_to_use]
y_new = new_data['STARR_seq_binary']

# Fill NaN values with zero in the dataset
X_new_filled = X_new.fillna(0)

# Convert Dask DataFrame to Pandas DataFrame for model prediction
X_new_filled_pd = X_new_filled.compute()
y_new_pd = y_new.compute()

# Function to compute and save ROC curve data with predictions for each threshold
def save_roc_curve_data_with_predictions(model, X, y, original_data, filename):
    # Predict probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]

    # Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    # Compute the Area Under the ROC Curve (AUC)
    roc_auc = roc_auc_score(y, y_prob)

    # Create a DataFrame to save the ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })

    # Save the ROC curve data
    roc_data.to_csv(filename.replace('.csv', '_roc.csv'), index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', '_roc.csv')}")

    # Create a DataFrame to save the predictions with the required columns
    pred_data = pd.DataFrame({
        'chr': original_data['seqnames'].compute(),  
        'start': original_data['start'].compute(),
        'end': original_data['end'].compute(),
        'Actual': y,
        'Probabilities': y_prob
    })

    # Save the prediction data
    pred_data.to_csv(filename.replace('.csv', '_predictions.csv'), index=False)
    print(f"Predictions saved to {filename.replace('.csv', '_predictions.csv')}")

    return thresholds

# Attempt to free up memory
gc.collect()

# Compute and save ROC curve data for the LightGBM model on new data
save_roc_curve_data_with_predictions(best_lgb_model, X_new_filled_pd, y_new_pd, new_data, 'roc_curve_lightgbm_whole_genome_7.csv')
