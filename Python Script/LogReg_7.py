import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from joblib import dump
import matplotlib.pyplot as plt
from scipy.stats import uniform
# Load training data
train_data = pd.read_csv('H9/DMR_method_input_NAs.csv')
# Load test data (holdout set)
test_data = pd.read_csv('H9/DMR_method_holdout_NAs.csv')
# Specify the features to use
features_to_use = ['H3K18ac_1', 'H3K14ac_1', 'H2AFZ_1', 'H3K4me1_1', 'H3K4me2_1', 'H3K4me3_1', 'ATAC_signalValues']

# Separate features and the target variable
X_train = train_data[features_to_use]
y_train = train_data['STARR_seq_binary']
X_test = test_data[features_to_use]
y_test = test_data['STARR_seq_binary']
# Drop the specified columns
columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)
# Fill NaN values with zero in the datasets
X_train_filled = X_train.fillna(0)
X_test_filled = X_test.fillna(0)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filled, y_train)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'C': uniform(0.01, 10),  # Adjusted range for regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # Penalty types
    'solver': ['saga'],  # Solver to handle L1 and elasticnet
    'l1_ratio': uniform(0, 1)  # For elasticnet
}
# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=logreg_model, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=-1, verbose=10, scoring='roc_auc')
# Fit RandomizedSearchCV to the training data
random_search.fit(X_train_balanced, y_train_balanced)
# Best model from RandomizedSearchCV
best_logreg_model = random_search.best_estimator_
print(f"Best parameters found: {random_search.best_params_}")
# Save the best model
dump(best_logreg_model, 'logistic_regression_model_balanced_smote_7.joblib')
print("Best Logistic Regression model saved successfully!")
def save_roc_curve_data_with_predictions(model, X, y, original_data, filename):
    """
    Computes and saves the ROC curve data and predictions for a given model and dataset.

    Parameters:
    - model: Trained model to be evaluated.
    - X: Feature matrix.
    - y: True labels.
    - original_data: Original dataframe containing 'chr', 'start', and 'end' columns.
    - filename: Base filename for saving the outputs.

    Outputs:
    - Saves ROC curve data to a CSV file.
    - Saves prediction data to a CSV file.
    """
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

    # Create a DataFrame to save the predictions
    pred_data = original_data[['chr', 'start', 'end']].copy()
    pred_data['Actual'] = y
    pred_data['Probabilities'] = y_prob

    # Ensure pred_data only contains the specified columns
    pred_data = pred_data[['chr', 'start', 'end', 'Actual', 'Probabilities']]

    # Save the ROC curve data
    roc_data.to_csv(filename.replace('.csv', '_roc.csv'), index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', '_roc.csv')}")

    # Save the prediction data
    pred_data.to_csv(filename.replace('.csv', '_predictions.csv'), index=False)
    print(f"Predictions saved to {filename.replace('.csv', '_predictions.csv')}")

    return thresholds
# Compute and save ROC curve data for the best Logistic Regression model on test data
save_roc_curve_data_with_predictions(best_logreg_model, X_test_filled, y_test, 'roc_curve_log_reg_test_7.csv')
# Plot the ROC curve for the test data
y_prob_test = best_logreg_model.predict_proba(X_test_filled)[:, 1]
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob_test)
roc_auc_test = roc_auc_score(y_test, y_prob_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Data')
plt.legend(loc='lower right')
plt.show()
# Find the optimal threshold using Youden's J statistic
youden_index = tpr_test - fpr_test
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds_test[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold:.2f}')
# Apply the optimal threshold to make new class predictions
y_pred_optimal = (y_prob_test >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Accuracy of the Logistic Regression model on holdout data with optimal threshold: {accuracy_optimal:.2f}')
# Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_test, y_pred_optimal, target_names=['0', '1'], digits=2)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)