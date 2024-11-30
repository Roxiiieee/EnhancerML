import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from joblib import dump
import matplotlib.pyplot as plt
# Load training data
train_data = pd.read_csv('H9/DMR_method_input_NAs.csv')
# Load test data (holdout set)
test_data = pd.read_csv('H9/DMR_method_holdout_NAs.csv')
# Separate features and the target variable
X_train = train_data.drop('STARR_seq_binary', axis=1)
y_train = train_data['STARR_seq_binary']
X_test = test_data.drop('STARR_seq_binary', axis=1)
y_test = test_data['STARR_seq_binary']
# Drop the specified columns
columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
X_train_dropped = X_train.drop(columns=columns_to_drop)
X_test_dropped = X_test.drop(columns=columns_to_drop)
# Fill NaN values with zero in the datasets
X_train_filled = X_train_dropped.fillna(0)
X_test_filled = X_test_dropped.fillna(0)
# Normalize the data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_filled)
X_test_normalized = scaler.transform(X_test_filled)
# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_normalized, y_train)
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=10, scoring='accuracy')
# Fit GridSearchCV to the training data
grid_search.fit(X_train_balanced, y_train_balanced)
# Best model from GridSearchCV
best_rf_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
# Save the best model
dump(best_rf_model, 'random_forest_model_balanced_smote_best_all_features.joblib')
print("Best Random Forest model saved successfully!")
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

    # Save the ROC curve data
    roc_data.to_csv(filename.replace('.csv', '_roc.csv'), index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', '_roc.csv')}")

    # Save the prediction data
    pred_data.to_csv(filename.replace('.csv', '_predictions.csv'), index=False)
    print(f"Predictions saved to {filename.replace('.csv', '_predictions.csv')}")

    return thresholds
# Compute and save ROC curve data for the best Random Forest model on test data
save_roc_curve_data_with_predictions(best_rf_model, X_test_normalized, y_test, 'roc_curve_random_forest_test_all_features.csv')
# Plot the ROC curve for the test data
y_prob_test = best_rf_model.predict_proba(X_test_normalized)[:, 1]
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
# Find the optimal threshold
optimal_idx = np.argmax(tpr_test - fpr_test)
optimal_threshold = thresholds_test[optimal_idx]
# Apply the optimal threshold to make new class predictions
y_pred_optimal = (y_prob_test >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Accuracy of the Random Forest model on holdout data with optimal threshold: {accuracy_optimal}')
# Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_test, y_pred_optimal, target_names=['0', '1'], digits=2)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)