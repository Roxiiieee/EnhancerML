import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report
import optuna
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
# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filled, y_train)
# Reshape data for CNN
num_features = X_train_balanced.shape[1]
X_train_reshaped = X_train_balanced.values.reshape(-1, 1, num_features, 1)
X_test_reshaped = X_test_filled.values.reshape(-1, 1, num_features, 1)
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class DeepSTARR(nn.Module):
    def __init__(self, num_filters, kernel_size, dropout_rate, num_conv_layers, num_fc_layers):
        super(DeepSTARR, self).__init__()
        layers = []
        in_channels = 1

        # Add convolutional layers
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(1, 0)))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            in_channels = num_filters
        
        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output size after conv and pooling layers
        conv_output_size = num_features
        for _ in range(num_conv_layers):
            conv_output_size = (conv_output_size + 2 * 1 - kernel_size) // 1 + 1  # After conv
            conv_output_size = (conv_output_size - 2) // 2 + 1  # After pool
        self.fc_input_dim = num_filters * conv_output_size * 1  # Calculate fc input dim

        # Add fully connected layers
        fc_layers = []
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.fc_input_dim if len(fc_layers) == 0 else 256, 256))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
        
        fc_layers.append(nn.Linear(256, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return torch.sigmoid(x)

def objective(trial):
    num_filters = trial.suggest_categorical('num_filters', [32, 64])
    kernel_size = trial.suggest_int('kernel_size', 1, 2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    num_conv_layers = trial.suggest_int('num_conv_layers', 2, 2)
    num_fc_layers = trial.suggest_int('num_fc_layers', 2, 3)

    model = DeepSTARR(num_filters, kernel_size, dropout_rate, num_conv_layers, num_fc_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = 10
    early_stop_patience = 3
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_running_loss += loss.item() * inputs.size(0)
        val_loss = val_running_loss / len(test_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            break

    return best_loss
# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
print('Best hyperparameters found: ', best_params)
# Train the final model with the best hyperparameters
best_model = DeepSTARR(best_params['num_filters'], best_params['kernel_size'], best_params['dropout_rate'], best_params['num_conv_layers'], best_params['num_fc_layers'])
criterion = nn.BCELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
n_epochs = 20
train_losses = []
test_losses = []
for epoch in range(n_epochs):
    best_model.train()
    running_loss = 0.0
    batch_count = 0  # For additional batch-level logging
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = best_model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        # Batch-level logging for debugging
        batch_count += 1
        if batch_count % 10 == 0:  # Log every 10 batches
            print(f'Epoch {epoch + 1}, Batch {batch_count}: Batch loss: {loss.item()}')

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validate the model
    best_model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = best_model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            running_loss += loss.item() * inputs.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f'Epoch {epoch + 1}/{n_epochs}.. Train loss: {train_loss:.4f}.. Test loss: {test_loss:.4f}')
# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the model
best_model.eval()
y_true = []
y_prob = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_model(inputs).squeeze()
        probabilities = torch.sigmoid(outputs)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)
# Save the model
torch.save({
    'model_state_dict': best_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_params': best_params
}, 'deepstarr_model_optuna_all_features.pth')
print("DeepSTARR model saved successfully!")
def save_roc_curve_data_with_predictions(model, dataloader, original_data, filename):
    """
    Computes and saves the ROC curve data and predictions for a given model and dataset.

    Parameters:
    - model: Trained model to be evaluated.
    - dataloader: DataLoader for the dataset to evaluate.
    - original_data: Original dataframe containing 'chr', 'start', and 'end' columns.
    - filename: Base filename for saving the outputs.

    Outputs:
    - Saves ROC curve data to a CSV file.
    - Saves prediction data to a CSV file.
    """
    model.eval()
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).squeeze()
            probabilities = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Compute the Area Under the ROC Curve (AUC)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Create a DataFrame to save the ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })

    # Create a DataFrame to save the predictions
    pred_data = original_data[['chr', 'start', 'end']].copy()
    pred_data['Actual'] = y_true
    pred_data['Probabilities'] = y_prob

    # Save the ROC curve data
    roc_data.to_csv(filename.replace('.csv', '_roc.csv'), index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', '_roc.csv')}")

    # Save the prediction data
    pred_data.to_csv(filename.replace('.csv', '_predictions.csv'), index=False)
    print(f"Predictions saved to {filename.replace('.csv', '_predictions.csv')}")

    return thresholds

# Compute and save ROC curve data for the trained DeepSTARR model on test data
thresholds = save_roc_curve_data_with_predictions(best_model, test_loader, 'roc_curve_deepstarr_test_all_features_optuna.csv')
# Plot the ROC curve for the test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_true, y_prob)
roc_auc_test = roc_auc_score(y_true, y_prob)
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
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
print(f'Accuracy of the DeepSTARR model on holdout data with optimal threshold: {accuracy_optimal:.2f}')
# Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_true, y_pred_optimal, target_names=['0', '1'], digits=2)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)