import dask.dataframe as dd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import load
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import gc

# Define the DeepSTARR class (same as the one used for training)
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
        conv_output_size = 7  # Number of features
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

# Load the saved DeepSTARR model
model_path = 'deepstarr_model_optuna_seven_features.pth'
model_checkpoint = torch.load(model_path)
best_model = DeepSTARR(
    num_filters=model_checkpoint['best_params']['num_filters'],
    kernel_size=model_checkpoint['best_params']['kernel_size'],
    dropout_rate=model_checkpoint['best_params']['dropout_rate'],
    num_conv_layers=model_checkpoint['best_params']['num_conv_layers'],
    num_fc_layers=model_checkpoint['best_params']['num_fc_layers']
)
best_model.load_state_dict(model_checkpoint['model_state_dict'])
print("DeepSTARR model loaded successfully!")

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

# Reshape data for CNN
X_new_reshaped = X_new_filled_pd.values.reshape(-1, 1, len(features_to_use), 1)

# Convert data to PyTorch tensors
X_new_tensor = torch.tensor(X_new_reshaped, dtype=torch.float32)
y_new_tensor = torch.tensor(y_new_pd.values, dtype=torch.long)

# Create DataLoader
new_dataset = TensorDataset(X_new_tensor, y_new_tensor)
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)

# Function to compute and save ROC curve data with predictions for each threshold
def save_roc_curve_data_with_predictions(model, dataloader, original_data, filename):
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

    # Save the ROC curve data
    roc_data.to_csv(filename.replace('.csv', '_roc.csv'), index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', '_roc.csv')}")

    # Create a DataFrame to save the predictions with the required columns
    pred_data = pd.DataFrame({
        'chr': original_data['seqnames'].compute(),  
        'start': original_data['start'].compute(),
        'end': original_data['end'].compute(),
        'Actual': y_true,
        'Probabilities': y_prob
    })

    # Save the prediction data
    pred_data.to_csv(filename.replace('.csv', '_predictions.csv'), index=False)
    print(f"Predictions saved to {filename.replace('.csv', '_predictions.csv')}")

    return thresholds

# Attempt to free up memory
gc.collect()

# Compute and save ROC curve data for the DeepSTARR model on new data
save_roc_curve_data_with_predictions(best_model, new_loader, new_data, 'roc_curve_deepstarr_whole_genome_7.csv')