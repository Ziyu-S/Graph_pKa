import os
import sys
import torch
import pickle
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch.nn.functional as F
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from Net.GNN_Grid_Search.GAT import load_training_data

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Define the model class
class GATConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout, heads):
        super(GATConv, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.pooling_function = global_mean_pool
        self.out_layer = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch
        x = data.x

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pooling_function(x, batch)
        x = self.out_layer(x)
        return x

# Load the dataset
def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    pkl_path = dataset_dir / "data_list_0.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    print(f"Loaded dataset: {pkl_path} ({len(data_list)} graphs)")
    return data_list

# Predict and save results
def predict_and_save(model, data_list, output_csv):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions = []

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)

            for i in range(len(batch.y)):
                predictions.append({
                    "PDB_ID": batch.PDB_ID[i],
                    "Residue_Number": batch.Residue_Number[i],
                    "Residue": batch.Residue_Name[i],
                    "True_pKa": batch.y[i].item(),
                    "Predicted_pKa": out[i].item()
                })

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return predictions

def inspect_checkpoint(model_path):
    """Inspect the structure of the checkpoint file."""
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    print(f"Keys in the checkpoint: {list(checkpoint.keys())}")
    return checkpoint

# Predict and save results for all models
def predict_with_all_models(model_dir, data_sets, output_dir):
    device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pth files in the directory
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))

    metrics = []  # To store MAE and RMSE for each model and dataset

    for dataset_idx, data_list in enumerate(data_sets):
        print(f"Processing Dataset {dataset_idx + 1}...")

        for model_path in model_paths:
            print(f"Loading model: {model_path}")

            # Inspect the checkpoint structure
            checkpoint = inspect_checkpoint(model_path)

            # Load model
            model = GATConv(input_dim=data_list[0].x.shape[1], hidden_channels=48, dropout=0.3, heads=6)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                print("Warning: 'model_state_dict' not found. Attempting to load the entire checkpoint.")
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            # Predict and save
            model_name = os.path.basename(model_path).replace(".pth", "")
            output_csv = os.path.join(output_dir, f"predictions_dataset_{dataset_idx + 1}_{model_name}.csv")
            predictions = predict_and_save(model, data_list, output_csv)

            # Calculate MAE and RMSE
            true_labels = [pred["True_pKa"] for pred in predictions]
            predicted_labels = [pred["Predicted_pKa"] for pred in predictions]
            mae = mean_absolute_error(true_labels, predicted_labels)
            rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))

            metrics.append({"Dataset": dataset_idx + 1, "Model": model_name, "MAE": mae, "RMSE": rmse})
            print(f"Dataset {dataset_idx + 1}, Model: {model_name}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.join(output_dir, "model_metrics_all_datasets.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")

if __name__ == "__main__":
    # Paths
    model_dir = "/home/ziyu-song/Graph_pKa/Results/GAT_Grid_Search/loss_MSELoss_h48_b16_lr0.01_d0.3_hd6"
    dataset_dir = "/home/ziyu-song/Graph_pKa/Data_1/4_Residues_W_Local_Frame/Subsets"
    output_dir = "/home/ziyu-song/Graph_pKa/Results/All_Predictions"

    # Load all datasets
    data_sets, input_dim, Residue_Type_labels = load_training_data(dataset_dir)

    # Predict with all models and datasets
    predict_with_all_models(model_dir, data_sets, output_dir)