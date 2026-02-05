import os
import sys
import glob
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main(base_dir):
    base_dir = Path(base_dir)  

    # Directories for adjacency matrices and node feature sets
    adjacency_matrix_dir = base_dir / 'Features/Adjacency_Matrices/With_Self_Loop/'
    node_feature_dirs = [
        base_dir / 'Features/Node_Feature_Vectors' / '7',
        base_dir / 'Features/Node_Feature_Vectors' / '8',
        base_dir / 'Features/Node_Feature_Vectors' / '9',
        base_dir / 'Features/Node_Feature_Vectors' / '10',
        base_dir / 'Features/Node_Feature_Vectors' / '11'
    ]

    # Output directory for saving datasets as PKL
    output_dir = os.path.join(base_dir, "Features/Node_Feature_Vectors/Prediction_Datasets")
    os.makedirs(output_dir, exist_ok=True)

    for idx, node_features_dir in enumerate(node_feature_dirs):
        print(f"\nProcessing node features from: {node_features_dir}")
        data_list = []

        for filename in os.listdir(adjacency_matrix_dir):
            if filename.endswith('.csv'):
                print(f"Processing adjacency matrix: {filename}")
                adjacency_matrix = pd.read_csv(
                    os.path.join(adjacency_matrix_dir, filename), header=0, index_col=0).values
                print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
                adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.int)

                features_filename = filename.replace('_adjacency.csv', '.csv')
                node_features_path = os.path.join(node_features_dir, features_filename)

                if not os.path.exists(node_features_path):
                    print(f"Warning: {features_filename} not found in {node_features_dir}. Skipping...")
                    continue

                print(f"Processing node features: {features_filename}")
                base_name = features_filename.replace('.csv', '')
                pdb_id, residue_info = base_name.split('_')
                residue_number, residue_name = residue_info.split('.')

                node_features = pd.read_csv(node_features_path, header=0)
                print(f"Node features shape: {node_features.shape}")
                if 'atom_label' not in node_features:
                    print("Missing Atom columns in node features. Skipping...")
                    continue

                # Make pKa labels optional for feature vectors without experimental values
                if 'Expt. pKa' in node_features.columns:
                    pKa_labels_tensor = torch.tensor([node_features['Expt. pKa'].values[0]], dtype=torch.float)
                else:
                    pKa_labels_tensor = None

                atom_label_encoded = F.one_hot(torch.tensor(node_features['atom_label'].values, dtype=torch.long), num_classes=9).float()
                one_hot_cols = node_features.filter(like='Residue Name_').values
                residue_label = np.argmax(one_hot_cols, axis=1)[0]

                features_tensor = torch.tensor(
                    node_features.drop(columns=['Expt. pKa', 'atom_label']).values, dtype=torch.float
                )
                features_tensor = torch.cat([features_tensor, atom_label_encoded], dim=1)

                edge_index = adjacency_tensor.nonzero(as_tuple=True)
                edge_index = torch.stack(edge_index, dim=0)

                data = Data(
                    x=features_tensor,
                    edge_index=edge_index,
                    y=pKa_labels_tensor,
                    residue_label=residue_label
                )
                data.PDB_ID = pdb_id
                data.Residue_Number = residue_number
                data.Residue_Name = residue_name

                data_list.append(data)

        print(f"Generated {len(data_list)} graphs for node features directory {node_features_dir}")
        pkl_path = os.path.join(output_dir, f'data_list_{idx}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved dataset {idx} with {len(data_list)} graphs to {pkl_path}")


# Define the model class for predictions
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


def inspect_checkpoint(model_path):
    """Inspect the structure of the checkpoint file."""
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    print(f"Keys in the checkpoint: {list(checkpoint.keys())}")
    return checkpoint


def predict_and_save(model, data_list, output_csv):
    """Predict and save results with optional labels."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions = []

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)

            for i in range(len(out)):
                pred_dict = {
                    "PDB_ID": batch.PDB_ID[i],
                    "Residue_Number": batch.Residue_Number[i],
                    "Residue": batch.Residue_Name[i],
                    "Predicted_pKa": out[i].item()
                }
                # Add True_pKa only if it exists (optional labels)
                if batch.y is not None:
                    pred_dict["True_pKa"] = batch.y[i].item()
                predictions.append(pred_dict)

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return predictions


def predict_unlabeled_data(model, data_list, output_csv):
    """Make predictions on feature vectors without experimental pKa labels."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions = []

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)

            for i in range(len(out)):
                predictions.append({
                    "PDB_ID": batch.PDB_ID[i],
                    "Residue_Number": batch.Residue_Number[i],
                    "Residue": batch.Residue_Name[i],
                    "Predicted_pKa": out[i].item()
                })

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return predictions


def predict_with_all_models(model_dir, data_sets, base_dir):
    """Predict and save results for all models with optional metrics calculation."""
    device = torch.device("cpu")
    output_dir = os.path.join(base_dir, "Results/Predictions")
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

            # Calculate MAE and RMSE only if True_pKa exists in predictions
            if predictions and "True_pKa" in predictions[0]:
                true_labels = [pred["True_pKa"] for pred in predictions]
                predicted_labels = [pred["Predicted_pKa"] for pred in predictions]
                mae = mean_absolute_error(true_labels, predicted_labels)
                rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))

                metrics.append({"Dataset": dataset_idx + 1, "Model": model_name, "MAE": mae, "RMSE": rmse})
                print(f"Dataset {dataset_idx + 1}, Model: {model_name}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            else:
                print(f"Dataset {dataset_idx + 1}, Model: {model_name} - No true labels (unlabeled data)")

    # Save metrics to a CSV file if metrics exist
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_csv = os.path.join(output_dir, "model_metrics_all_datasets.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pKa dataset PKL files and make predictions")
    parser.add_argument("--base_dir", type=str, default="../Graph_pKa",
                        help="Base directory containing the data folders")
    parser.add_argument("--model_dir", type=str, 
                        default="Results/GAT_Grid_Search/loss_MSELoss_h48_b16_lr0.01_d0.3_hd6",
                        help="Directory containing trained model files")
    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent / args.base_dir
    
    # Generate datasets
    print("=" * 60)
    print("STEP 1: Generate PyTorch Geometric Datasets")
    print("=" * 60)
    main(base_dir)
    
    # Make predictions with trained models
    print("\n" + "=" * 60)
    print("STEP 2: Make Predictions with Trained Models")
    print("=" * 60)
    from Net.GNN_Grid_Search.GAT import load_training_data
    
    model_dir = os.path.join(base_dir, args.model_dir)
    dataset_dir = os.path.join(base_dir, "Features/Node_Feature_Vectors/Prediction_Datasets")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    data_sets, input_dim, Residue_Type_labels = load_training_data(dataset_dir)
    predict_with_all_models(model_dir, data_sets, base_dir)
