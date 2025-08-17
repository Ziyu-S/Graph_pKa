#!/usr/bin/env python3
import os
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import torch.nn.functional as F
from pathlib import Path
from itertools import product
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader

# Edit here to use your dir
base_dir = "/Graph_pKa/Results/GAT_Grid_Search"
DATASET_DIR = Path("/Graph_pKa/Data/4_Residues_W_Local_Frame/Subsets")

def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Control the number of threads to ensure consistency
    torch.set_num_threads(1)  

set_seed(42)

# Data loading
def load_training_data(dataset_dir, max_index=5):
    """
    Load graph datasets from pickled files in the given directory.
    
    """
    dataset_dir = Path(dataset_dir)
    data_sets = []

    for i in range(max_index):
        pkl_path = dataset_dir / f"data_list_{i}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                dl = pickle.load(f)
                if len(dl) > 0:
                    data_sets.append(dl)
                    print(f"Loaded dataset {i}: {pkl_path} ({len(dl)} graphs)")
        else:
            if i > 0:
                break

    assert len(data_sets) > 0, f"No datasets found under {dataset_dir}"

    # Use the first graph in the first dataset to infer dimensions
    data = data_sets[0][0]
    input_dim = data.x.shape[1]  # Node feature dimension
    Residue_Type_labels = defaultdict(lambda: None)

    return data_sets, input_dim, Residue_Type_labels


class GATConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout, heads):
        super(GATConv, self).__init__()
        # input_dim = data.x.shape[1]  # Adjust input dimension for one-hot encoding
        print(f"Initializing GATLayer: input_dim={input_dim}, hidden={hidden_channels}, heads={heads}")
        
        self.conv1 = GATv2Conv(input_dim, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.dropout = torch.nn.Dropout(dropout) 
        self.pooling_function = global_mean_pool
        self.out_layer = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch
        x = data.x
        
        # Apply GAT Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pooling_function(x, batch)
        x = self.out_layer(x)
        return x



def save_predictions_to_csv(all_best_predictions, dataset_idx, loss_function, hidden_channels, batch_size, patience, k_folds, lr, dropout, heads):
    # Define output directory
    output_dir = f"{base_dir}/all_best_predictions/dataset_4_{dataset_idx}/{loss_function.__class__.__name__}_hidden_{hidden_channels}_bs_{batch_size}_lr_{lr}_dropout_{dropout}_head_{heads}"
    os.makedirs(output_dir, exist_ok=True)

    # Construct filename with hyperparameters
    filename = f"predictions_dataset_4_{dataset_idx}_loss_{loss_function.__class__.__name__}_hidden_{hidden_channels}_bs_{batch_size}_lr_{lr}_dropout_{dropout}_head_{heads}.csv"
    output_csv_path = os.path.join(output_dir, filename)

    # Convert to Pandas DataFrame and save
    df_predictions = pd.DataFrame(all_best_predictions)
    df_predictions.to_csv(output_csv_path, index=False)

    # print(f"Predictions saved: {output_csv_path}")

# Function to train and evaluate a model on a given set of hyperparameters
def train_and_evaluate(loss_function, hidden_channels, batch_size, patience, k_folds, lr, dropout, heads, dataset_idx, data_list, input_dim, Residue_Type_labels): 
    """
    This function trains and evaluates a GAT model using separate CPU cores.
    """
    device = torch.device("cpu")  # Force CPU usage for parallel training

    save_dir = f"{base_dir}/saved_models/dataset_4_{dataset_idx}/loss_{loss_function.__class__.__name__}_h{hidden_channels}_b{batch_size}_lr{lr}_d{dropout}_hd{heads}"
    os.makedirs(save_dir, exist_ok=True)

    # Cross-validation setup
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    mae_scores, mse_scores = 0.0, 0.0
    total_samples_fold = 0
    all_best_predictions = []
    # mae_by_residue_type = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):

        # Initialize model on CPU
        model = GATConv(input_dim, hidden_channels, dropout, heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_path = os.path.join(save_dir, f"best_model_fold_{fold+1}.pth")

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        # Training loop with early stopping
        best_MAE = float("inf")
        best_total_asbe = 0.0
        best_total_sque = 0.0
        best_predictions = []
        counter = 0

        # Training loop with early stopping
        for epoch in range(500):  
            model.train()
            total_loss = 0
            total_samples_train = 0


            for batch in train_loader:
                batch = batch.to(device)
                batch_size_actual_x = batch.y.size(0)
                optimizer.zero_grad()
                out = model(batch).squeeze()
                loss = loss_function(out, batch.y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size_actual_x
                total_samples_train += batch_size_actual_x

            avg_loss = total_loss / total_samples_train

            # Validation loop
            model.eval()
            total_abse = 0.0
            total_sque = 0.0
            total_samples_val = 0
            epoch_predictions = []

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(device)

                    out = model(batch).view(-1)
                    loss = loss_function(out.view(-1), batch.y.to(device).view(-1))
                    total_samples_val += batch.y.size(0)
                    # abs_error = torch.abs(out - batch.y)
                    
                    # Compute MAE and RMSE
                    batch_abse = F.l1_loss(out, batch.y, reduction='sum')
                    batch_sque = F.mse_loss(out, batch.y, reduction='sum')
                    total_abse += batch_abse.item()
                    total_sque += batch_sque.item()

                    # print(i, 'batch_actual', batch_size_actual_y, total_samples_val)
            
                    # Collect metrices by residue type and save predictions
                    for j in range(len(batch.y)):
                        if (i * batch_size + j) < len(val_idx):
                            graph_id = val_idx[i * batch_size + j]
                            residue_type_value = Residue_Type_labels[graph_id]
                            # mae_by_residue_type[residue_type_value].append(abs_error[j].item())

                            # Collect predictions
                            pdb_id = data_list[graph_id].PDB_ID
                            residue_number = data_list[graph_id].Residue_Number
                            residue_name = data_list[graph_id].Residue_Name
                            true_pKa_label = batch.y[j].item()
                            prediction = out[j].item()

                            epoch_predictions.append({
                                'graph_id': graph_id,
                                'PDB_ID': pdb_id,
                                'Residue_Number': residue_number,
                                'Residue': residue_name,
                                'residue_type': residue_type_value,
                                'true_pKa_label': true_pKa_label,
                                'prediction': prediction
                        })

                avg_val_mae = total_abse / total_samples_val

                # Early Stopping
                if avg_val_mae < best_MAE:
                    best_MAE = avg_val_mae
                    best_total_asbe = total_abse
                    best_total_sque = total_sque
                    best_predictions = epoch_predictions.copy()
                    torch.save(model.state_dict(), model_path)
                    # print(f'model saved as {model_path}')
                    counter = 0  # Reset patience counter
                else:
                    counter += 1
                    if counter >= patience and epoch > 60:  
                        print(f"Early stopping at epoch {epoch} (No improvement)")
                        break

                if epoch % 10 == 0:
                    print(f" Epoch {epoch}: Train Loss={avg_loss:.4f}, Val MAE={avg_val_mae:.4f}")

        # Accumulate from last epoch
        mae_scores += best_total_asbe
        mse_scores += best_total_sque
        total_samples_fold += total_samples_val
        all_best_predictions.extend(best_predictions)              

    # Compute average metrics
    avg_mae = mae_scores/ total_samples_fold
    avg_rmse = torch.sqrt(torch.tensor(mse_scores / total_samples_fold)).item()
    # Save predictions to CSV
    save_predictions_to_csv(all_best_predictions, dataset_idx, loss_function, hidden_channels, batch_size, patience, k_folds, lr, dropout, heads)
    return (dataset_idx, loss_function.__class__.__name__, hidden_channels, batch_size, patience, k_folds, lr, dropout, heads, avg_mae, avg_rmse)

if __name__ == "__main__":

    # Get the number of available CPU cores
    available_cores = multiprocessing.cpu_count()
    num_cores = min(60, available_cores - 3)  # Use up to 60 cores
    print(f"Using {num_cores} CPU cores for parallel training.")

    # Load data (makes input_dim & Residue_Type_labels available for train_and_evaluate)
    data_sets, input_dim, Residue_Type_labels = load_training_data(DATASET_DIR)

    # Hyper Parameters ranges:
    heads = [4, 6, 8]
    hidden_channels_range = list(range(16, 65, 16))
    batch_size_range = list(range(16, 41, 8))
    k_folds_range = [10]
    patience_range = [20]
    learning_rate_range = [0.001, 0.006, 0.01, 0.06, 0.1]
    dropout_range = np.round(np.arange(0.2, 0.51, 0.1), 2).tolist()
    loss_functions = [torch.nn.SmoothL1Loss(beta = 0.5), torch.nn.L1Loss(), torch.nn.MSELoss()]
    
    # Store the results
    results = []
    for dataset_idx, data_list in enumerate(data_sets):
        print(f'\nPerforming Grid Search on Dataset {dataset_idx + 1}')

        # Create all hyperparameter combinations
        param_combinations = list(product(
            loss_functions,
            hidden_channels_range,
            batch_size_range,
            patience_range,
            k_folds_range,
            learning_rate_range,
            dropout_range, 
            heads
        ))

        # Run training in parallel using multiple CPU cores
        dataset_results = Parallel(n_jobs=num_cores)(
            delayed(train_and_evaluate)(*params, dataset_idx, data_list, input_dim, Residue_Type_labels)
            for params in param_combinations
        )

        # Append results for this dataset
        results.extend(dataset_results)

        # #Perform grid search across all datasets using single CPU training
        # for params in param_combinations:
        #     dataset_results = train_and_evaluate(*params, dataset_idx, data_list, input_dim, Residue_Type_labels)
        
        # # Append results for this dataset
        # results.extend(dataset_results)    

    # Display the best combination based on MAE
    best_result = min(results, key=lambda x: (x[9], x[10]))  # Sort by MAE
    print(f"\nBest Combination: Dataset={best_result[0]}, Loss_fn = {best_result[1]}, Hidden={best_result[2]}, Batch={best_result[3]}, "
        f"Patience={best_result[4]}, K-Folds={best_result[5]}, LR={best_result[6]}, Dropout={best_result[7]}, heads = {best_result[8]}")
    print(f"Best MAE: {best_result[9]:.4f}, Best RMSE: {best_result[10]:.4f}")

    # Save best result to a CSV
    best_result_df = pd.DataFrame([best_result], 
                                columns=['Dataset', 'Loss_fn', 'Hidden', 'Batch', 'Patience', 
                                        'K-Folds', 'LR', 'Dropout', 'heads', 'MAE', 'RMSE'])
    best_result_df.to_csv(f'{base_dir}/best_result_4_REsidues.csv', index=False)

    # Save all results to a CSV
    results_df = pd.DataFrame(results, 
                            columns=['Dataset', 'Loss_fn', 'Hidden', 'Batch', 'Patience', 
                                        'K-Folds', 'LR', 'Dropout', 'heads', 'MAE', 'RMSE'])
    results_df.to_csv(f'{base_dir}/grid_search_results_4_REsidues.csv', index=False)

    print("Results saved to 'grid_search_results.csv' and 'best_result.csv'")
