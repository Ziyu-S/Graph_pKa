import os
import re
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
  
def load_models_and_hparams(model_dir):
    """
    Automatically detect hyperparameters from the model directory name
    and list all .pth files inside that directory.

    Args:
        model_dir (str): Path to the model directory or a .pth file path

    Returns:
        dict: Dictionary containing hyperparameters
        list: Sorted list of .pth file paths
    """

    # If model_dir is a .pth file, use its parent folder
    if model_dir.endswith(".pth"):
        model_dir = os.path.dirname(model_dir)

    # Extract directory name that contains hyperparameters
    folder_name = os.path.basename(model_dir)

    # Regex pattern to match the parameters in your naming style
    pattern = r"loss_(\w+)_h(\d+)_b(\d+)_p(\d+)_f(\d+)_lr([0-9.]+)_d([0-9.]+)$"
    match = re.search(pattern, folder_name)
    if not match:
        raise ValueError(f"Could not parse hyperparameters from: {folder_name}")

    loss_name, hidden, batch, patience, folds, lr, dropout = match.groups()

    # Map strings to actual objects (expand as needed)
    loss_fn_map = {
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss":torch.nn.SmoothL1Loss,
    }

    hparams = {
        "loss_function": loss_fn_map.get(loss_name, None),
        "hidden_channels": int(hidden),
        "batch_size": int(batch),
        "patience": int(patience),
        "folds": int(folds),
        "learning_rate": float(lr),
        "dropout": float(dropout),
        "pooling_function": global_mean_pool,
    }

    # List all .pth files in sorted order
    pth_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found under: {model_dir}")
    else:
        print(f"Found {len(pth_files)} pth files:")
    
    return hparams, pth_files

class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout):
        super(GCNNet, self).__init__()
    

        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x.view(-1)

def process_fold(pth_file, hidden_channels, dropout, device, data_pkl_path):
    with open(data_pkl_path, "rb") as f:
        obj = pickle.load(f)

    # normalize to list[Data]
    if isinstance(obj, list):
        data_list = obj
    elif hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        data_list = [obj[i] for i in range(len(obj))]
    else:
        raise TypeError(f"Unsupported .pkl content: {type(obj)}")

    if not data_list:
        raise ValueError(f"Empty dataset at {data_pkl_path}")

    input_dim = int(data_list[0].x.size(1))
    fold_name = os.path.basename(pth_file)
    print(f"\n[INFO] Explaining {fold_name} (input_dim={input_dim})")

    model = GCNNet(input_dim, hidden_channels, dropout=dropout).to(device)
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw')
    )

    data_loader = DataLoader(data_list, batch_size=1, shuffle=False)
    all_importances = []

    for idx, data in enumerate(data_loader):
        data = data.to(device)
        print(f"[{fold_name}] Explaining graph {idx+1}/{len(data_list)}...")
        y_t = data.y.view(-1)
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch=data.batch, target=y_t)
        node_feat_mask = explanation.node_mask

        mean_feat_importance = node_feat_mask.detach().cpu().numpy().mean(axis=0)
        print(f"[{fold_name}] Graph {idx+1} mean feature importance: {np.round(mean_feat_importance, 4)}")

        all_importances.append(mean_feat_importance)

    fold_importance = np.mean(np.vstack(all_importances), axis=0)
    print(f"[INFO] Finished fold {fold_name}. Mean feature importance: {np.round(fold_importance, 4)}")
    return fold_importance

if __name__ == "__main__":

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model_dir = "../../Results/GCN_Grid_Search/saved_models/dataset_2/loss_MSELoss_h64_b16_p20_f10_lr0.006_d0.3"
    hparams, pth_files = load_models_and_hparams(model_dir)
    print(f"Found {len(pth_files)} pth files")
    print("Detected hyperparameters:", {k: v for k, v in hparams.items() if k.endswith("_name") or k in ("hidden_channels","batch_size","learning_rate","dropout")})

    data_pkl_path = "../../Data/4_Residues_W_Local_Frame/Subsets/data_list_2.pkl"
    
    procs = 1 if device.type == "cuda" else min(len(pth_files), mp.cpu_count())
    
    # Run in parallel using Pool
    worker = partial(
        process_fold,
        hidden_channels=hparams["hidden_channels"],
        dropout=hparams["dropout"],
        device=device,
        data_pkl_path=data_pkl_path,
    )

    with mp.get_context("spawn").Pool(processes=procs) as pool:
        feature_importances = list(tqdm(pool.imap(worker, pth_files), total=len(pth_files)))
    # Final average across folds
    global_feature_importance = np.mean(np.vstack(feature_importances), axis=0)

    # Save to CSV
    mean_importance_df = pd.DataFrame({
        'Feature_Index': np.arange(len(global_feature_importance)),
        'Importance': global_feature_importance
    })
    mean_importance_df.to_csv('../../Results/GCN_Grid_Search/Feature_Importance/GCN_Feature_Importance_1.csv', index=False)


