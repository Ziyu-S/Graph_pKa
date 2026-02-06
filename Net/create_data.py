import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data


def main(base_dir):
    # Directories for adjacency matrices and node feature sets
    adjacency_matrix_dir = os.path.join(base_dir, 'Adj_Matrix/With_Self_Loop')
    node_feature_dirs = [
        os.path.join(base_dir, '4_Residues_W_Local_Frame/7'),
        os.path.join(base_dir, '4_Residues_W_Local_Frame/8'),
        os.path.join(base_dir, '4_Residues_W_Local_Frame/9'),
        os.path.join(base_dir, '4_Residues_W_Local_Frame/10'),
        os.path.join(base_dir, '4_Residues_W_Local_Frame/11')
    ]

    # Output directory for saving datasets as PKL
    output_dir = os.path.join(base_dir, "4_Residues_W_Local_Frame/Subsets")
    os.makedirs(output_dir, exist_ok=True)

    for idx, node_features_dir in enumerate(node_feature_dirs):
        print(f"\nProcessing node features from: {node_features_dir}")
        data_list = []

        for filename in os.listdir(adjacency_matrix_dir):
            if filename.endswith('.csv'):
                adjacency_matrix = pd.read_csv(
                    os.path.join(adjacency_matrix_dir, filename), header=0, index_col=0).values
                adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.int)

                features_filename = filename.replace('_adjacency.csv', '.csv')
                node_features_path = os.path.join(node_features_dir, features_filename)

                if not os.path.exists(node_features_path):
                    print(f"Warning: {features_filename} not found in {node_features_dir}. Skipping...")
                    continue

                base_name = features_filename.replace('.csv', '')
                pdb_id, residue_info = base_name.split('_')
                residue_number, residue_name = residue_info.split('.')

                node_features = pd.read_csv(node_features_path, header=0)
                if 'Expt.pKa' not in node_features or 'atom_label' not in node_features:
                    continue
                if len(node_features['Expt.pKa']) == 0:
                    continue

                pKa_labels_tensor = torch.tensor([node_features['Expt.pKa'].values[0]], dtype=torch.float)

                atom_label_encoded = F.one_hot(torch.tensor(node_features['atom_label'].values, dtype=torch.long), num_classes=9).float()
                one_hot_cols = node_features.filter(like='Residue Name_').values
                residue_label = np.argmax(one_hot_cols, axis=1)[0]

                features_tensor = torch.tensor(
                    node_features.drop(columns=['Expt.pKa', 'atom_label']).values, dtype=torch.float
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

        pkl_path = os.path.join(output_dir, f'data_list_{idx}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved dataset {idx} with {len(data_list)} graphs to {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pKa dataset PKL files")
    parser.add_argument("--base_dir", type=str, default="../Graph_pKa/PKAD_Data",
                        help="Base directory containing the data folders")
    args = parser.parse_args()

    main(args.base_dir)
