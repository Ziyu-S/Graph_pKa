import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def merge_predictions_by_dataset(input_dir, output_dir):
    """Merge predictions for each dataset by averaging across 10 folds."""
    os.makedirs(output_dir, exist_ok=True)

    # Group files by dataset
    files = sorted(Path(input_dir).glob("predictions_dataset_*_best_model_fold_*.csv"))
    grouped_files = {}

    for file in files:
        parts = file.stem.split("_")
        dataset = parts[2]  # Adjusted index to correctly extract dataset number
        grouped_files.setdefault(dataset, []).append(file)

    metrics = []  # To store MAE and RMSE for each dataset

    # Process each dataset
    for dataset, file_group in grouped_files.items():
        all_predictions = []

        for file in file_group:
            df = pd.read_csv(file)
            all_predictions.append(df.set_index(["PDB_ID", "Residue_Number", "Residue", "True_pKa"]))

        # Average predictions
        merged_df = pd.concat(all_predictions, axis=1).mean(axis=1).reset_index()
        merged_df.columns = ["PDB_ID", "Residue_Number", "Residue", "True_pKa", "Average_Predicted_pKa"]

        # Calculate MAE and RMSE
        mae = mean_absolute_error(merged_df["True_pKa"], merged_df["Average_Predicted_pKa"])
        rmse = np.sqrt(mean_squared_error(merged_df["True_pKa"], merged_df["Average_Predicted_pKa"]))
        metrics.append({"Dataset": dataset, "MAE": mae, "RMSE": rmse})

        # Save merged predictions
        output_file = Path(output_dir) / f"merged_predictions_dataset_{dataset}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Merged predictions saved to {output_file}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_file = Path(output_dir) / "merged_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    input_dir = "/home/ziyu-song/Graph_pKa/Mutant_Data/All_Predictions"
    output_dir = "/home/ziyu-song/Graph_pKa/Mutant_Data/All_Predictions/Merged"

    merge_predictions_by_dataset(input_dir, output_dir)