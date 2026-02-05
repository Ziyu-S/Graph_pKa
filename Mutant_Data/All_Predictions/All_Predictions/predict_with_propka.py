import os
import subprocess
import pandas as pd
from pathlib import Path

def run_propka(pdb_file, output_dir):
    """Run PROPKA on a given PDB file and save the output."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    command = ["propka3", str(pdb_file)]  # Remove the incorrect -o flag
    subprocess.run(command, check=True, cwd=output_dir)  # Run in the output directory
    print(f"PROPKA results saved in {output_dir}")

def predict_pdbs_with_propka(pdb_list_file, pdb_dir, output_dir):
    """Predict pKa values for all PDBs using PROPKA."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the list of PDB IDs
    pdb_ids = pd.read_csv(pdb_list_file, header=None)[0].tolist()

    for pdb_id in pdb_ids:
        pdb_file = Path(pdb_dir) / f"{pdb_id}.pdb"
        if pdb_file.exists():
            print(f"Processing {pdb_file}...")
            run_propka(pdb_file, output_dir)
        else:
            print(f"PDB file not found: {pdb_file}")

if __name__ == "__main__":
    # Paths
    pdb_list_file = "/home/ziyu-song/Graph_pKa/Results/NewData/All_Predictions/unique_pdb_ids.csv"
    pdb_dir = "/home/ziyu-song/Graph_pKa/Results/NewData/All_Predictions/PROPKA/"
    output_dir = "/home/ziyu-song/Graph_pKa/Results/NewData/All_Predictions/PROPKA_Results"

    # Predict pKa values for all PDBs
    predict_pdbs_with_propka(pdb_list_file, pdb_dir, output_dir)
