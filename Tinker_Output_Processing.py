#!/usr/bin/env python3
"""Master script containing all procedures for processing Tinker output.

This pipeline processes Tinker molecular dynamics output files through multiple stages:
1. Convert .xyz files to CSV format
2. Assign atom types from AMOEBA forcefield
3. Extract dipole moments from .uind files
4. Transform coordinates to local reference frames
5. Map atoms to amino acid residues
6. Generate final PDB files with refined coordinates
"""

import re
import os
import ast
import csv
import glob
import shutil
import logging
import numpy as np
import pandas as pd
import mdtraj as md
import MDAnalysis as mda
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Centralized Amino Acid Mappings ===
# Standard 3-letter code to full name mapping
AMINO_ACID_3_TO_FULL = {
    "ALA": "Alanine",
    "ARG": "Arginine",
    "ASN": "Asparagine",
    "ASP": "Aspartate",
    "CYS": "Cysteine",
    "GLU": "Glutamate",
    "GLN": "Glutamine",
    "GLY": "Glycine",
    "HIS": "Histidine",
    "ILE": "Isoleucine",
    "LEU": "Leucine",
    "LYS": "Lysine",
    "MET": "Methionine",
    "PHE": "Phenylalanine",
    "PRO": "Proline",
    "SER": "Serine",
    "THR": "Threonine",
    "TRP": "Tryptophan",
    "TYR": "Tyrosine",
    "VAL": "Valine",
}

# Derived mappings
AMINO_ACID_FULL_TO_3 = {v: k for k, v in AMINO_ACID_3_TO_FULL.items()}
AMINO_ACID_LOWERCASE_TO_3 = {k.lower(): v for k, v in AMINO_ACID_3_TO_FULL.items()}

def extract_pdb_id_from_pdb_file(pdb_file):
    """Extract PDB ID from refined PDB filename.
    
    Converts: /path/to/1A6K_refined_coordinates.pdb -> 1A6K
    """
    return os.path.basename(pdb_file).replace("_refined_coordinates.pdb", "").split("_")[0]


def normalize_atom_name_for_merge(atom_name):
    """Normalize atom names for merge matching: treat HN as 'H'."""
    if pd.isna(atom_name):
        return None
    atom_str = str(atom_name).strip()
    return "H" if atom_str == "HN" else atom_str


def find_pdb_files(output_dir):
    """Find all refined PDB files in PDB-ID named subfolders.
    
    Returns:
        list: Full paths to all *_refined_coordinates.pdb files found
    """
    pdb_id_pattern = re.compile(r"^[A-Za-z0-9]{4}$")
    pdb_files = []
    
    for entry in os.scandir(output_dir):
        if entry.is_dir():
            folder_name = os.path.basename(entry.path)
            if pdb_id_pattern.match(folder_name):
                found = glob.glob(
                    os.path.join(entry.path, "**", "*_refined_coordinates.pdb"),
                    recursive=True,
                )
                if found:
                    pdb_files.extend(found)
    
    return pdb_files



def normalize_residue_name(name):
    """Convert residue names in any format to standard 3-letter code."""
    if pd.isna(name):
        return None

    name_str = str(name).strip()
    name_lower = name_str.lower()

    # Try lowercase 3-letter code first
    if name_lower in AMINO_ACID_LOWERCASE_TO_3:
        return AMINO_ACID_LOWERCASE_TO_3[name_lower]
    
    # Try full name
    if name_str in AMINO_ACID_FULL_TO_3:
        return AMINO_ACID_FULL_TO_3[name_str]
    
    # If it's already 3 letters, uppercase it
    if len(name_str) == 3:
        return name_str.upper()
    
    return name_str

# === Configuration ===
CONFIG = {
    "input_dir": "../Graph_pKa/Data/7_Energy_Minimization_Systems",
    "output_dir": "../Graph_pKa/Features",
    "raw_pdb_dir": "../Graph_pKa/Data/0_Raw_PDB",
    "atom_types_file": "../Graph_pKa/Tinker_params/ff_atoms.csv",
}


def process_all_xyz_files(Dir):
    """Convert Tinker .xyz_2 files to CSV and track solvent stop atoms.
    
    Creates a subfolder within Features for each .xyz_2 file.
    """
    stop_info = []
    output_csv = os.path.join(CONFIG["output_dir"], "stop_info.csv")

    # Recursively walk through the directory
    for root, dirs, files in os.walk(Dir):
        for filename in files:
            if not filename.endswith(".xyz_2"):
                continue

            file_path = os.path.join(root, filename)
            
            # Extract base name without extension to create subfolder
            base_name = filename.replace(".xyz_2", "")
            subfolder_path = os.path.join(CONFIG["output_dir"], base_name)
            os.makedirs(subfolder_path, exist_ok=True)
            
            output_csv_file = os.path.join(subfolder_path, filename.replace(".xyz_2", "_XYZ.csv"))

            # Parse tinker XYZ and detect stop atom
            tinkerXYZ = []
            stop_atom_number = None
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.split()
                    
                    # Skip empty lines and header lines
                    if not parts or not parts[0][0].isdigit():
                        continue

                    try:
                        # Process current line
                        if len(parts) > 6 and parts[0].isdigit() and parts[1].isalpha():
                            if stop_atom_number is not None:
                                break

                            atom_number = int(parts[0])
                            atom_name = parts[1]
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            atom_type = parts[5]
                            bonds = parts[6:]

                            # Check if the current atom follows O-H-H-O-H-H pattern
                            if atom_name == "O" and i + 5 < len(lines):
                                next_parts_1 = lines[i + 1].split()
                                next_parts_2 = lines[i + 2].split()
                                next_parts_3 = lines[i + 3].split()
                                next_parts_4 = lines[i + 4].split()
                                next_parts_5 = lines[i + 5].split()

                                if (
                                    len(next_parts_1) > 1 and next_parts_1[1] == "H"
                                    and len(next_parts_2) > 1 and next_parts_2[1] == "H"
                                    and len(next_parts_3) > 1 and next_parts_3[1] == "O"
                                    and len(next_parts_4) > 1 and next_parts_4[1] == "H"
                                    and len(next_parts_5) > 1 and next_parts_5[1] == "H"
                                ):
                                    stop_atom_number = atom_number
                                    continue  # Stop adding more records to tinkerXYZ

                            tinkerXYZ.append(
                                {
                                    "atom_number": atom_number,
                                    "atom_name": atom_name,
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "atom_type": atom_type,
                                    "bonds": bonds,
                                }
                            )
                    except (ValueError, IndexError) as e:
                        logger.error(f"Skipping line due to error: {e} in file {file_path}")
                        continue
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

            if stop_atom_number is not None:
                stop_info.append(
                    {
                        "filename": os.path.splitext(filename)[0],
                        "stop_atom_number": stop_atom_number,
                    }
                )

            # Write the Output CSV file
            fieldnames = [
                "atom_number",
                "atom_name",
                "x",
                "y",
                "z",
                "atom_type",
                "bonds",
            ]
            try:
                with open(output_csv_file, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for residue in tinkerXYZ:
                        writer.writerow(residue)
                logger.info(f"Data successfully written to {output_csv_file}")
            except Exception as e:
                logger.error(f"Error writing CSV {output_csv_file}: {e}")

    # Save the stop information to a CSV file
    try:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["filename", "stop_atom_number"])
            writer.writeheader()
            writer.writerows(stop_info)
        logger.info(f"Stop info saved to {output_csv}")
    except Exception as e:
        logger.error(f"Error writing stop_info CSV: {e}")



# === Assign AMOEBAbio18 Atom Types to Converted CSV ===


def map_bonds_to_atom_types(row, atom_types_df):
    """Map bond numbers to atom types from the Atom_types dataframe."""
    try:
        # Convert the string representation of a list into an actual list
        bond_numbers = ast.literal_eval(row["bonds"])
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse bonds for row: {row}")
        return ""

    bond_atom_types = []
    for bond in bond_numbers:
        try:
            # Convert bond from string to integer, and map it to the corresponding atom type
            bond_value = int(bond)
            matching_rows = atom_types_df[atom_types_df["atom_number"] == bond_value]
            
            if not matching_rows.empty:
                bond_atom_type = str(matching_rows["atom_type"].values[0])
                bond_atom_types.append(bond_atom_type)
            else:
                logger.debug(f"No atom type found for bond number: {bond}")
        except (ValueError, IndexError) as e:
            logger.debug(f"Error processing bond {bond}: {e}")

    return ", ".join(bond_atom_types)


def process_all_csv_files_to_add_Atom_Types(Dir, Atom_types):
    for root, dirs, files in os.walk(Dir):  # Recursively walk through the directory
        for filename in files:
            if not filename.endswith("_XYZ.csv"):
                continue

            file_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(file_path)

                # Apply the function to each row in the DataFrame
                df["Multipole_Atoms"] = df.apply(
                    lambda row: map_bonds_to_atom_types(row, Atom_types), axis=1
                )

                # Save the updated DataFrame to a new CSV file
                output_file_path = file_path.replace(".csv", "_Atom_Types_updated.csv")
                df.to_csv(output_file_path, index=False)

                logger.info(f"Data successfully written to {output_file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")


Atom_types = None
try:
    Atom_types = pd.read_csv(CONFIG["atom_types_file"])
    logger.info(f"Loaded atom types from {CONFIG['atom_types_file']}")
except FileNotFoundError:
    logger.error(f"Atom types file not found: {CONFIG['atom_types_file']}")
except Exception as e:
    logger.error(f"Error loading atom types: {e}")


# === Extract Dipole Moments and Combine with CSV ===


def process_all_csv_files_add_dipoles(Dir):
    """Add dipole moment vectors from .uind files to CSV data."""
    for root, dirs, files in os.walk(Dir):
        for filename in files:
            if filename.endswith("_Atom_Types_updated.csv"):
                tinkerXYZ = os.path.join(root, filename)

                # Correctly identify the base name to find the corresponding .uind file
                # Extract just the protein ID (e.g., 1QH7 from 1QH7_XYZ_Atom_Types_updated.csv)
                base_name = filename.replace("_XYZ_Atom_Types_updated.csv", "")
                # Look for .uind file in a subfolder with the same name as the protein ID
                uind_file = os.path.join(CONFIG["input_dir"], base_name, base_name + ".uind")

                if not os.path.exists(uind_file):
                    logger.warning(f"Corresponding .uind file not found at {uind_file}")
                    continue

                # Extract dipole moments from the .uind file
                dipole_moments = []
                try:
                    with open(uind_file, "r") as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            # Skip empty lines and header lines (FRAME, etc.)
                            if not parts or not parts[0][0].isdigit():
                                continue
                            # Must have at least 7 parts and parts[2], [3], [4] must be numeric (dipole coordinates)
                            if len(parts) > 6:
                                try:
                                    # Verify parts[2], [3], [4] are valid floats (dipole coordinates)
                                    float(parts[2])
                                    float(parts[3])
                                    float(parts[4])
                                    dipole_vector = f"{parts[2]}, {parts[3]}, {parts[4]}"
                                    dipole_moments.append(dipole_vector)
                                except ValueError:
                                    # Skip lines where coordinates are not numeric
                                    continue
                except Exception as e:
                    logger.error(f"Error reading .uind file {uind_file}: {e}")
                    continue

                # Append dipole moments to the CSV
                try:
                    df = pd.read_csv(tinkerXYZ)
                    fieldnames = list(df.columns) + ["Dipole_Vector"]
                    
                    # Add dipole vectors
                    dipole_col = []
                    for i in range(len(df)):
                        if i < len(dipole_moments):
                            dipole_col.append(dipole_moments[i])
                        else:
                            dipole_col.append("")
                    df["Dipole_Vector"] = dipole_col

                    output_csv = tinkerXYZ.replace(".csv", "_XYZ_with_dipoles.csv")
                    df.to_csv(output_csv, index=False)
                    logger.info(f"Updated CSV with Dipole_Vector: {output_csv}")
                    
                except Exception as e:
                    logger.error(f"Error processing dipoles for {tinkerXYZ}: {e}")
                    continue




# === Normalize Atoms' Coordinates within a Universal Local Frame ===


def process_all_csv_files_local_frame(Dir):
    failed_files = []
    for root, dirs, files in os.walk(Dir):
        for filename in files:
            if filename.endswith("_XYZ_with_dipoles.csv"):
                file_path = os.path.join(root, filename)

                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Recalculate the coordinates based on CA atoms
                    def normalize(v):
                        """Normalize a vector."""
                        norm = np.linalg.norm(v)
                        return v / norm if norm != 0 else v

                    def define_local_frame(ca_coords, c_coords, o_coords):
                        """
                        Define a local frame:
                        - x-axis along the vector from CA to its immediate C.
                        - z-axis perpendicular to the CA-C-O plane.
                        - y-axis orthogonal to both x and z axes.
                        Returns the rotation matrix and the translated CA coordinates.
                        """
                        vec_ca_c = np.array(c_coords) - np.array(ca_coords)
                        vec_c_o = np.array(o_coords) - np.array(c_coords)

                        x_axis = normalize(vec_ca_c)
                        z_axis = normalize(np.cross(vec_ca_c, vec_c_o))
                        y_axis = np.cross(z_axis, x_axis)

                        R = np.array([x_axis, y_axis, z_axis]).T
                        return R, np.array(ca_coords)

                    recalculated_x, recalculated_y, recalculated_z = [], [], []
                    current_ca_coords, current_c_coords, current_o_coords = None, None, None
                    R = None

                    for index, row in df.iterrows():
                        if row["atom_name"] == "N":
                            if (
                                index + 1 < len(df)
                                and df.iloc[index + 1]["atom_name"] == "CA"
                                and index + 2 < len(df)
                                and df.iloc[index + 2]["atom_name"] == "C"
                                and index + 3 < len(df)
                                and df.iloc[index + 3]["atom_name"] == "O"
                            ):
                                current_ca_coords = [
                                    df.iloc[index + 1]["x"],
                                    df.iloc[index + 1]["y"],
                                    df.iloc[index + 1]["z"],
                                ]
                                current_c_coords = [
                                    df.iloc[index + 2]["x"],
                                    df.iloc[index + 2]["y"],
                                    df.iloc[index + 2]["z"],
                                ]
                                current_o_coords = [
                                    df.iloc[index + 3]["x"],
                                    df.iloc[index + 3]["y"],
                                    df.iloc[index + 3]["z"],
                                ]
                                R, current_ca_coords = define_local_frame(
                                    current_ca_coords, current_c_coords, current_o_coords
                                )

                        if R is None:
                            recalculated_x.append(None)
                            recalculated_y.append(None)
                            recalculated_z.append(None)
                            continue

                        translated_coords = np.array(
                            [row["x"], row["y"], row["z"]]
                        ) - np.array(current_ca_coords)
                        local_coords = np.dot(R.T, translated_coords)

                        recalculated_x.append(local_coords[0])
                        recalculated_y.append(local_coords[1])
                        recalculated_z.append(local_coords[2])

                    df["recalculated_x"] = recalculated_x
                    df["recalculated_y"] = recalculated_y
                    df["recalculated_z"] = recalculated_z

                    updated_df = df

                    # Save the updated DataFrame to a new CSV file
                    output_csv_file_path = file_path.replace(
                        ".csv", "_within_local_frame.csv"
                    )
                    updated_df.to_csv(output_csv_file_path, index=False)

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
                    failed_files.append(
                        {"file_name": os.path.basename(file_path), "error": str(e)}
                    )

    # Convert failed files list to a DataFrame
    if failed_files:
        failed_df = pd.DataFrame(failed_files)
        logging.warning("The following files failed to process:")
        for _, row in failed_df.iterrows():
            logging.warning(
                f"Failed file: {row['file_name']} with error: {row['error']}"
            )
    else:
        failed_df = pd.DataFrame(columns=["file_name", "error"])

    # Return the DataFrame of failed files
    return failed_df




# === Align the Atoms with Corresponding Amino Acids ===


def process_all_csv_files_add_description(Dir):
    df2 = pd.read_csv(
        "../Graph_pKa/Tinker_params/ff_atoms.csv"
    )
    description_dict = df2.set_index("atom_number")["Description"].to_dict()

    for root, dirs, files in os.walk(Dir):
        # Get base folder name (protein ID like 1A6K, 3ft7, etc.)
        base_folder_name = os.path.basename(root)
        
        for filename in files:
            if not filename.endswith("_within_local_frame.csv"):
                continue

            file_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(file_path)
                df["Description"] = df["atom_type"].map(description_dict)

                # Use clean naming: {base_folder}_XYZ_with_dipoles_within_local_frame_described.csv
                output_file_path = os.path.join(root, f"{base_folder_name}_XYZ_with_dipoles_within_local_frame_described.csv")
                df.to_csv(output_file_path, index=False)

                print(f"Processed: {file_path}")
                logger.info(f"Created description file: {output_file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                logger.error(f"Failed to process {file_path}: {e}")


DESCRIPTION_DIR = CONFIG["output_dir"]


# === Copy Raw PDB Files, Extract Residue Information, and Add Residue IDs ===

def copy_pdbs_and_align_descriptions(root_dir, raw_pdb_dir):
    """
    Combined Step 6 & 7: Copy PDB files, extract residue info to CSV, then align with
    atomic descriptions from the molecular dynamics data.
    
    Parameters:
        root_dir (str): The root directory containing the subfolders.
        raw_pdb_dir (str): Directory containing existing PDB files (e.g., 0_Raw_PDB).
    """
    valid_amino_acids = set(AMINO_ACID_3_TO_FULL.values())
    processed_count = 0
    summary_records = []

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        base_name = os.path.basename(subdir_path)
        pdb_id = base_name.upper()
        
        # ===== STEP 6: COPY PDB AND EXTRACT RESIDUE INFO =====
        # Check for both uppercase and lowercase PDB files
        source_path = os.path.join(raw_pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(source_path):
            source_path = os.path.join(raw_pdb_dir, f"{base_name}.pdb")
        
        output_path = os.path.join(subdir_path, f"{pdb_id}.pdb")

        if not os.path.exists(source_path):
            logger.warning(f"Source PDB not found for {pdb_id}: {source_path}")
            continue

        try:
            shutil.copy2(source_path, output_path)
            logger.info(f"Copied {pdb_id}.pdb to {output_path}")
        except Exception as e:
            logger.error(f"Failed to copy {pdb_id}.pdb: {e}")
            continue

        # Extract residue info from the copied PDB and write CSV
        data = []
        with open(output_path, "r") as file:
            for line in file:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    residue_seq = line[22:26].strip()

                    full_name = AMINO_ACID_3_TO_FULL.get(residue_name, residue_name)
                    description = full_name + " " + atom_name
                    data.append(
                        [atom_name, full_name, chain_id, residue_seq, description]
                    )

        df1_path = os.path.join(subdir_path, f"{base_name}.csv")
        with open(df1_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Atom Name",
                    "Residue Name",
                    "Chain ID",
                    "Residue Sequence Number",
                    "Description",
                ]
            )
            writer.writerows(data)

        logger.info(f"Processed: {output_path} -> {df1_path}")

        # ===== STEP 7: ALIGN ATOMIC DESCRIPTIONS =====
        # Now look for the alignment file (df1 = XYZ with dipoles and descriptions)
        df1_alignment_filename = f"{base_name}_XYZ_with_dipoles_within_local_frame_described.csv"
        df1_alignment_path = os.path.join(subdir_path, df1_alignment_filename)

        if not os.path.exists(df1_alignment_path):
            logger.debug(f"Alignment file not yet ready: {df1_alignment_filename}")
            continue

        try:
            df1_alignment = pd.read_csv(df1_alignment_path)
            df2 = pd.read_csv(df1_path)

            df1_alignment["Description"] = df1_alignment["Description"].str.strip('"')

            prefix = None
            block_start = None
            first_block_processed = False

            residue_sequence_numbers = df2["Residue Sequence Number"].unique()
            residue_sequence_numbers.sort()

            residue_number_iterator = iter(residue_sequence_numbers)
            residue_number = next(residue_number_iterator, None)
            residue_numbers = []

            def is_first_block(index):
                if index + 2 < len(df1_alignment):
                    return (
                        df1_alignment.iloc[index]["Description"].endswith("NH2+")
                        or df1_alignment.iloc[index]["Description"].endswith("NH3+")
                        and df1_alignment.iloc[index + 1]["Description"].endswith("CA")
                        and df1_alignment.iloc[index + 2]["Description"].endswith("C")
                    )
                return False

            def is_subsequent_block(index):
                if index + 1 < len(df1_alignment):
                    return (
                        (
                            df1_alignment.iloc[index]["Description"].endswith("N")
                            or df1_alignment.iloc[index]["Description"].endswith("NH3+")
                        )
                        and df1_alignment.iloc[index + 1]["Description"].endswith("CA")
                    )
                return False

            for i in range(len(df1_alignment)):
                if residue_number is None:
                    residue_number = residue_sequence_numbers[-1]

                if not first_block_processed and is_first_block(i):
                    block_start = i
                    prefix = None
                    first_block_processed = True

                elif first_block_processed and is_subsequent_block(i):
                    block_start = i
                    try:
                        residue_number = next(residue_number_iterator)
                    except StopIteration:
                        residue_number += 1
                        logger.warning(
                            f"No more residue numbers available. "
                            f"Incrementing to the next number: {residue_number}"
                        )

                if df1_alignment.iloc[i]["Description"].endswith("CB"):
                    prefix = df1_alignment.iloc[i]["Description"].split()[0]

                if df1_alignment.iloc[i]["Description"].startswith("Glycine"):
                    prefix = df1_alignment.iloc[i]["Description"].split()[0]

                if prefix and block_start is not None:
                    for j in range(block_start, i + 1):
                        current_description = df1_alignment.at[j, "Description"]
                        if " " in current_description:
                            df1_alignment.at[j, "Updated_Description"] = prefix + " " + " ".join(
                                current_description.split()[1:]
                            )
                        else:
                            df1_alignment.at[j, "Updated_Description"] = prefix + " " + current_description

                residue_numbers.append(residue_number)

            while len(residue_numbers) < len(df1_alignment):
                residue_numbers.append(residue_number)

            df1_alignment["Residue_Number_PDB"] = residue_numbers

            output_file_path = df1_alignment_path.replace(
                "_XYZ_with_dipoles_within_local_frame_described.csv", "_NEW.csv"
            )
            df1_alignment.to_csv(output_file_path, index=False)
            logger.info(f"Created _NEW.csv: {output_file_path}")
            print(f"Aligned {base_name}: PDB->CSV and atomic descriptions->_NEW.csv")

            # ===== ADD RESIDUE IDS =====
            # Now read the _NEW.csv file and the PDB CSV to add Residue IDs
            try:
                df1_for_ids = pd.read_csv(df1_path)
                df2_for_ids = pd.read_csv(output_file_path)

                # Filter df1 for amino acids (Residue Name length >= 6 indicates full name like "Alanine")
                df1_for_ids = df1_for_ids[df1_for_ids["Residue Name"].str.len() >= 6]
                df1_for_ids["Residue ID"] = df1_for_ids["Residue Sequence Number"].ne(
                    df1_for_ids["Residue Sequence Number"].shift()
                ).cumsum()

                # Extract residue name from Updated_Description in df2
                df2_for_ids["Residue Name"] = df2_for_ids["Updated_Description"].str.split().str[0]
                df2_for_ids = df2_for_ids[df2_for_ids["Residue Name"].isin(valid_amino_acids)]
                df2_for_ids["Residue Combined"] = (
                    df2_for_ids["Residue_Number_PDB"].astype(str) + "_" + df2_for_ids["Residue Name"]
                )

                # Assign Residue IDs to df2 based on unique Residue_Number + Residue_Name combinations
                residue_number = 0
                residue_ids = []
                for i in range(len(df2_for_ids)):
                    if i == 0 or df2_for_ids["Residue Combined"].iloc[i] != df2_for_ids["Residue Combined"].iloc[i - 1]:
                        residue_number += 1
                    residue_ids.append(residue_number)

                df2_for_ids["Residue ID"] = residue_ids
                df2_for_ids.drop(columns=["Residue Combined"], inplace=True)

                # Save the ID-added files
                df1_output_path = df1_path.replace(".csv", "_ID_Added.csv")
                df2_output_path = output_file_path.replace(".csv", "_ID_Added.csv")

                df1_for_ids.to_csv(df1_output_path, index=False)
                df2_for_ids.to_csv(df2_output_path, index=False)

                logger.info(f"Added Residue IDs: {df1_output_path}, {df2_output_path}")
                print(f"Updated final files with Residue ID: {df1_output_path}, {df2_output_path}")
                processed_count += 1

                summary_records.append(
                    {
                        "base_name": base_name,
                        "pdb_csv": df1_output_path,
                        "alignment_csv": df2_output_path,
                    }
                )

            except Exception as e:
                logger.error(f"Error adding Residue IDs for {base_name}: {e}")
                print(f"Error adding Residue IDs for {base_name}: {e}")

        except Exception as e:
            logger.error(f"Error aligning {subdir_path}: {e}")
            print(f"Error aligning {subdir_path}: {e}")

    logger.info(f"Combined PDB copy and alignment complete. Processed {processed_count} protein datasets.")
    print(f"Processed {processed_count} datasets with PDB copy and alignment.")


root_dir = CONFIG["output_dir"]
raw_pdb_dir = CONFIG["raw_pdb_dir"]


# === Map Chain/Residue/Residue Number Info with PDB ===


def process_all_files_in_subfolders(root_dir, failed_files_log_path, comparison_results_path):
    failed_files = []
    comparison_results = []

    for subdir, dirs, files in os.walk(root_dir):
        base_name = os.path.basename(subdir)

        df1_filename = f"{base_name}_NEW_ID_Added.csv"
        df2_filename = f"{base_name}_ID_Added.csv"

        if df1_filename in files and df2_filename in files:
            df1_path = os.path.join(subdir, df1_filename)
            df2_path = os.path.join(subdir, df2_filename)

            try:
                df1 = pd.read_csv(df1_path)
                df2 = pd.read_csv(df2_path)

                residue_mapping = df2[
                    ["Residue Sequence Number", "Residue ID", "Chain ID"]
                ].drop_duplicates()
                residue_mapping_dict = residue_mapping.set_index("Residue ID").to_dict(
                    "index"
                )

                df1["Residue_Number_PDB"] = df1["Residue ID"].map(
                    lambda x: residue_mapping_dict.get(x, {}).get(
                        "Residue Sequence Number"
                    )
                )
                df1["Chain ID"] = df1["Residue ID"].map(
                    lambda x: residue_mapping_dict.get(x, {}).get("Chain ID", "")
                )

                output_file_path = os.path.join(
                    subdir, os.path.basename(df1_path).replace(".csv", "_Merged.csv")
                )
                df1.to_csv(output_file_path, index=False)

                print(f"Processed and saved: {output_file_path}")

                last_row_df2 = df2.iloc[-1]
                residue_seq_number_df2 = last_row_df2["Residue Sequence Number"]
                chain_id_df2 = last_row_df2["Chain ID"]
                residue_name_df2 = last_row_df2["Residue Name"]

                df_output = pd.read_csv(output_file_path)
                last_row_output = df_output.iloc[-1]
                residue_number_output = last_row_output["Residue_Number_PDB"]
                chain_id_output = last_row_output["Chain ID"]
                residue_name_output = last_row_output["Residue Name"]

                comparison_results.append(
                    {
                        "File": output_file_path,
                        "Residue Sequence Number (df2)": residue_seq_number_df2,
                        "Residue_Number_PDB (Output)": residue_number_output,
                        "Chain ID (df2)": chain_id_df2,
                        "Chain ID (Output)": chain_id_output,
                        "Residue Name (df2)": residue_name_df2,
                        "Residue Name (Output)": residue_name_output,
                    }
                )

                column_to_check = "Residue_Number_PDB"
                empty_cell_rows = df_output[
                    df_output[column_to_check].isna()
                    & df_output.drop(columns=[column_to_check]).notna().any(axis=1)
                ]

                if not empty_cell_rows.empty:
                    raise ValueError(
                        "Empty cells detected in column 'Residue_Number' while other "
                        "columns have values in the output file."
                    )

            except Exception as e:
                print(f"Failed to process {df1_path} and {df2_path}: {e}")
                failed_files.append(
                    {"df1_file": df1_path, "df2_file": df2_path, "error": str(e)}
                )

    if failed_files:
        with open(failed_files_log_path, "w", newline="") as csvfile:
            fieldnames = ["df1_file", "df2_file", "output_file", "error"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(failed_files)

        print(f"Failed files recorded in {failed_files_log_path}")

    if comparison_results:
        with open(comparison_results_path, "w", newline="") as csvfile:
            fieldnames = [
                "File",
                "Residue Sequence Number (df2)",
                "Residue_Number_PDB (Output)",
                "Chain ID (df2)",
                "Chain ID (Output)",
                "Residue Name (df2)",
                "Residue Name (Output)",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(comparison_results)

        print(f"Comparison results saved to {comparison_results_path}")


root_dir = CONFIG["output_dir"]
failed_files_log_path = os.path.join(CONFIG["output_dir"], "failed_files_log.csv")
comparison_results_path = os.path.join(CONFIG["output_dir"], "comparison_results_1.csv")


# === Combine all the Updated CSV Files with Bonds Info ===


def combine_csv_files_in_subfolders(root_dir, output_file_path, columns_to_keep):
    # List to store DataFrames from all CSV files
    combined_data = []

    # Walk through the directory and its subfolders
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a CSV file - look for either _Merged.csv
            if file.endswith("_Merged.csv"):
                # Construct the full file path
                file_path = os.path.join(subdir, file)
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path)

                    # Rename existing 'Residue_Number' column to 'Residue_Number_PDB' to avoid conflicts
                    if "Residue_Number" in df.columns:
                        df = df.rename(columns={"Residue_Number": "Residue_Number_PDB"})

                    # Select only the columns that exist in this file
                    available_cols = [col for col in columns_to_keep if col in df.columns]
                    
                    # Log which columns are missing
                    missing_cols = [col for col in columns_to_keep if col not in df.columns]
                    if missing_cols:
                        logger.debug(f"Missing columns in {file}: {missing_cols}")
                    
                    if available_cols:
                        df_selected = df[available_cols]

                        # Add a new column with the base name of the file
                        pdb_name = file.split("_")[0]
                        df_selected.insert(0, "PDB", pdb_name)

                        # Append the DataFrame to the list
                        combined_data.append(df_selected)
                        print(f"Successfully read: {file_path} ({len(df_selected)} rows, {len(available_cols)} columns)")
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    logger.error(f"Failed to read {file_path}: {e}")

    # Combine all DataFrames into a single DataFrame
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Save to a new CSV file
        combined_df.to_csv(output_file_path, index=False)
        print(
            f"All CSV files have been combined and saved to {output_file_path}. Total rows: {len(combined_df)}"
        )
    else:
        print("No CSV files found to combine.")


output_file_path = os.path.join(CONFIG["output_dir"], "Edge_Masterfile.csv")
columns_to_keep = [
    "atom_number",
    "atom_name",
    "Residue Name",
    "Chain ID",
    "Residue_Number_PDB",
    "Residue ID",
    "recalculated_x",
    "recalculated_y",
    "recalculated_z",
    "Dipole_Vector",
    "bonds",
]


# === Generate PDB Files from Final CSV Data for Feature Extracting ===


PDB_DIR = CONFIG["output_dir"]


def generate_complete_pdb_from_csv(pdb_dir):
    """
    Generate complete PDB files from individual _Merged.csv files in each subfolder.
    """
    processing_log = os.path.join(pdb_dir, "pdb_generation_from_csv.log")

    with open(processing_log, "w") as log:
        log.write("PDB Generation from CSV Log:\n")
        log.write("=" * 50 + "\n")

    success_count = 0
    failed_count = 0

    # Process individual _Merged.csv files in each subfolder
    for root, dirs, files in os.walk(pdb_dir):
        final_csv_files = [f for f in files if f.endswith("_Merged.csv")]

        if not final_csv_files:
            continue

        for csv_file in final_csv_files:
            csv_path = os.path.join(root, csv_file)
            pdb_id = csv_file.replace("_Merged.csv", "")
            output_dir = root  # Save PDB in the same folder as the CSV

            print(f"\nProcessing {csv_file} for {pdb_id}")
            logger.info(f"Processing {csv_file} from {root}")
            
            if create_pdb_from_merged_csv(csv_path, output_dir, pdb_id, processing_log):
                success_count += 1
            else:
                failed_count += 1
                with open(processing_log, "a") as log:
                    log.write(f"FAILED: {pdb_id} - PDB generation unsuccessful\n")

    print("\n=== PDB Generation Summary ===")
    print(f"Successfully generated: {success_count} PDB files")
    print(f"Failed: {failed_count} PDB files")
    print(f"Log file: {processing_log}")
    logger.info(f"PDB generation complete: {success_count} successful, {failed_count} failed")


def create_pdb_from_merged_csv(csv_path, output_dir, pdb_id, log_path):
    """Create a complete PDB file from Merged CSV data."""
    try:
        # Read the Merged CSV file
        df = pd.read_csv(csv_path)

        # Check if required columns exist
        required_cols = [
            "x",
            "y",
            "z",
            "atom_name",
            "Residue Name",
            "Chain ID",
            "Residue ID",
            "atom_type",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            with open(log_path, "a") as log:
                log.write(f"CSV missing required columns: {missing_cols}\n")
                log.write(f"Available columns: {list(df.columns)}\n")
            return False

        print(f"    Available columns: {list(df.columns)}")
        print(f"    Processing {len(df)} atoms")

        # Create PDB lines
        pdb_lines = []

        for i, row in df.iterrows():
            # Get coordinates
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])

            # Get atom information from CSV columns
            atom_serial = int(row["atom_number"]) if "atom_number" in df.columns else i + 1
            atom_name = str(row["atom_name"]).strip()[:4].ljust(4)
            alt_loc = " "

            # Convert full residue name to 3-letter code
            full_residue_name = str(row["Residue Name"]).strip()
            residue_name = AMINO_ACID_FULL_TO_3.get(full_residue_name, full_residue_name[:3].upper()).ljust(3)

            chain_id = str(row["Chain ID"]).strip()[:1]
            residue_number = int(row["Residue ID"])
            insertion_code = " "
            occupancy = 1.00
            temp_factor = 0.00
            element = str(row["atom_name"]).strip()[0:1].ljust(2)
            charge = ""

            pdb_line = (
                f"ATOM  "
                f"{atom_serial:5d}"
                f" "
                f"{atom_name:4s}"
                f"{alt_loc:1s}"
                f"{residue_name:3s}"
                f" "
                f"{chain_id:1s}"
                f"{residue_number:4d}"
                f"{insertion_code:1s}"
                f"   "
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"{occupancy:6.2f}"
                f"{temp_factor:6.2f}"
                f"          "
                f"{element:2s}"
                f"{charge:2s}"
                f"\n"
            )

            pdb_lines.append(pdb_line)

        # Create output filename
        output_filename = f"{pdb_id}_refined_coordinates.pdb"
        output_path = os.path.join(output_dir, output_filename)

        # Write PDB file with proper header and footer
        with open(output_path, "w") as f:
            # Write header
            f.write(f"HEADER    MOLECULAR DYNAMICS STRUCTURE       {pdb_id}\n")
            f.write("TITLE     STRUCTURE GENERATED FROM FINAL CSV DATA\n")
            f.write("REMARK    ENERGY MINIMIZED COORDINATES FROM TINKER\n")
            f.write(f"REMARK    TOTAL ATOMS: {len(pdb_lines)}\n")
            f.write("MODEL        1\n")

            # Write all ATOM records
            f.writelines(pdb_lines)

            # Write footer
            f.write("ENDMDL\n")
            f.write("END\n")

        with open(log_path, "a") as log:
            log.write(
                f"SUCCESS: {pdb_id} -> {len(pdb_lines)} atoms generated in PDB format\n"
            )

        print(f"    Generated {len(pdb_lines)} atoms in PDB format")
        return True

    except Exception as e:
        with open(log_path, "a") as log:
            log.write(f"Exception in create_pdb_from_merged_csv: {str(e)}\n")
        return False


def create_pdb_generation_summary(pdb_dir):
    """Create a summary report of all generated PDB files."""
    summary_data = []

    for root, dirs, files in os.walk(pdb_dir):
        generated_files = [f for f in files if f.endswith("_refined_coordinates.pdb")]

        for pdb_file in generated_files:
            pdb_path = os.path.join(root, pdb_file)

            # Count atoms in generated file
            atom_count = 0
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        atom_count += 1

            # Extract information
            base_name = pdb_file.replace("_refined_coordinates.pdb", "")
            folder_name = os.path.basename(root)

            # Get file size
            file_size = os.path.getsize(pdb_path)

            summary_data.append(
                {
                    "Folder": folder_name,
                    "PDB_ID": base_name,
                    "Generated_File": pdb_file,
                    "Atom_Count": atom_count,
                    "File_Size_KB": round(file_size / 1024, 2),
                    "Full_Path": pdb_path,
                }
            )

    # Save summary to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(pdb_dir, "generated_pdb_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n=== Generated PDB Summary ===")
        print(f"Total PDB files generated: {len(summary_data)}")
        print(f"Total atoms across all files: {summary_df['Atom_Count'].sum()}")
        print(f"Average atoms per file: {summary_df['Atom_Count'].mean():.1f}")
        print(f"Summary saved to: {summary_path}")

        return summary_path
    else:
        print("No generated PDB files found")
        return None


def calculate_neighbor_heavy_atoms(output_dir):
    """Step 11: Count neighbor heavy atoms within specified radii for each atom.
    
    Outputs results to each PDB's subfolder as CSV files.
    """
    try:
        # Find all PDB files in PDB-ID named subfolders
        pdb_files = find_pdb_files(output_dir)
        
        if not pdb_files:
            logger.warning("No refined PDB files found for neighbor atom analysis")
            return
        
        logger.info(f"Found {len(pdb_files)} PDB file(s) for neighbor analysis")
        
        # Radii to consider
        radii = [7, 8, 9, 10, 11]
        
        # Process each PDB file
        for pdb_file in pdb_files:
            try:
                logger.info(f"Processing neighbor atoms for: {pdb_file}")
                
                # Get output folder (same as PDB file location)
                output_folder = os.path.dirname(pdb_file)
                output_csv = os.path.join(output_folder, "neighbor_atom_counts.csv")
                
                # Load structure
                u = mda.Universe(pdb_file)
                all_atoms = u.atoms
                results = []
                
                # Process each atom
                for atom in all_atoms:
                    atom_group = u.atoms[atom.index : atom.index + 1]
                    atom_type_counts = {}
                    
                    for radius in radii:
                        atom_type_counts[f"Radius_{radius}A_N_Count"] = 0
                        atom_type_counts[f"Radius_{radius}A_CA_C_Count"] = 0
                        atom_type_counts[f"Radius_{radius}A_O_Count"] = 0
                        atom_type_counts[f"Radius_{radius}A_S_Count"] = 0
                    
                    for radius in radii:
                        try:
                            sphere_atoms = u.select_atoms(
                                f"around {radius} group atom_group", atom_group=atom_group
                            )
                            filtered_atoms = sphere_atoms.select_atoms(
                                f"not (resid {atom.resid}) and not (resname HOH and name O) and not (name H or name HO or name HN)"
                            )
                            
                            for neighbor_atom in filtered_atoms:
                                if neighbor_atom.name == "N":
                                    atom_type_counts[f"Radius_{radius}A_N_Count"] += 1
                                elif neighbor_atom.name in ["CA", "C"]:
                                    atom_type_counts[f"Radius_{radius}A_CA_C_Count"] += 1
                                elif neighbor_atom.name == "O":
                                    atom_type_counts[f"Radius_{radius}A_O_Count"] += 1
                                elif neighbor_atom.name == "S":
                                    atom_type_counts[f"Radius_{radius}A_S_Count"] += 1
                        except Exception as e:
                            logger.warning(f"Error processing radius {radius} for atom {atom.index}: {e}")
                    
                    results.append({
                        "PDB": extract_pdb_id_from_pdb_file(pdb_file),
                        "atom_number": atom.index + 1,
                        "Chain ID": atom.segment.segid if atom.segment else "Unknown",
                        "Residue ID": atom.resid,
                        "Residue Name": atom.resname,
                        "atom_name": atom.name,
                        **atom_type_counts,
                    })
                
                # Save results
                df = pd.DataFrame(results)
                df.to_csv(output_csv, index=False)
                logger.info(f"Neighbor atom counts saved to {output_csv}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdb_file}: {e}")
                continue
                
    except ImportError:
        logger.error("MDAnalysis not installed. Skipping neighbor atom calculation.")
    except Exception as e:
        logger.error(f"Error in calculate_neighbor_heavy_atoms: {e}")


def calculate_hbonds_and_sasa_per_atom(output_dir):
    """Step 12: Calculate hydrogen bonds and SASA per atom.
    
    Outputs results to each PDB's subfolder as separate CSV files.
    """
    try:
        # Find all PDB files in PDB-ID named subfolders
        pdb_files = find_pdb_files(output_dir)
        
        if not pdb_files:
            logger.warning("No refined PDB files found for H-bond and SASA analysis")
            return
        
        logger.info(f"Found {len(pdb_files)} PDB file(s) for H-bond/SASA analysis")
        
        def calculate_hbonds_and_sasa(pdb_file):
            """Calculate H-bonds and SASA for a single PDB file."""
            try:
                traj = md.load(pdb_file)
                hbonds = md.baker_hubbard(traj, periodic=False)
                
                donor_counts = defaultdict(int)
                acceptor_counts = defaultdict(int)
                
                for donor_idx, hydrogen_idx, acceptor_idx in hbonds:
                    donor_atom = traj.topology.atom(donor_idx)
                    acceptor_atom = traj.topology.atom(acceptor_idx)
                    donor_counts[donor_atom] += 1
                    acceptor_counts[acceptor_atom] += 1
                
                sasa_values = md.shrake_rupley(traj, mode="atom")
                
                hbonds_data = []
                sasa_data = []
                
                for atom in traj.topology.atoms:
                    chain_id = atom.residue.chain.chain_id if atom.residue.chain else "Unknown"
                    
                    hbonds_data.append({
                        "PDB": extract_pdb_id_from_pdb_file(pdb_file),
                        "atom_number": atom.index + 1,
                        "Chain ID": chain_id,
                        "Residue ID": atom.residue.resSeq,
                        "Residue Name": atom.residue.name,
                        "atom_name": atom.name,
                        "Number of H-Bonds as donor": donor_counts[atom],
                        "Number of H-Bonds as acceptor": acceptor_counts[atom],
                    })
                    
                    sasa_data.append({
                        "PDB": extract_pdb_id_from_pdb_file(pdb_file),
                        "atom_number": atom.index + 1,
                        "Chain ID": chain_id,
                        "Residue ID": atom.residue.resSeq,
                        "Residue Name": atom.residue.name,
                        "atom_name": atom.name,
                        "SASA_Value": sasa_values[0][atom.index],
                    })
                
                return hbonds_data, sasa_data
                
            except Exception as e:
                logger.error(f"Error calculating H-bonds/SASA for {pdb_file}: {e}")
                return None, None
        
        # Process each PDB file
        for pdb_file in pdb_files:
            try:
                logger.info(f"Processing H-bonds/SASA for: {pdb_file}")
                
                # Get output folder (same as PDB file location)
                output_folder = os.path.dirname(pdb_file)
                hbonds_output = os.path.join(output_folder, "hbonds_per_atom.csv")
                sasa_output = os.path.join(output_folder, "sasa_per_atom.csv")
                
                # Calculate H-bonds and SASA
                hbonds_data, sasa_data = calculate_hbonds_and_sasa(pdb_file)
                
                if hbonds_data is not None and sasa_data is not None:
                    # Save H-bonds
                    hbonds_df = pd.DataFrame(hbonds_data)
                    hbonds_df.to_csv(hbonds_output, index=False)
                    logger.info(f"H-bonds saved to {hbonds_output}")
                    
                    # Save SASA
                    sasa_df = pd.DataFrame(sasa_data)
                    sasa_df.to_csv(sasa_output, index=False)
                    logger.info(f"SASA saved to {sasa_output}")
                else:
                    logger.warning(f"Failed to calculate H-bonds/SASA for {pdb_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")
                continue
                
    except ImportError:
        logger.error("mdtraj not installed. Skipping H-bond and SASA calculation.")
    except Exception as e:
        logger.error(f"Error in calculate_hbonds_and_sasa_per_atom: {e}")


def merge_feature_vectors_from_subfolders(output_dir):
    """Step 13: Merge neighbor atoms, H-bonds, and SASA features with:
    1. Individual PDB NEW_ID_Added_Merged.csv files (per-PDB merge)
    2. The main edge masterfile (global merge)
    
    Reads features from each PDB subfolder and creates:
    - Individual merged files with features for each PDB
    - A global merged file combining all PDBs
    """
    try:
        # Read the main masterfile
        masterfile_path = os.path.join(output_dir, "Edge_Masterfile.csv")
        if not os.path.exists(masterfile_path):
            logger.warning(f"Masterfile not found: {masterfile_path}")
            return

        pdb_df = pd.read_csv(masterfile_path)
        logger.info(f"Loaded masterfile with {len(pdb_df)} rows and {len(pdb_df.columns)} columns")
        logger.info(f"Masterfile columns: {list(pdb_df.columns)}")
        logger.info(f"Masterfile PDB unique values: {pdb_df['PDB'].nunique()}")

        # Collect feature dataframes from all subfolders AND perform per-PDB merges
        all_neighbor_data = []
        all_hbonds_data = []
        all_sasa_data = []
        
        pdb_merge_count = 0

        for entry in os.scandir(output_dir):
            if not entry.is_dir():
                continue

            pdb_id = os.path.basename(entry.path)
            neighbor_csv = os.path.join(entry.path, "neighbor_atom_counts.csv")
            hbonds_csv = os.path.join(entry.path, "hbonds_per_atom.csv")
            sasa_csv = os.path.join(entry.path, "sasa_per_atom.csv")
            pdb_merged_csv = os.path.join(entry.path, f"{pdb_id}_NEW_ID_Added_Merged.csv")

            try:
                # Load feature files for this PDB
                df_neighbor = None
                df_hbonds = None
                df_sasa = None
                
                if os.path.exists(neighbor_csv):
                    df_neighbor = pd.read_csv(neighbor_csv)
                    all_neighbor_data.append(df_neighbor)
                    logger.debug(f"Loaded neighbor CSV for {pdb_id}: {len(df_neighbor)} rows")

                if os.path.exists(hbonds_csv):
                    df_hbonds = pd.read_csv(hbonds_csv)
                    all_hbonds_data.append(df_hbonds)
                    logger.debug(f"Loaded hbonds CSV for {pdb_id}: {len(df_hbonds)} rows")

                if os.path.exists(sasa_csv):
                    df_sasa = pd.read_csv(sasa_csv)
                    all_sasa_data.append(df_sasa)
                    logger.debug(f"Loaded SASA CSV for {pdb_id}: {len(df_sasa)} rows")
                
                # === PER-PDB MERGE ===
                # Merge features with individual PDB's NEW_ID_Added_Merged.csv file
                if os.path.exists(pdb_merged_csv) and df_neighbor is not None and df_hbonds is not None and df_sasa is not None:
                    try:
                        logger.info(f"Performing per-PDB merge for {pdb_id}")
                        pdb_individual_df = pd.read_csv(pdb_merged_csv)
                        logger.debug(f"Loaded PDB individual file {pdb_id}: {len(pdb_individual_df)} rows")
                        
                        # Add PDB column if it doesn't exist
                        if "PDB" not in pdb_individual_df.columns:
                            pdb_individual_df.insert(0, "PDB", pdb_id.upper())
                        
                        
                        # Create merge keys for per-PDB merge
                        # Normalize names in individual PDB file
                        pdb_individual_df["Residue Name"] = pdb_individual_df["Residue Name"].apply(normalize_residue_name)
                        pdb_individual_df["atom_name_normalized"] = pdb_individual_df["atom_name"].apply(normalize_atom_name_for_merge)
                        
                        # Normalize names in feature files
                        df_neighbor_pdb = df_neighbor.copy()
                        df_hbonds_pdb = df_hbonds.copy()
                        df_sasa_pdb = df_sasa.copy()
                        
                        df_neighbor_pdb["Residue Name"] = df_neighbor_pdb["Residue Name"].apply(normalize_residue_name)
                        df_neighbor_pdb["PDB"] = df_neighbor_pdb["PDB"].str.upper()
                        
                        df_hbonds_pdb["Residue Name"] = df_hbonds_pdb["Residue Name"].apply(normalize_residue_name)
                        df_hbonds_pdb["PDB"] = df_hbonds_pdb["PDB"].str.upper()
                        
                        df_sasa_pdb["Residue Name"] = df_sasa_pdb["Residue Name"].apply(normalize_residue_name)
                        df_sasa_pdb["PDB"] = df_sasa_pdb["PDB"].str.upper()
                        
                        # Perform per-PDB merges using composite keys: PDB, atom_number, Chain ID, Residue_Number
                        pdb_merged = pdb_individual_df.copy()
                        logger.debug(f"Starting per-PDB merge with {len(pdb_merged)} rows")
                        
                        # Merge neighbor atoms - exclude columns not in merge key
                        neighbor_cols_to_merge = [col for col in df_neighbor_pdb.columns if col not in ["PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name"]]
                        df_neighbor_pdb_to_merge = df_neighbor_pdb[["PDB", "atom_number", "Chain ID", "Residue ID"] + neighbor_cols_to_merge]
                        
                        pdb_merged = pd.merge(
                            pdb_merged, 
                            df_neighbor_pdb_to_merge, 
                            on=["PDB", "atom_number", "Chain ID", "Residue ID"],
                            how="left",
                            suffixes=("", "_neighbor")
                        )
                        logger.debug(f"After neighbor merge: {len(pdb_merged)} rows")
                        
                        # Merge H-bonds
                        hbond_cols_to_merge = [col for col in df_hbonds_pdb.columns if col not in ["PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name"]]
                        df_hbonds_pdb_to_merge = df_hbonds_pdb[["PDB", "atom_number", "Chain ID", "Residue ID"] + hbond_cols_to_merge]
                        
                        pdb_merged = pd.merge(
                            pdb_merged,
                            df_hbonds_pdb_to_merge,
                            on=["PDB", "atom_number", "Chain ID", "Residue ID"],
                            how="left",
                            suffixes=("", "_hbonds")
                        )
                        logger.debug(f"After H-bonds merge: {len(pdb_merged)} rows")
                        
                        # Merge SASA - columns already match after standardization
                        sasa_cols_to_merge = [col for col in df_sasa_pdb.columns if col not in ["PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name"]]
                        df_sasa_pdb_to_merge = df_sasa_pdb[["PDB", "atom_number", "Chain ID", "Residue ID"] + sasa_cols_to_merge]
                        
                        pdb_merged = pd.merge(
                            pdb_merged,
                            df_sasa_pdb_to_merge,
                            on=["PDB", "atom_number", "Chain ID", "Residue ID"],
                            how="left",
                            suffixes=("", "_sasa")
                        )
                        logger.debug(f"After SASA merge: {len(pdb_merged)} rows")
                        
                        # Clean up temporary columns
                        cols_to_drop = ["atom_name_normalized"]
                        pdb_merged = pdb_merged.drop([col for col in cols_to_drop if col in pdb_merged.columns], axis=1)
                        
                        # Save per-PDB merged file
                        pdb_output_path = os.path.join(entry.path, f"{pdb_id}_NEW_ID_Added_Merged_with_features.csv")
                        pdb_merged.to_csv(pdb_output_path, index=False)
                        logger.info(f"Per-PDB merged file saved for {pdb_id}: {pdb_output_path} ({len(pdb_merged)} rows)")
                        pdb_merge_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error performing per-PDB merge for {pdb_id}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error reading feature files from {pdb_id}: {e}")

        logger.info(f"Per-PDB merges completed: {pdb_merge_count} PDB files merged with their features")
        
        if not all_neighbor_data or not all_hbonds_data or not all_sasa_data:
            logger.warning("Not all feature files found in subfolders")
            logger.warning(f"  Neighbor files: {len(all_neighbor_data)}")
            logger.warning(f"  H-bonds files: {len(all_hbonds_data)}")
            logger.warning(f"  SASA files: {len(all_sasa_data)}")
            return

        # Combine all feature dataframes
        df_neighbor = pd.concat(all_neighbor_data, ignore_index=True)
        df_hbonds = pd.concat(all_hbonds_data, ignore_index=True)
        df_sasa = pd.concat(all_sasa_data, ignore_index=True)

        logger.info(f"Combined {len(df_neighbor)} neighbor atom records")
        logger.info(f"Neighbor columns: {list(df_neighbor.columns)}")
        logger.info(f"Combined {len(df_hbonds)} H-bond records")
        logger.info(f"H-bonds columns: {list(df_hbonds.columns)}")
        logger.info(f"Combined {len(df_sasa)} SASA records")
        logger.info(f"SASA columns: {list(df_sasa.columns)}")

        # Clean up PDB names - all three now use "PDB" column
        df_neighbor["PDB"] = df_neighbor["PDB"].str.upper()
        df_hbonds["PDB"] = df_hbonds["PDB"].str.upper()
        df_sasa["PDB"] = df_sasa["PDB"].str.upper()

        logger.info(f"Neighbor PDB unique: {df_neighbor['PDB'].nunique()}")
        logger.info(f"H-bonds PDB unique: {df_hbonds['PDB'].nunique()}")
        logger.info(f"SASA PDB unique: {df_sasa['PDB'].nunique()}")

        # Normalize residue names to match between dataframes
        pdb_df["Residue Name"] = pdb_df["Residue Name"].apply(normalize_residue_name)
        df_neighbor["Residue Name"] = df_neighbor["Residue Name"].apply(
            normalize_residue_name
        )
        df_hbonds["Residue Name"] = df_hbonds["Residue Name"].apply(
            normalize_residue_name
        )
        df_sasa["Residue Name"] = df_sasa["Residue Name"].apply(
            normalize_residue_name
        )

        # Create normalized atom names for merge
        pdb_df["atom_name_normalized"] = pdb_df["atom_name"].apply(
            normalize_atom_name_for_merge
        )
        df_hbonds["atom_name_normalized"] = df_hbonds["atom_name"].apply(
            normalize_atom_name_for_merge
        )
        df_sasa["atom_name_normalized"] = df_sasa["atom_name"].apply(
            normalize_atom_name_for_merge
        )

        logger.info("Normalized residue names and atom names")

        # Log sample data for debugging
        logger.info(f"Sample from pdb_df: PDB={pdb_df['PDB'].iloc[0]}, atom_number={pdb_df['atom_number'].iloc[0]}, Residue ID={pdb_df['Residue ID'].iloc[0]}")
        logger.info(f"Sample from df_neighbor: PDB={df_neighbor['PDB'].iloc[0]}, atom_number={df_neighbor['atom_number'].iloc[0]}, Residue ID={df_neighbor['Residue ID'].iloc[0]}")
        logger.info(f"Sample from df_hbonds: PDB={df_hbonds['PDB'].iloc[0]}, atom_number={df_hbonds['atom_number'].iloc[0]}, Residue ID={df_hbonds['Residue ID'].iloc[0]}")

        # Create a merge key for more robust joining
        pdb_df["merge_key"] = (
            pdb_df["PDB"].astype(str) + "_" + 
            pdb_df["Chain ID"].astype(str) + "_" +
            pdb_df["Residue ID"].astype(str) + "_" +
            pdb_df["atom_number"].astype(str)
        )
        
        df_neighbor["merge_key"] = (
            df_neighbor["PDB"].astype(str) + "_" + 
            df_neighbor["Chain ID"].astype(str) + "_" +
            df_neighbor["Residue ID"].astype(str) + "_" +
            df_neighbor["atom_number"].astype(str)
        )

        df_hbonds["merge_key"] = (
            df_hbonds["PDB"].astype(str) + "_" + 
            df_hbonds["Chain ID"].astype(str) + "_" +
            df_hbonds["Residue ID"].astype(str) + "_" +
            df_hbonds["atom_number"].astype(str)
        )

        df_sasa["merge_key"] = (
            df_sasa["PDB"].astype(str) + "_" + 
            df_sasa["Chain ID"].astype(str) + "_" +
            df_sasa["Residue ID"].astype(str) + "_" +
            df_sasa["atom_number"].astype(str)
        )

        logger.info(f"Created merge keys. Sample pdb_df key: {pdb_df['merge_key'].iloc[0]}")
        logger.info(f"Sample neighbor key: {df_neighbor['merge_key'].iloc[0]}")

        # Perform merges using merge keys
        merged_df = pdb_df.copy()
        logger.info(f"Starting merge with {len(merged_df)} rows from masterfile")

        # Merge neighbor atoms
        neighbor_cols = [col for col in df_neighbor.columns if col not in ["merge_key", "PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name"]]
        df_neighbor_to_merge = df_neighbor[["merge_key"] + neighbor_cols].drop_duplicates()
        
        merged_df = pd.merge(
            merged_df,
            df_neighbor_to_merge,
            on="merge_key",
            how="left",
            suffixes=("", "_neighbor"),
        )
        logger.info(f"After neighbor merge: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        # Merge H-bonds
        hbond_cols = [col for col in df_hbonds.columns if col not in ["merge_key", "PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name", "atom_name_normalized"]]
        df_hbonds_to_merge = df_hbonds[["merge_key"] + hbond_cols].drop_duplicates()
        
        merged_df = pd.merge(
            merged_df,
            df_hbonds_to_merge,
            on="merge_key",
            how="left",
            suffixes=("", "_hbonds"),
        )
        logger.info(f"After H-bonds merge: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        # Merge SASA
        sasa_cols = [col for col in df_sasa.columns if col not in ["merge_key", "PDB", "Chain ID", "Residue ID", "atom_number", "Residue Name", "atom_name", "atom_name_normalized"]]
        df_sasa_to_merge = df_sasa[["merge_key"] + sasa_cols].drop_duplicates()
        
        merged_df = pd.merge(
            merged_df,
            df_sasa_to_merge,
            on="merge_key",
            how="left",
            suffixes=("", "_sasa"),
        )
        logger.info(f"After SASA merge: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        # Clean up temporary columns
        cols_to_drop = ["merge_key", "atom_name_normalized"]
        if "atom_name_normalized" in merged_df.columns:
            cols_to_drop.append("atom_name_normalized")
        
        merged_df = merged_df.drop([col for col in cols_to_drop if col in merged_df.columns], axis=1)

        logger.info(f"Final merged dataframe has {len(merged_df.columns)} columns")

        # Check how many non-null values in feature columns
        feature_cols = [col for col in merged_df.columns if "Radius" in col or "H-Bond" in col or "SASA" in col]
        non_null_counts = merged_df[feature_cols].notna().sum().sum()
        logger.info(f"Total non-null feature values: {non_null_counts}")

        # Filter to only include ionizable residues: ASP, GLU, LYS, HIS
        ionizable_residues = ["ASP", "GLU", "LYS", "HIS"]
        merged_df_filtered = merged_df[merged_df["Residue Name"].isin(ionizable_residues)]
        logger.info(f"Filtered merged dataframe to ionizable residues: {len(merged_df_filtered)} rows (from {len(merged_df)} total rows)")
        
        # Save the filtered merged dataframe
        output_path = os.path.join(output_dir, "merged_features_with_neighbors_hbonds_sasa.csv")
        merged_df_filtered.to_csv(output_path, index=False)
        logger.info(f"Merged feature vectors saved to {output_path}")
        
        logger.info(f"=== MERGE SUMMARY ===")
        logger.info(f"Per-PDB merges: {pdb_merge_count} files created (one per PDB)")
        logger.info(f"Global merge: 1 file created with all PDBs combined")
        logger.info(f"Global merge file: {output_path}")
        logger.info(f"Residues included: ASP, GLU, LYS, HIS (ionizable residues only)")

    except Exception as e:
        logger.error(f"Error in merge_feature_vectors_from_subfolders: {e}", exc_info=True)


def get_adjacency_matrix():
    """Generate adjacency matrices from merged PDB features.
    
    Creates adjacency matrices for each residue based on bonding information,
    and saves them to CSV format with self-loops included.
    
    Excludes PDBs with NaN values in Residue_Number_PDB column and PDBs from failed file processing.
    """
    edges_input = os.path.join(CONFIG["output_dir"], "merged_features_with_neighbors_hbonds_sasa.csv")
    edges_output = os.path.join(CONFIG["output_dir"], "merged_PDB_Features_bonds_updated.csv")
    failed_pdb_log = os.path.join(CONFIG["output_dir"], "Failed_Feature_Generation_PDBs.csv")
    failed_files_log = os.path.join(CONFIG["output_dir"], "failed_files_log.csv")

    try:
        df_edges = pd.read_csv(edges_input)
    except FileNotFoundError:
        logger.error(f"Input file not found: {edges_input}")
        return

    # === LOAD PDBs FAILED IN PREVIOUS PROCESSING STEP ===
    failed_pdb_from_files = []
    if os.path.exists(failed_files_log):
        try:
            df_failed_files = pd.read_csv(failed_files_log)
            # Extract PDB ID from df1_file path (e.g., /path/to/1A6K/1A6K_NEW_ID_Added.csv -> 1A6K)
            failed_pdb_from_files = [os.path.basename(os.path.dirname(fpath)) for fpath in df_failed_files["df1_file"].tolist()]
            logger.info(f"Loaded {len(failed_pdb_from_files)} failed PDBs from {failed_files_log}")
            logger.info(f"Failed PDBs from file processing: {failed_pdb_from_files}")
        except Exception as e:
            logger.warning(f"Could not load failed files log: {e}")
    
    # === STEP 1: IDENTIFY PDBs WITH NaN IN Residue_Number_PDB ===
    logger.info("Checking for PDBs with NaN values in Residue_Number_PDB...")
    
    # Check which PDBs have NaN values in Residue_Number_PDB BEFORE conversion
    pdb_nan_counts = df_edges[df_edges["Residue_Number_PDB"].isna()].groupby("PDB").size().reset_index(name="NaN_Count")
    failed_pdb_list_nan = pdb_nan_counts["PDB"].tolist()
    
    # Combine both sources of failed PDBs
    failed_pdb_list = list(set(failed_pdb_list_nan + failed_pdb_from_files))
    
    if failed_pdb_list:
        logger.warning(f"Found {len(failed_pdb_list)} total PDB(s) to exclude from adjacency matrix generation")
        for pdb_id in failed_pdb_list:
            if pdb_id in pdb_nan_counts["PDB"].values:
                nan_count = pdb_nan_counts[pdb_nan_counts["PDB"] == pdb_id]["NaN_Count"].values[0]
                logger.warning(f"  - {pdb_id}: NaN values in Residue_Number_PDB ({nan_count} atoms)")
            else:
                logger.warning(f"  - {pdb_id}: Failed in file processing step")
        
        # Save failed PDB list to file
        reasons = []
        for pdb_id in failed_pdb_list:
            if pdb_id in pdb_nan_counts["PDB"].values:
                reasons.append("NaN values in Residue_Number_PDB")
            else:
                reasons.append("Failed in file processing step")
        failed_df = pd.DataFrame({"PDB": failed_pdb_list, "Reason": reasons})
        failed_df.to_csv(failed_pdb_log, index=False)
        logger.info(f"Failed PDB list saved to: {failed_pdb_log}")
    
    # === STEP 2: EXCLUDE FAILED PDBs FROM PROCESSING ===
    df_edges_filtered = df_edges[~df_edges["PDB"].isin(failed_pdb_list)].copy()
    
    if len(df_edges_filtered) == 0:
        logger.error("No valid PDBs remaining after filtering out those with NaN residue numbers!")
        return
    
    logger.info(f"Original records: {len(df_edges)}, After filtering failed PDBs: {len(df_edges_filtered)}")
    logger.info(f"Valid PDBs for feature generation: {df_edges_filtered['PDB'].nunique()}")

    # Format bonds if they exist
    def format_bonds(bond):
        try:
            if pd.isna(bond):
                return bond
            if isinstance(bond, str):
                bond_list = ast.literal_eval(bond)
                return ", ".join(map(str, bond_list))
            if isinstance(bond, list):
                return ", ".join(map(str, bond))
            return str(bond)
        except (ValueError, SyntaxError, TypeError):
            return str(bond)

    df_edges_filtered["bonds"] = df_edges_filtered["bonds"].apply(format_bonds)
    logger.info("Bonds formatting complete.")

    df_edges_filtered.to_csv(edges_output, index=False)
    logger.info(f"Formatted edges saved to {edges_output}")

    df = df_edges_filtered.copy()

    df["Residue_Name"] = df["Residue Name"].str.strip()
    
    # Now safe to convert Residue_Number_PDB to int (no NaN values remain)
    df["Residue_Number_PDB"] = pd.to_numeric(df["Residue_Number_PDB"], errors='coerce').astype(int)

    save_directory = os.path.join(
        CONFIG["output_dir"], "Adjacency_Matrices/With_Self_Loop"
    )
    os.makedirs(save_directory, exist_ok=True)

    def create_all_atom_adjacency_matrix(df, pdb_id, chain_id, residue_number, residue_name):
        residue_df = df[
            (df["PDB"] == pdb_id)
            & (df["Chain ID"] == chain_id)
            & (df["Residue_Number_PDB"] == residue_number)
            & (df["Residue_Name"] == residue_name)
        ]

        num_atoms = residue_df.shape[0]
        logger.debug(
            f"Filtered DataFrame for {pdb_id}, {residue_number}, {residue_name}: {num_atoms} atoms found."
        )

        atom_names = residue_df["atom_name"].tolist()
        adjacency_matrix = np.zeros((num_atoms, num_atoms))
        np.fill_diagonal(adjacency_matrix, 1)

        try:
            atom_index = {
                int(row.atom_number): index
                for index, row in enumerate(residue_df.itertuples(index=False))
            }
        except ValueError as e:
            logger.error(
                f"Error: Invalid atom number in {pdb_id}, {residue_number}, {residue_name}: {e}"
            )
            return None

        for index, row in enumerate(residue_df.itertuples(index=False)):
            # Safely access bonds attribute (may not exist or be NaN)
            bonds = getattr(row, 'bonds', None)
            if bonds is not None and pd.notna(bonds):
                try:
                    bonded_atoms = [int(bond.strip()) for bond in str(bonds).split(",")]
                    logger.debug(
                        f"Processing {pdb_id}, residue number {residue_number}, residue {residue_name}, atom {row.atom_number} with bonds: {bonded_atoms}"
                    )
                    for bonded_atom in bonded_atoms:
                        if bonded_atom in atom_index:
                            adj_index = atom_index[bonded_atom]
                            if adj_index < num_atoms:
                                adjacency_matrix[index, adj_index] = 1
                                adjacency_matrix[adj_index, index] = 1
                            else:
                                logger.warning(
                                    f"Bonded atom index {adj_index} is out of bounds for {pdb_id} residue number {residue_number} residue {residue_name}"
                                )
                        else:
                            logger.warning(
                                f"Atom {bonded_atom} not found in {pdb_id} residue number {residue_number} residue {residue_name}"
                            )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error processing bonds for atom {row.atom_number}: {e}")

        adjacency_df = pd.DataFrame(adjacency_matrix, index=atom_names, columns=atom_names)
        # Convert 3-letter code to full name for filename
        full_residue_name = AMINO_ACID_3_TO_FULL.get(residue_name, residue_name)
        filename = os.path.join(
            save_directory,
            f"{pdb_id}_{chain_id}_{residue_number}.{full_residue_name}_adjacency.csv",
        )
        adjacency_df.to_csv(filename, index=True)
        logger.info(f"Adjacency matrix saved as {filename}")

        return adjacency_df

    adjacency_matrices = {}

    for (
        pdb_id,
        chain_id,
        residue_number,
        residue_name,
    ) in df[["PDB", "Chain ID", "Residue_Number_PDB", "Residue_Name"]].drop_duplicates().values:
        residue_number = int(residue_number)  # Ensure integer format for filename
        adjacency_matrices[(pdb_id, chain_id, residue_number, residue_name)] = (
            create_all_atom_adjacency_matrix(df, pdb_id, chain_id, residue_number, residue_name)
        )

    logger.info("=== ADJACENCY MATRIX SUMMARY ===")
    for (pdb_id, chain_id, residue_number, residue_name), matrix in adjacency_matrices.items():
        if matrix is not None:
            logger.info(
                f"PDB ID: {pdb_id}, Chain: {chain_id}, Residue Number: {residue_number}, Residue Name: {residue_name} has an adjacency matrix of shape: {matrix.shape}"
            )


def normalize_and_generate_node_features():
    """Step 15: Normalize dataset and generate node feature vectors for all radii.
    
    Combines normalization, feature engineering, and node feature generation.
    Excludes PDBs with NaN values in Residue_Number_PDB and PDBs from failed file processing.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: LOAD DATA AND NORMALIZE NUMERIC FEATURES")
    print("=" * 80)

    input_path = os.path.join(CONFIG["output_dir"], "merged_features_with_neighbors_hbonds_sasa.csv")
    failed_pdb_log = os.path.join(CONFIG["output_dir"], "Failed_Feature_Generation_PDBs.csv")
    failed_files_log = os.path.join(CONFIG["output_dir"], "failed_files_log.csv")
    
    df = pd.read_csv(input_path)
    print(f"\nLoaded dataset shape: {df.shape}")
    
    # === LOAD FAILED PDB LIST AND EXCLUDE THEM ===
    failed_pdb_list = []
    
    # Load PDBs failed due to NaN values
    if os.path.exists(failed_pdb_log):
        failed_df = pd.read_csv(failed_pdb_log)
        failed_pdb_list.extend(failed_df["PDB"].tolist())
        logger.info(f"Loaded {len(failed_df)} PDBs with NaN values from {failed_pdb_log}")
    
    # Load PDBs failed in file processing step
    if os.path.exists(failed_files_log):
        try:
            df_failed_files = pd.read_csv(failed_files_log)
            # Extract PDB ID from df1_file path
            failed_from_files = [os.path.basename(os.path.dirname(fpath)) for fpath in df_failed_files["df1_file"].tolist()]
            failed_pdb_list.extend(failed_from_files)
            logger.info(f"Loaded {len(failed_from_files)} PDBs from file processing failures from {failed_files_log}")
        except Exception as e:
            logger.warning(f"Could not load failed files log: {e}")
    
    # Remove duplicates
    failed_pdb_list = list(set(failed_pdb_list))
    
    if failed_pdb_list:
        logger.info(f"Total {len(failed_pdb_list)} failed PDBs to exclude: {failed_pdb_list}")
        
        # Filter out failed PDBs
        initial_count = len(df)
        df = df[~df["PDB"].isin(failed_pdb_list)].copy()
        filtered_count = len(df)
        logger.info(f"Filtered out {initial_count - filtered_count} rows from {len(failed_pdb_list)} failed PDB(s)")
        print(f"Excluded {initial_count - filtered_count} rows from {len(failed_pdb_list)} failed PDB(s)")
        print(f"Valid PDBs for feature generation: {df['PDB'].nunique()}")
    else:
        logger.warning(f"No failed PDB logs found")
    
    if len(df) == 0:
        logger.error("No valid data remaining after filtering failed PDBs!")
        return

    features_to_normalize = [
        "Radius_7A_N_Count",
        "Radius_7A_CA_C_Count",
        "Radius_7A_O_Count",
        "Radius_7A_S_Count",
        "Radius_8A_N_Count",
        "Radius_8A_CA_C_Count",
        "Radius_8A_O_Count",
        "Radius_8A_S_Count",
        "Radius_9A_N_Count",
        "Radius_9A_CA_C_Count",
        "Radius_9A_O_Count",
        "Radius_9A_S_Count",
        "Radius_10A_N_Count",
        "Radius_10A_CA_C_Count",
        "Radius_10A_O_Count",
        "Radius_10A_S_Count",
        "Radius_11A_N_Count",
        "Radius_11A_CA_C_Count",
        "Radius_11A_O_Count",
        "Radius_11A_S_Count",
    ]

    missing_features = [col for col in features_to_normalize if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing numeric features (skipping): {missing_features}")

    available_features = [col for col in features_to_normalize if col in df.columns]
    if available_features:
        scaler = MinMaxScaler()
        df[available_features] = scaler.fit_transform(df[available_features])
        logger.info("✓ Normalized neighbor atom count features to [0, 1] range")
    else:
        logger.warning("No numeric features available to normalize")

    normalized_path = os.path.join(CONFIG["output_dir"], "merged_features_normalized.csv")
    df.to_csv(normalized_path, index=False)
    logger.info(f"✓ Saved normalized data to: {normalized_path}")

    print("\n" + "=" * 80)
    print("PHASE 2: SPLIT DIPOLE VECTOR INTO X, Y, Z COMPONENTS")
    print("=" * 80)

    if "Dipole_Vector" in df.columns:
        dipole_parts = df["Dipole_Vector"].astype(str).str.split(",", expand=True)
        if dipole_parts.shape[1] >= 3:
            df[["Dipole_X", "Dipole_Y", "Dipole_Z"]] = dipole_parts.iloc[:, :3]
            df["Dipole_X"] = pd.to_numeric(df["Dipole_X"], errors="coerce")
            df["Dipole_Y"] = pd.to_numeric(df["Dipole_Y"], errors="coerce")
            df["Dipole_Z"] = pd.to_numeric(df["Dipole_Z"], errors="coerce")
        else:
            df[["Dipole_X", "Dipole_Y", "Dipole_Z"]] = np.nan
        df = df.drop(columns=["Dipole_Vector"])
        logger.info("✓ Extracted Dipole_X, Dipole_Y, Dipole_Z from Dipole_Vector")
    else:
        logger.warning("Dipole_Vector column not found, skipping vector split")

    print("\n" + "=" * 80)
    print("PHASE 3: CLASSIFY ATOMS AS BACKBONE (BB) OR SIDECHAIN (SC)")
    print("=" * 80)

    backbone_motif = ["N", "CA", "C", "O"]
    backbone_hydrogens = ["H", "HN", "H1", "H2", "H3", "HA"]

    atom_types = ["SC"] * len(df)
    atom_names = df["atom_name"].astype(str).str.strip().tolist()

    i = 0
    while i < len(atom_names):
        if atom_names[i] == "N":
            matched_indices = []
            j = 0
            k = i

            while k < len(atom_names) and j < len(backbone_motif):
                atom = atom_names[k]
                if atom == backbone_motif[j]:
                    matched_indices.append(k)
                    k += 1
                    j += 1
                elif atom in backbone_hydrogens:
                    matched_indices.append(k)
                    k += 1
                else:
                    break

            if j == len(backbone_motif):
                while k < len(atom_names) and atom_names[k] in backbone_hydrogens:
                    matched_indices.append(k)
                    k += 1

                for idx in matched_indices:
                    atom_types[idx] = "BB"
                i = k
            else:
                i += 1
        else:
            i += 1

    df["atom_type"] = atom_types
    backbone_count = sum(1 for t in atom_types if t == "BB")
    sidechain_count = sum(1 for t in atom_types if t == "SC")
    logger.info(f"✓ Classified atoms: {backbone_count} backbone, {sidechain_count} sidechain")

    print("\n" + "=" * 80)
    print("PHASE 4: CREATE ATOM LABELS FROM ATOM NAME AND TYPE")
    print("=" * 80)

    df["atom_name[0]"] = df["atom_name"].apply(lambda x: x if x == "CA" else x[0])

    mapping_dict = {
        ("N", "BB"): 0,
        ("CA", "BB"): 1,
        ("C", "BB"): 2,
        ("O", "BB"): 3,
        ("H", "BB"): 4,
        ("C", "SC"): 5,
        ("N", "SC"): 6,
        ("H", "SC"): 7,
        ("O", "SC"): 8,
        ("S", "SC"): 9,
    }

    df["atom_label"] = df.apply(
        lambda row: mapping_dict.get((row["atom_name[0]"], row["atom_type"]), -1),
        axis=1,
    )

    logger.info("✓ Created atom labels (0-9 mapping)")
    logger.info(f"Unique atom labels: {sorted(df['atom_label'].unique())}")
    logger.info(f"Unmapped atoms (label -1): {(df['atom_label'] == -1).sum()}")

    print("\n" + "=" * 80)
    print("PHASE 5: ONE-HOT ENCODE RESIDUE NAMES (FULL NAMES)")
    print("=" * 80)

    # Convert 3-letter codes to full residue names for one-hot encoding
    df["Residue Name Full"] = df["Residue Name"].apply(
        lambda x: AMINO_ACID_3_TO_FULL.get(x, x)
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[["Residue Name Full"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Residue Name Full"]))
    
    # Rename columns to use "Residue Name_" prefix with full names
    encoded_df.columns = [col.replace("Residue Name Full_", "Residue Name_") for col in encoded_df.columns]
    
    # Reset indices to ensure proper alignment during concat
    df_reset = df.reset_index(drop=True)
    encoded_df_reset = encoded_df.reset_index(drop=True)
    
    df_combined = pd.concat([df_reset, encoded_df_reset], axis=1)
    # Drop the temporary full name column
    df_combined = df_combined.drop(columns=["Residue Name Full"])

    residue_names = df["Residue Name"].unique()
    logger.info(f"✓ One-hot encoded {len(residue_names)} unique residue types (using full names)")
    logger.info(f"Residue names (3-letter): {sorted(residue_names)}")
    logger.info(f"One-hot encoded columns: {[col for col in encoded_df.columns if col.startswith('Residue Name_')]}")

    output_path = os.path.join(CONFIG["output_dir"], "pKa_Values_Normalized_Final.csv")
    df_combined.to_csv(output_path, index=False)
    logger.info(f"✓ Saved final processed dataset to: {output_path}")

    print("\n" + "=" * 80)
    print("PHASE 6: GENERATE NODE FEATURE VECTORS FOR ALL RADII (7Å - 11Å)")
    print("=" * 80)

    radii = [7, 8, 9, 10, 11]
    base_directory = os.path.join(CONFIG["output_dir"], "Node_Feature_Vectors")
    os.makedirs(base_directory, exist_ok=True)

    def create_node_features_for_radius(df_with_features, radius):
        logger.info(f"Processing radius {radius}Å")
        
        radius_directory = os.path.join(base_directory, str(radius))
        os.makedirs(radius_directory, exist_ok=True)

        # Find all one-hot encoded residue columns (3-letter codes: ASP, GLU, LYS, HIS)
        residue_onehot_cols = [col for col in df_with_features.columns if col.startswith("Residue Name_")]
        
        columns_to_include = residue_onehot_cols + [
            "atom_label",
            "recalculated_x",
            "recalculated_y",
            "recalculated_z",
            "Dipole_X",
            "Dipole_Y",
            "Dipole_Z",
            f"Radius_{radius}A_N_Count",
            f"Radius_{radius}A_CA_C_Count",
            f"Radius_{radius}A_O_Count",
            f"Radius_{radius}A_S_Count",
            "Number of H-Bonds as donor",
            "Number of H-Bonds as acceptor",
            "SASA_Value",
        ]

        available_cols = [col for col in columns_to_include if col in df_with_features.columns]
        missing_cols = [col for col in columns_to_include if col not in df_with_features.columns]
        if missing_cols:
            logger.warning(f"Missing columns for radius {radius}Å: {missing_cols}")

        files_saved = 0
        unique_residues = df_with_features[["PDB", "Chain ID", "Residue_Number_PDB", "Residue Name"]].drop_duplicates()

        for pdb_id, chain_id, residue_number, residue_name in unique_residues.values:
            residue_number = int(residue_number)  # Ensure integer format for filename
            residue_df = df_with_features[
                (df_with_features["PDB"] == pdb_id)
                & (df_with_features["Chain ID"] == chain_id)
                & (df_with_features["Residue_Number_PDB"] == residue_number)
                & (df_with_features["Residue Name"] == residue_name)
            ]

            if len(residue_df) == 0:
                continue

            node_features_df = residue_df[available_cols].copy()
            # Convert 3-letter code to full name for filename
            full_residue_name = AMINO_ACID_3_TO_FULL.get(residue_name, residue_name)
            filename = os.path.join(
                radius_directory, f"{pdb_id}_{chain_id}_{residue_number}.{full_residue_name}.csv"
            )

            try:
                node_features_df.to_csv(filename, index=False)
                files_saved += 1
            except Exception as e:
                logger.error(f"Error saving file {filename}: {e}")

        logger.info(f"✓ Radius {radius}Å: {files_saved} node feature files created")
        logger.info(f"  One-hot encoded residue columns: {residue_onehot_cols}")
        return radius_directory

    for radius in radii:
        try:
            create_node_features_for_radius(df_combined, radius)
        except Exception as e:
            logger.error(f"Error processing radius {radius}Å: {e}")

    logger.info(f"✓ Node feature generation complete. Files saved in: {base_directory}")


if __name__ == "__main__":
    try:
        # === Step 1: Convert XYZ Files ===
        logger.info("Step 1: Processing XYZ files...")
        process_all_xyz_files(CONFIG["input_dir"])

        # === Step 2: Assign Atom Types ===
        logger.info("Step 2: Assigning atom types...")
        process_all_csv_files_to_add_Atom_Types(CONFIG["output_dir"], Atom_types)

        # === Step 3: Extract Dipole Moments ===
        logger.info("Step 3: Extracting dipole moments...")
        process_all_csv_files_add_dipoles(CONFIG["output_dir"])

        # === Step 4: Local Frame Transformation ===
        logger.info("Step 4: Transforming coordinates to local frame...")
        process_all_csv_files_local_frame(CONFIG["output_dir"])

        # === Step 5: Add Descriptions ===
        logger.info("Step 5: Adding atomic descriptions...")
        process_all_csv_files_add_description(DESCRIPTION_DIR)

        # === Step 6 & 7: Copy PDBs and Align Descriptions ===
        logger.info("Step 6 & 7: Copying PDB files and aligning descriptions...")
        copy_pdbs_and_align_descriptions(root_dir, raw_pdb_dir)

        # === Step 8: Map Chain/Residue Info ===
        logger.info("Step 8: Mapping chain and residue information...")
        process_all_files_in_subfolders(root_dir, failed_files_log_path, comparison_results_path)

        # === Step 9: Combine CSV Files ===
        logger.info("Step 9: Combining CSV files...")
        combine_csv_files_in_subfolders(root_dir, output_file_path, columns_to_keep)

        # === Step 10: Generate PDB Files ===
        logger.info("Step 10: Generating PDB files from final CSV data...")
        generate_complete_pdb_from_csv(CONFIG["output_dir"])
        create_pdb_generation_summary(CONFIG["output_dir"])

        # === Calculate Neighbor Atom Counts ===
        logger.info("Step 11: Calculating neighbor heavy atom counts...")
        calculate_neighbor_heavy_atoms(CONFIG["output_dir"])

        # === Calculate H-bonds and SASA ===
        logger.info("Step 12: Calculating H-bonds and SASA per atom...")
        calculate_hbonds_and_sasa_per_atom(CONFIG["output_dir"])

        # === Merge Feature Vectors ===
        logger.info("Step 13: Merging feature vectors from subfolders...")
        merge_feature_vectors_from_subfolders(CONFIG["output_dir"])

        # === Generate Adjacency Matrices ===
        logger.info("Step 14: Generating adjacency matrices...")
        get_adjacency_matrix()

        # === Step 15: Normalize and Generate Node Features ===
        logger.info("Step 15: Normalizing dataset and generating node feature vectors...")
        normalize_and_generate_node_features()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        exit(1)

