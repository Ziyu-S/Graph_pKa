from __future__ import annotations

import warnings
# Suppress pkg_resources deprecation warning BEFORE importing pdbfixer
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import os
import re
import csv
import math
import time
import shutil
import requests
import subprocess
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda

from pathlib import Path
from openmm.app import PDBFile
from pdbfixer import PDBFixer


def fix_pdb_files_with_pdbfixer(
    pdb_dir: str = "../Graph_pKa/Data/0_Raw_PDB",
    output_dir: str = "../Graph_pKa/Data/1_PDB_After_Fixer",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def fix_pdb_file(input_pdb: str, output_pdb: str) -> None:
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with open(output_pdb, "w") as out_file:
            PDBFile.writeFile(fixer.topology, fixer.positions, out_file)

    for file_name in os.listdir(pdb_dir):
        if file_name.endswith(".pdb"):
            input_pdb = os.path.join(pdb_dir, file_name)
            output_pdb = os.path.join(output_dir, file_name.replace(".pdb", "_fixed.pdb"))
            try:
                fix_pdb_file(input_pdb, output_pdb)
            except Exception as e:
                print(f"   Skipping {file_name}: {str(e)[:60]}", flush=True)


def remove_water_with_mdanalysis(
    pdb_dir: str = "../Graph_pKa/Data/1_PDB_After_Fixer",
    output_dir: str = "../Graph_pKa/Data/2_Cleaned_PDB",
    log_file_path: str = "../Graph_pKa/Data/2_Cleaned_PDB/mdanalysis_output.log",
) -> None:
    """
    Remove solvent (water) from PDB files using MDAnalysis.
    Logs one line per file.
    """
    os.makedirs(output_dir, exist_ok=True)

    def log(msg: str) -> None:
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"Input directory not found: {pdb_dir}")

    files = sorted(fn for fn in os.listdir(pdb_dir) if fn.endswith("_fixed.pdb"))
    if not files:
        log(f"No *_fixed.pdb files found in {pdb_dir}")
        return

    log(f"Processing {len(files)} PDB files")

    for file_name in files:
        input_pdb = os.path.join(pdb_dir, file_name)
        output_pdb = os.path.join(
            output_dir, file_name.replace("_fixed.pdb", ".pdb")
        )

        try:
            u = mda.Universe(input_pdb)

            # Remove common water residue names
            atoms = u.select_atoms(
                "not resname HOH WAT SOL TIP3 TIP3P SPC SPCE"
            )

            if atoms.n_atoms == 0:
                log(f"WARNING: {file_name} → 0 atoms after solvent removal")
                continue

            atoms.write(output_pdb)
            log(f"OK: {file_name} → {os.path.basename(output_pdb)}")

        except Exception as e:
            log(f"ERROR: {file_name} → {e}")

    log("Finished solvent removal")


def convert_pdb_to_xyz_files(
    pdb_dir: str = "../Graph_pKa/Data/2_Cleaned_PDB",
    param_file: str = "../Graph_pKa/Tinker_params/amoebabio18.prm",
    error_log_name: str = "conversion_errors.log",
) -> None:
    error_log_file = os.path.join(pdb_dir, error_log_name)
    with open(error_log_file, "w") as log:
        log.write("PDB to XYZ Conversion Log\n")
        log.write("=" * 60 + "\n")

    pdb_files = [fn for fn in os.listdir(pdb_dir) if fn.endswith(".pdb")]
    successful_count = 0
    failed_count = 0

    with open(error_log_file, "a") as log:
        log.write(f"Processing {len(pdb_files)} PDB files\n\n")

    for filename in pdb_files:
        pdb_file = os.path.join(pdb_dir, filename)
        base_name = os.path.splitext(filename)[0]
        try:
            command = "pdbxyz.x"
            input_str = f"{pdb_file}\nALL\n{param_file}\n"
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=input_str)
            if process.returncode != 0:
                with open(error_log_file, "a") as log:
                    log.write(f"ERROR: {filename} → Failed to convert\n{stderr}\n\n")
                failed_count += 1
            else:
                with open(error_log_file, "a") as log:
                    log.write(f"OK: {filename} → {base_name}.xyz\n")
                successful_count += 1
            _ = stdout
        except Exception as e:
            with open(error_log_file, "a") as log:
                log.write(f"ERROR: {filename} → Exception: {str(e)}\n")
            failed_count += 1
            print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)

    with open(error_log_file, "a") as log:
        log.write("\n" + "=" * 60 + "\n")
        log.write(f"Summary: {successful_count} succeeded, {failed_count} failed\n")
        log.write("Finished PDB to XYZ conversion\n")


def move_center_of_mass_and_relocate(
    cleaned_pdb_dir: str = "../Graph_pKa/Data/2_Cleaned_PDB",
    xyz_dir: str = "../Graph_pKa/Data/3_PDB_XYZ",
    center_moved_xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
    param_file: str = "../../Tinker_params/amoebabio18.prm",
) -> None:
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(center_moved_xyz_dir, exist_ok=True)

    for filename in os.listdir(cleaned_pdb_dir):
        if filename.endswith(".xyz"):
            source_path = os.path.join(cleaned_pdb_dir, filename)
            destination_path = os.path.join(xyz_dir, filename)
            shutil.move(source_path, destination_path)

    error_log_file = os.path.join(xyz_dir, "translate_errors.log")
    with open(error_log_file, "w") as log:
        log.write("Translation Errors:\n")

    successful_count = 0
    failed_count = 0
    
    for filename in os.listdir(xyz_dir):
        if filename.endswith(".xyz"):
            try:
                command = "xyzedit.x"
                input_str = f"{filename}\n{param_file}\n16\n\n"
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=xyz_dir,
                )
                stdout, stderr = process.communicate(input=input_str)
                if process.returncode != 0:
                    with open(error_log_file, "a") as log:
                        log.write(f"Error moving the center of mass for {filename}:\n{stderr}\n\n")
                    failed_count += 1
                else:
                    with open(error_log_file, "a") as log:
                        log.write(f"OK: {filename} processed\n")
                    successful_count += 1
                _ = stdout
            except Exception as e:
                with open(error_log_file, "a") as log:
                    log.write(f"Error processing {filename}: {str(e)}\n")
                failed_count += 1
                print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)

    with open(error_log_file, "a") as log:
        log.write(f"\nSummary: {successful_count} succeeded, {failed_count} failed\n")
    
    # Move the output files
    moved_count = 0
    xyz_2_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz_2")]
    
    with open(error_log_file, "a") as log:
        log.write(f"\nLooking for .xyz_2 files: found {len(xyz_2_files)}\n")
    
    for filename in xyz_2_files:
        source_path = os.path.join(xyz_dir, filename)
        new_filename = filename.replace(".xyz_2", ".xyz")
        destination_path = os.path.join(center_moved_xyz_dir, new_filename)
        shutil.move(source_path, destination_path)
        moved_count += 1
    
    with open(error_log_file, "a") as log:
        log.write(f"Moved {moved_count} files to center_moved_xyz_dir\n")


def write_xyz_coordinate_ranges(
    xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
    output_csv: str = "../Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv",
) -> None:
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "PDB_ID",
                "Min_X",
                "Min_Y",
                "Min_Z",
                "Max_X",
                "Max_Y",
                "Max_Z",
                "Range_X",
                "Range_Y",
                "Range_Z",
                "Cube_Size",
                "Waterbox"
            ]
        )

        for filename in os.listdir(xyz_dir):
            if filename.endswith(".xyz"):
                try:
                    xyz_file = os.path.join(xyz_dir, filename)
                    coordinates = []
                    with open(xyz_file, "r") as xyz_handle:
                        lines = xyz_handle.readlines()
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) >= 5:
                                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                                coordinates.append([x, y, z])
                    all_coords = np.array(coordinates)
                    min_coords = all_coords.min(axis=0)
                    max_coords = all_coords.max(axis=0)
                    range_coords = max_coords - min_coords
                    cube_size = math.ceil(max(range_coords)) + 20
                    
                    # Compute waterbox based on cube_size
                    if cube_size <= 120:
                        if cube_size < 30:
                            waterbox = 30
                        else:
                            # Round up to nearest multiple of 5
                            waterbox = ((cube_size + 4) // 5) * 5
                            waterbox = min(waterbox, 120)
                    else:
                        # cube_size > 120, pick smallest waterbox >= cube_size from 127, 145, 161, 196
                        options = [127, 145, 161, 196]
                        valid_options = [x for x in options if x >= cube_size]
                        waterbox = min(valid_options) if valid_options else options[-1]
                    
                    pdb_name = os.path.splitext(filename)[0].strip()
                    pdb_name = f"'{pdb_name}"
                    writer.writerow(
                        [
                            pdb_name,
                            *min_coords,
                            *max_coords,
                            *range_coords,
                            cube_size,
                            waterbox
                            
                        ]
                    )
                except Exception as e:
                    print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)


def soak_proteins_with_waterbox(
    tinker_coords_csv: str = "../Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv",
    center_moved_xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
    soaked_proteins_dir: str = "../Graph_pKa/Data/5_Dissolved_Proteins",
    param_file: str = "../Graph_pKa/Tinker_params/amoebabio18.prm",
) -> None:
    try:
        df = pd.read_csv(tinker_coords_csv)
        df["PDB_ID"] = df["PDB_ID"].str.replace("'", "")
        water_map = df.set_index("PDB_ID")["Waterbox"].to_dict()
    except Exception as e:
        print(f"   Skipping soak_proteins_with_waterbox: {str(e)[:60]}", flush=True)
        return

    os.makedirs(soaked_proteins_dir, exist_ok=True)
    error_log_file = os.path.join(soaked_proteins_dir, "soak_errors.log")
    with open(error_log_file, "w") as log:
        log.write("Soaking Errors:\n")

    for filename in os.listdir(center_moved_xyz_dir):
        if filename.endswith(".xyz"):
            try:
                pdb_id = os.path.splitext(filename)[0]
                if pdb_id not in water_map:
                    continue
                waterbox_value = water_map[pdb_id]
                waterbox_file = f"../Graph_pKa/Tinker_params/waterbox/water{waterbox_value}.xyz"
                xyz_file = os.path.join(center_moved_xyz_dir, filename)
                output_file = os.path.join(center_moved_xyz_dir, f"{pdb_id}_{waterbox_value}.xyz")

                command = "xyzedit.x"
                input_str = f"{xyz_file}\n{param_file}\n27\n{waterbox_file}\n\n"
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate(input=input_str)
                if process.returncode != 0:
                    with open(error_log_file, "a") as log:
                        log.write(f"Error soaking {filename}:\n{stderr}\n\n")
                _ = stdout, output_file
            except Exception as e:
                print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)

    for filename in os.listdir(center_moved_xyz_dir):
        if filename.endswith(".xyz_2"):
            try:
                source_path = os.path.join(center_moved_xyz_dir, filename)
                new_filename = filename.replace(".xyz_2", ".xyz")
                destination_path = os.path.join(soaked_proteins_dir, new_filename)
                shutil.move(source_path, destination_path)
            except Exception as e:
                print(f"   Skipping move {filename}: {str(e)[:60]}", flush=True)


def analyze_and_collect_charge(
    xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
    param_file: str = "../Graph_pKa/Tinker_params/amoebabio18.prm",
) -> None:
    error_log_file = os.path.join(xyz_dir, "analyze_errors.log")
    with open(error_log_file, "w") as log:
        log.write("Analysis Errors:\n")

    charge_info_file = os.path.join(xyz_dir, "charge_info.csv")
    with open(charge_info_file, "w") as log:
        log.write("PDB_ID,Total_Electric_Charge\n")

    for filename in os.listdir(xyz_dir):
        if filename.endswith(".xyz"):
            try:
                pdb_id = os.path.splitext(filename)[0]
                xyz_file = os.path.join(xyz_dir, filename)

                analyze_command = "analyze.x"
                analyze_input = f"{xyz_file}\n{param_file}\nm\n"
                process = subprocess.Popen(
                    analyze_command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate(input=analyze_input)
                if process.returncode == 0:
                    match = re.search(r"Total Electric Charge :\s+([-+]?[0-9]*\.?[0-9]+)", stdout)
                    if match:
                        total_charge = match.group(1)
                        with open(charge_info_file, "a") as log:
                            log.write(f"{pdb_id},{total_charge}\n")
                    else:
                        with open(error_log_file, "a") as log:
                            log.write(f"Failed to extract total electric charge for {filename}\n")
                else:
                    with open(error_log_file, "a") as log:
                        log.write(f"Error processing {filename}:\n{stderr}\n\n")
            except Exception as e:
                print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)


def generate_system_solvent_info(
    soaked_proteins_dir: str = "../Graph_pKa/Data/5_Dissolved_Proteins",
    solvent_info_csv: str = "../Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv",
    source_xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
) -> None:
    """Generate System_Solve_Info.csv by finding OHH-OHH patterns in soaked proteins."""
    def find_ohh_ohh_pattern(filepath: str) -> tuple[int | None, int | None]:
        """Find OHH-OHH pattern in soaked protein file"""
        with open(filepath, "r") as file:
            lines = file.readlines()

        first_pattern_id = None
        last_h_id = None

        # Find OHH-OHH pattern
        for i in range(len(lines) - 5):
            current_line = lines[i].split()
            next_line1 = lines[i + 1].split()
            next_line2 = lines[i + 2].split()
            next_line3 = lines[i + 3].split()
            next_line4 = lines[i + 4].split()
            next_line5 = lines[i + 5].split()

            if (
                len(current_line) > 1
                and len(next_line1) > 1
                and len(next_line2) > 1
                and len(next_line3) > 1
                and len(next_line4) > 1
                and len(next_line5) > 1
            ):
                if (
                    current_line[1] == "O"
                    and next_line1[1] == "H"
                    and next_line2[1] == "H"
                    and next_line3[1] == "O"
                    and next_line4[1] == "H"
                    and next_line5[1] == "H"
                ):
                    if first_pattern_id is None:
                        first_pattern_id = int(current_line[0])
                    last_h_id = int(next_line5[0])

        return first_pattern_id, last_h_id

    def get_last_atom_id_from_source(filepath: str) -> int | None:
        """Get last atom ID from source XYZ file (4_Center_Moved_XYZ)"""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, "r") as file:
                lines = file.readlines()
            
            # Scan backwards from the end to find last atom ID
            for i in range(len(lines) - 1, 0, -1):
                parts = lines[i].split()
                if len(parts) > 0 and parts[0].isdigit():
                    try:
                        return int(parts[0])
                    except ValueError:
                        continue
        except Exception:
            pass
        
        return None

    os.makedirs(os.path.dirname(solvent_info_csv), exist_ok=True)
    results = []
    for filename in os.listdir(soaked_proteins_dir):
        if filename.endswith(".xyz"):
            try:
                soaked_filepath = os.path.join(soaked_proteins_dir, filename)
                first_pattern_id, last_h_id = find_ohh_ohh_pattern(soaked_filepath)
                
                # Get last atom ID from corresponding source XYZ file
                source_xyz_path = os.path.join(source_xyz_dir, filename)
                last_atom_id = get_last_atom_id_from_source(source_xyz_path)
                
                # Check if last atom ID is consecutive to first OHH-OHH row
                is_consecutive = None
                if last_atom_id is not None and first_pattern_id is not None:
                    is_consecutive = (last_atom_id + 1) == first_pattern_id
                
                results.append((filename, first_pattern_id, last_h_id, last_atom_id, is_consecutive))
            except Exception as e:
                print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)

    with open(solvent_info_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "Filename",
                "First O-H-H-O-H-H Row",
                "Last H in Last O-H-H-O-H-H Row",
                "Last_Atom_ID_In_Source_XYZ",
                "Is_Consecutive_To_First_OHH_OHH",
            ]
        )
        csv_writer.writerows(results)


def process_solvent_info_and_ohh_ohh(
    xyz_dir: str = "../Graph_pKa/Data/4_Center_Moved_XYZ",
    solvent_info_csv: str = "../Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv",
    merged_output_csv: str = "../Graph_pKa/Data/6_Neutralized_System/System_Solve_Info_with_Charge.csv",
    merge_mode: str = "extracted_pdb_id",
) -> None:
    """Merge charge info with solvent info."""
    # Create output directories
    os.makedirs(os.path.dirname(merged_output_csv), exist_ok=True)
    
    charge_info_csv = os.path.join(xyz_dir, "charge_info.csv")
    
    # Only merge if both files exist
    if os.path.exists(charge_info_csv) and os.path.exists(solvent_info_csv):
        try:
            charge_info_df = pd.read_csv(charge_info_csv)
            solvent_info_df = pd.read_csv(solvent_info_csv)

            if merge_mode == "concat":
                merged_df = pd.concat([charge_info_df, solvent_info_df], axis=1)
            elif merge_mode == "pdb_id":
                merged_df = pd.merge(charge_info_df, solvent_info_df, on="PDB_ID", how="inner")
            elif merge_mode == "extracted_pdb_id":
                def extract_pdb_id_from_filename(filename: str) -> str:
                    base_name = filename.replace(".xyz", "")
                    return base_name.split("_")[0]

                solvent_info_df["PDB_ID_extracted"] = solvent_info_df["Filename"].apply(
                    extract_pdb_id_from_filename
                )

                merged_df = pd.merge(
                    charge_info_df,
                    solvent_info_df,
                    left_on="PDB_ID",
                    right_on="PDB_ID_extracted",
                    how="inner",
                )
                merged_df = merged_df.drop("PDB_ID_extracted", axis=1)
            else:
                raise ValueError("merge_mode must be 'concat', 'pdb_id', or 'extracted_pdb_id'")

            merged_df.to_csv(merged_output_csv, index=False)
        except Exception as e:
            print(f"   Skipping merge: {str(e)[:60]}", flush=True)


def offset_charges_in_systems(
    merged_system_solve_file: str = "../Graph_pKa/Data/6_Neutralized_System/System_Solve_Info_with_Charge.csv",
    soaked_proteins_dir: str = "../Graph_pKa/Data/5_Dissolved_Proteins",
    neutralized_system_dir: str = "../Graph_pKa/Data/6_Neutralized_System",
    log_file_path: str = "../Graph_pKa/Data/6_Neutralized_System/Failed/Neutralization_log.txt",
) -> list[tuple[str, str]]:
    # Create output directories
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    os.makedirs(neutralized_system_dir, exist_ok=True)
    
    data = {}
    try:
        with open(merged_system_solve_file, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                filename = row["Filename"]
                try:
                    total_electric_charge = float(row["Total_Electric_Charge"])
                except ValueError:
                    total_electric_charge = 0
                data[filename] = {
                    "First_OHHOHH_Row": row["First O-H-H-O-H-H Row"],
                    "Last_H_in_Last_OHHOHH_Row": row["Last H in Last O-H-H-O-H-H Row"],
                    "Total_Electric_Charge": total_electric_charge,
                }
    except Exception as e:
        print(f"   Skipping offset_charges_in_systems: {str(e)[:60]}", flush=True)
        return []

    with open(log_file_path, "w") as log_file:
        log_file.write("")

    results = []
    for filename in os.listdir(soaked_proteins_dir):
        if filename.endswith(".xyz"):
            try:
                filepath = os.path.join(soaked_proteins_dir, filename)
                if filename in data:
                    file_data = data[filename]
                    ion_type = 363 if file_data["Total_Electric_Charge"] > 0 else 352
                    ion_count = abs(int(file_data["Total_Electric_Charge"]))
                    command = "xyzedit.x"
                    input_str = (
                        f"{filepath}\n"
                        "../Graph_pKa/Tinker_params/amoebabio18.prm\n"
                        "28\n"
                        f"{file_data['First_OHHOHH_Row']}, {file_data['Last_H_in_Last_OHHOHH_Row']}\n"
                        f"{ion_type}, {ion_count}\n"
                    )
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Command: {command}\nInput:\n{input_str}\n")

                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=True,
                    )
                    stdout, stderr = process.communicate(input=input_str)
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Standard Output: {stdout.strip()}\n")
                        log_file.write(f"Standard Error: {stderr.strip()}\n")
                    if process.returncode == 0:
                        results.append((filename, "Processed Successfully"))
                    else:
                        results.append((filename, f"Error: {stderr}"))
            except Exception as e:
                print(f"   Skipping {filename}: {str(e)[:60]}", flush=True)

    for filename in os.listdir(soaked_proteins_dir):
        if filename.endswith(".xyz_2"):
            try:
                source_path = os.path.join(soaked_proteins_dir, filename)
                new_filename = filename.replace(".xyz_2", ".xyz")
                destination_path = os.path.join(neutralized_system_dir, new_filename)
                shutil.move(source_path, destination_path)
            except Exception as e:
                print(f"   Skipping move {filename}: {str(e)[:60]}", flush=True)

    return results

# Iterative Neutralization for Failed Neturalization Systems
def filter_copy_and_iterative_neutralization(
    xyz_dir: str = "../Graph_pKa/Data/6_Neutralized_System",
    output_csv: str = "../Graph_pKa/Data/6_Neutralized_System/Failed/Failed_Neutralizations.csv",
    source_dir: str = "../Graph_pKa/Data/5_Dissolved_Proteins",
    work_dir: str = "../Graph_pKa/Data/6_Neutralized_System/Failed",
    redo_dir: str = "../Graph_pKa/Data/6_Neutralized_System/Redo",
    merged_csv_file: str = "../Graph_pKa/Data/6_Neutralized_System/System_Solve_Info_with_Charge.csv",
    log_file_path: str = "../Graph_pKa/Data/6_Neutralized_System/Failed/Iterative_Neutralization_log.txt",
    max_iterations: int = 10, ## The maxium neutralization iterations
) -> list[str]:
    # ===== HELPER FUNCTIONS =====
    def find_ohh_ohh_pattern(filepath: str) -> tuple[int | None, int | None]:
        """Find OHH-OHH pattern in XYZ file and return (first_pattern_line_idx, last_h_id)"""
        if not os.path.exists(filepath):
            return None, None
        
        try:
            with open(filepath, "r") as file:
                lines = file.readlines()

            first_pattern_idx = None
            last_h_id = None

            # Find OHH-OHH pattern
            for i in range(len(lines) - 5):
                current_line = lines[i].split()
                next_line1 = lines[i + 1].split()
                next_line2 = lines[i + 2].split()
                next_line3 = lines[i + 3].split()
                next_line4 = lines[i + 4].split()
                next_line5 = lines[i + 5].split()

                if (
                    len(current_line) > 1
                    and len(next_line1) > 1
                    and len(next_line2) > 1
                    and len(next_line3) > 1
                    and len(next_line4) > 1
                    and len(next_line5) > 1
                ):
                    if (
                        current_line[1] == "O"
                        and next_line1[1] == "H"
                        and next_line2[1] == "H"
                        and next_line3[1] == "O"
                        and next_line4[1] == "H"
                        and next_line5[1] == "H"
                    ):
                        if first_pattern_idx is None:
                            first_pattern_idx = i  # Store line index (0-based)
                        last_h_id = int(next_line5[0])

            return first_pattern_idx, last_h_id
        except Exception:
            return None, None

    def get_last_atom_id_before_pattern(filepath: str) -> int | None:
        """Get last atom ID BEFORE the first OHH-OHH pattern"""
        if not os.path.exists(filepath):
            return None
        
        try:
            first_pattern_idx, _ = find_ohh_ohh_pattern(filepath)
            
            if first_pattern_idx is None or first_pattern_idx <= 0:
                return None
            
            with open(filepath, "r") as file:
                lines = file.readlines()
            
            # Get line before the first OHH-OHH pattern
            prev_line_idx = first_pattern_idx - 1
            if prev_line_idx >= 0:
                parts = lines[prev_line_idx].split()
                if len(parts) > 0 and parts[0].isdigit():
                    try:
                        return int(parts[0])
                    except ValueError:
                        pass
        except Exception:
            pass
        
        return None

    # ===== STEP 1: LOAD DATA FROM CSV =====
    files_with_issues = {}  # {filename: [reasons]}
    data = {}
    
    with open(merged_csv_file, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["Filename"]
            try:
                total_electric_charge = float(row["Total_Electric_Charge"])
            except ValueError:
                total_electric_charge = 0
            
            # Get last atom ID BEFORE the first OHH-OHH pattern from dissolved system file
            dissolved_filepath = os.path.join(source_dir, filename)
            last_atom_id_dissolved = get_last_atom_id_before_pattern(dissolved_filepath)
            
            # Get last atom ID BEFORE the first OHH-OHH pattern from neutralized system file (xyz_dir)
            neutralized_filepath = os.path.join(xyz_dir, filename)
            last_atom_id_neutralized = get_last_atom_id_before_pattern(neutralized_filepath)
            
            data[filename] = {
                "First_OHHOHH_Row": row["First O-H-H-O-H-H Row"],
                "Last_H_in_Last_OHHOHH_Row": row["Last H in Last O-H-H-O-H-H Row"],
                "Total_Electric_Charge": total_electric_charge,
                "Last_Atom_ID_Dissolved": last_atom_id_dissolved,
                "Last_Atom_ID_Neutralized": last_atom_id_neutralized,
                "Atom_ID_Match": None,
            }

    # ===== STEP 2: WRITE DIAGNOSTIC CSV WITH ATOM ID ANALYSIS =====
    diagnostic_csv = os.path.join(os.path.dirname(output_csv), "Atom_ID_Analysis.csv")
    os.makedirs(os.path.dirname(diagnostic_csv), exist_ok=True)
    with open(diagnostic_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Last_Atom_ID_Dissolved", "Last_Atom_ID_Neutralized", "Difference", "Status"])
        for filename in data:
            last_dissolved = data[filename]["Last_Atom_ID_Dissolved"]
            last_neutralized = data[filename]["Last_Atom_ID_Neutralized"]
            
            if last_dissolved is not None and last_neutralized is not None:
                difference = last_neutralized - last_dissolved
                status = "OK" if last_dissolved == last_neutralized else "MISMATCH"
            else:
                difference = "N/A"
                status = "Missing Data"
            
            writer.writerow([filename, last_dissolved, last_neutralized, difference, status])

    # ===== STEP 3: IDENTIFY FILES WITH ATOM ID MISMATCH =====
    for filename in data:
        last_atom_id_dissolved = data[filename]["Last_Atom_ID_Dissolved"]
        last_atom_id_neutralized = data[filename]["Last_Atom_ID_Neutralized"]
        
        # Check if both systems have the same last atom ID BEFORE the first OHH-OHH pattern
        # They should match if neutralization worked correctly
        if last_atom_id_dissolved is not None and last_atom_id_neutralized is not None:
            atom_id_match = last_atom_id_dissolved == last_atom_id_neutralized
            data[filename]["Atom_ID_Match"] = atom_id_match
            
            # Flag only if they don't match
            if not atom_id_match:
                if filename not in files_with_issues:
                    files_with_issues[filename] = []
                files_with_issues[filename].append("Atom_ID_Mismatch")
        elif last_atom_id_dissolved is None or last_atom_id_neutralized is None:
            # Flag if we couldn't read the atom IDs
            if filename not in files_with_issues:
                files_with_issues[filename] = []
            files_with_issues[filename].append("Unable_to_Read_Atom_IDs")

    # ===== STEP 4: WRITE OUTPUT CSV =====
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Issue_Reason"])
        for filename in files_with_issues:
            reasons = ", ".join(files_with_issues[filename])
            writer.writerow([filename, reasons])

    # ===== STEP 5: COPY FLAGGED FILES TO WORK DIRECTORY =====
    os.makedirs(work_dir, exist_ok=True)
    for filename in files_with_issues:
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(work_dir, filename)
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        
        # Delete the flagged file from xyz_dir
        xyz_file = os.path.join(xyz_dir, filename)
        if os.path.exists(xyz_file):
            try:
                os.remove(xyz_file)
            except Exception:
                pass

    # ===== STEP 6: SETUP DIRECTORIES AND LOGGING =====
    os.makedirs(redo_dir, exist_ok=True)
    with open(log_file_path, "w") as file:
        file.write(f"Iterative Neutralization Log - Started at {time.ctime()}\n")
        file.write("=" * 60 + "\n")

    # ===== STEP 7: RUN ITERATIVE NEUTRALIZATION =====
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        initial_files = [f for f in os.listdir(work_dir) if f.endswith(".xyz")]
        if not initial_files:
            with open(log_file_path, "a") as log_file:
                log_file.write("SUCCESS: All files have been successfully neutralized!\n")
            break

        for filename in list(initial_files):
            filepath = os.path.join(work_dir, filename)
            if filename in data:
                file_data = data[filename]
                
                # Log atom ID verification
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\nFile: {filename}\n")
                    log_file.write(f"  Last Atom ID from Dissolved System: {file_data['Last_Atom_ID_Dissolved']}\n")
                    log_file.write(f"  Last Atom ID from Neutralized System: {file_data['Last_Atom_ID_Neutralized']}\n")
                    log_file.write(f"  Atom ID Match: {file_data['Atom_ID_Match']}\n")
                
                ion_type = 363 if file_data["Total_Electric_Charge"] > 0 else 352
                ion_count = abs(int(file_data["Total_Electric_Charge"]))
                command = "xyzedit.x"
                input_str = (
                    f"{filepath}\n"
                    "../Graph_pKa/Tinker_params/amoebabio18.prm\n"
                    "26\n"
                    f"{file_data['First_OHHOHH_Row']}, {file_data['Last_H_in_Last_OHHOHH_Row']}\n"
                    f"{ion_type}, {ion_count}\n"
                )
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Neutralizing {os.path.basename(filepath)}: {input_str.strip()}\n")
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                )
                stdout, stderr = process.communicate(input=input_str)
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Output: {stdout.strip()}\n")
                    log_file.write(f"Error: {stderr.strip()}\n")
            else:
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"No neutralization data found for {filename}\n")

        # Check all .xyz files in work_dir to see if they've been successfully neutralized
        successful_files = []
        current_xyz_files = [f for f in os.listdir(work_dir) if f.endswith(".xyz")]
        
        for filename in current_xyz_files:
            filepath = os.path.join(work_dir, filename)
            
            if filename not in data:
                continue
            
            # Check if this file's atom IDs now match (indicating successful neutralization)
            last_atom_id_neutralized = get_last_atom_id_before_pattern(filepath)
            last_atom_id_dissolved = data[filename]["Last_Atom_ID_Dissolved"]
            
            # Success if atom IDs now match
            is_success = (last_atom_id_dissolved is not None and 
                         last_atom_id_neutralized is not None and
                         last_atom_id_dissolved == last_atom_id_neutralized)
            
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\nChecking {filename}: ")
                log_file.write(f"Dissolved={last_atom_id_dissolved}, Neutralized={last_atom_id_neutralized}, ")
                log_file.write(f"Match={is_success}\n")
            
            if is_success:
                # Move the successful .xyz file to xyz_dir
                final_path = os.path.join(xyz_dir, filename)
                if os.path.exists(final_path):
                    os.remove(final_path)
                shutil.move(filepath, final_path)
                successful_files.append(filename)
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"SUCCESS: Moved {filename} to {final_path}\n")
                
                # Update data to reflect successful neutralization
                data[filename]["Last_Atom_ID_Neutralized"] = last_atom_id_neutralized
                data[filename]["Atom_ID_Match"] = True
            else:
                # File still not successful, will retry in next iteration
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"PENDING: {filename} still needs neutralization\n")

        # Clean up any leftover .xyz_2 files
        for xyz_2_file in os.listdir(work_dir):
            if xyz_2_file.endswith(".xyz_2"):
                try:
                    os.remove(os.path.join(work_dir, xyz_2_file))
                except Exception:
                    pass

        remaining_files = [f for f in os.listdir(work_dir) if f.endswith(".xyz")]
        if not remaining_files:
            with open(log_file_path, "a") as log_file:
                log_file.write("SUCCESS: All files successfully neutralized!\n")
            break

        if len(successful_files) == 0:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"WARNING: No files succeeded in iteration {iteration}\n")
                log_file.write(f"Remaining files: {remaining_files}\n")

        time.sleep(2)

    if iteration >= max_iterations:
        final_failed = [f for f in os.listdir(work_dir) if f.endswith(".xyz")]
        if final_failed:
            with open(log_file_path, "a") as log_file:
                log_file.write(
                    f"Files still failing after {max_iterations} iterations: {final_failed}\n"
                )

    return list(files_with_issues.keys())

def create_key_files_and_run_minimization(
    csv_file: str = "../Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv",
    directory: str = "../Graph_pKa/Data/6_Neutralized_System",
    energy_minimization_dir: str = "../Graph_pKa/Data/7_Energy_Minimization_Systems",
    param_file: str = "../Graph_pKa/Tinker_params/amoebabio18.prm",
) -> None:
    waterbox_data = {}
    try:
        with open(csv_file, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pdb_id = row["PDB_ID"].strip().lstrip("'")
                waterbox = row["Waterbox"].strip()
                waterbox_data[pdb_id] = waterbox
    except Exception as e:
        print(f"   Skipping create_key_files_and_run_minimization: {str(e)[:60]}", flush=True)
        return

    os.makedirs(energy_minimization_dir, exist_ok=True)

    for file in os.listdir(directory):
        if file.endswith(".xyz"):
            try:
                shutil.copy(os.path.join(directory, file), energy_minimization_dir)
            except Exception as e:
                print(f"   Skipping copy {file}: {str(e)[:60]}", flush=True)

    results = {}
    for file in os.listdir(energy_minimization_dir):
        if file.endswith(".xyz"):
            try:
                pdb_id = os.path.splitext(file)[0]
                subfolder_path = os.path.join(energy_minimization_dir, pdb_id)
                os.makedirs(subfolder_path, exist_ok=True)

                shutil.move(os.path.join(energy_minimization_dir, file), os.path.join(subfolder_path, file))

                xyz_path = os.path.join(subfolder_path, file)
                key_path = os.path.join(subfolder_path, file.replace(".xyz", ".key"))

                if pdb_id in waterbox_data:
                    content = (
                        f"parameters              {param_file}\n"
                        "verbose\n\n"
                        f"a-axis                      {waterbox_data[pdb_id]}\n"
                        f"b-axis                      {waterbox_data[pdb_id]}\n"
                        f"c-axis                      {waterbox_data[pdb_id]}\n"
                        "neighbor-list\n"
                        "vdw-cutoff                     12.0\n"
                        "ewald\n"
                        "ewald-cutoff                    7.0\n"
                        "integrator                    RESPA\n\n"
                        "polar-eps                   0.00001\n"
                        "polar-predict\n"
                        "save-induced\n"
                    )
                    with open(key_path, "w") as key_file:
                        key_file.write(content)

                command = ["minimize.x", xyz_path, "-k", key_path]
                start_time = time.time()
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate(input="1\n")
                elapsed_time = time.time() - start_time

                if process.returncode == 0:
                    for line in stdout.splitlines():
                        if "Final Function Value" in line:
                            final_value = line.split(":")[-1].strip()
                            results[file] = {
                                "Final Function Value": final_value,
                                "Time Taken (s)": elapsed_time,
                            }
                _ = stderr
            except Exception as e:
                print(f"   Skipping {file}: {str(e)[:60]}", flush=True)

    csv_file_path = os.path.join(energy_minimization_dir, "final_function_values.csv")
    try:
        with open(csv_file_path, mode="w", newline="") as csvfile:
            fieldnames = ["PDB_ID", "Final Function Value", "Time Taken (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file, data in results.items():
                writer.writerow(
                    {
                        "PDB_ID": file,
                        "Final Function Value": data["Final Function Value"],
                        "Time Taken (s)": data["Time Taken (s)"],
                    }
                )
    except Exception as e:
        print(f"   Skipping CSV write: {str(e)[:60]}", flush=True)


FUNCTION_MAP = {
    "fix_pdb_files_with_pdbfixer": fix_pdb_files_with_pdbfixer,
    "remove_water_with_mdanalysis": remove_water_with_mdanalysis,
    "convert_pdb_to_xyz_files": convert_pdb_to_xyz_files,
    "move_center_of_mass_and_relocate": move_center_of_mass_and_relocate,
    "write_xyz_coordinate_ranges": write_xyz_coordinate_ranges,
    "soak_proteins_with_waterbox": soak_proteins_with_waterbox,
    "analyze_and_collect_charge": analyze_and_collect_charge,
    "generate_system_solvent_info": generate_system_solvent_info,
    "process_solvent_info_and_ohh_ohh": process_solvent_info_and_ohh_ohh,
    "offset_charges_in_systems": offset_charges_in_systems,
    "filter_copy_and_iterative_neutralization": filter_copy_and_iterative_neutralization,
    "create_key_files_and_run_minimization": create_key_files_and_run_minimization,
}

__all__ = list(FUNCTION_MAP.keys())


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("DEBUG: Script started", flush=True)
    print(f"DEBUG: Current working directory: {os.getcwd()}", flush=True)
    print(f"DEBUG: Script location: {os.path.abspath(__file__)}", flush=True)
    print("=" * 60, flush=True)
    
    print("\nStarting Tinker Energy Minimization Pipeline...", flush=True)
    
    try:
        print("\n1. Fixing PDB files with PDBFixer...", flush=True)
        fix_pdb_files_with_pdbfixer()
        print("    PDB files fixed", flush=True)
        
        print("\n2. Removing water with MDAnalysis...", flush=True)
        remove_water_with_mdanalysis()
        print("    Water removed", flush=True)
        
        print("\n3. Converting PDB to XYZ files...", flush=True)
        convert_pdb_to_xyz_files()
        print("    PDB converted to XYZ", flush=True)
        
        print("\n4. Moving center of mass and relocating...", flush=True)
        move_center_of_mass_and_relocate()
        print("    Center of mass moved", flush=True)
        
        print("\n5. Writing XYZ coordinate ranges...", flush=True)
        write_xyz_coordinate_ranges()
        print("    Coordinate ranges written", flush=True)
        
        print("\n6. Soaking proteins with waterbox...", flush=True)
        soak_proteins_with_waterbox()
        print("    Proteins soaked", flush=True)
        
        print("\n7. Analyzing and collecting charges...", flush=True)
        analyze_and_collect_charge()
        print("    Charges collected", flush=True)
        
        print("\n8. Generating system solvent info...", flush=True)
        generate_system_solvent_info()
        print("    System solvent info generated", flush=True)
        
        print("\n9. Processing solvent info and OHH-OHH patterns...", flush=True)
        process_solvent_info_and_ohh_ohh()
        print("    Solvent info processed", flush=True)
        
        print("\n10. Offsetting charges in systems...", flush=True)
        offset_charges_in_systems()
        print("    Charges offset", flush=True)

        print("\nOptional. Iteratively offsetting charges in systems...", flush=True)
        filter_copy_and_iterative_neutralization()
        print("    Neutralizing Failed Systems", flush=True)
        
        print("\n11. Creating key files and running minimization...", flush=True)
        create_key_files_and_run_minimization()
        print("    Minimization complete", flush=True)
        
        print("HTP Tinker_EM completed successfully!", flush=True)
        
    except Exception as e:
        print(f" Error during pipeline execution: {e}", flush=True)
        import traceback
        traceback.print_exc()
