#!/usr/bin/env python3
"""Add residue one-hot columns to all CSV files under Data_1."""

import csv
import os
import re
from typing import List, Sequence

BASE_DIR = "/home/ziyu-song/Graph_pKa/Data_1"

RESIDUE_ORDER: Sequence[str] = (
    "ASPARTATE",
    "GLUTAMATE",
    "HISTIDINE",
    "LYSINE",
)

COLUMN_NAMES: Sequence[str] = (
    "Residue Name_Aspartate",
    "Residue Name_Glutamate",
    "Residue Name_Histidine",
    "Residue Name_Lysine",
)

CODE_PATTERN = re.compile(r"\.(ASP|GLU|HIS|LYS)(?=(_|\.|$))", re.IGNORECASE)
FULL_PATTERN = re.compile(
    r"\.(ASPARTATE|GLUTAMATE|HISTIDINE|LYSINE)(?=(_|\.|$))",
    re.IGNORECASE,
)
CODE_TO_RESIDUE = {
    "ASP": "ASPARTATE",
    "GLU": "GLUTAMATE",
    "HIS": "HISTIDINE",
    "LYS": "LYSINE",
}


def infer_residue(file_name: str) -> str | None:
    """Infer the residue from the CSV file name."""

    match = CODE_PATTERN.search(file_name)
    if match:
        return CODE_TO_RESIDUE[match.group(1).upper()]

    match = FULL_PATTERN.search(file_name)
    if match:
        return match.group(1).upper()

    return None


def add_one_hot_columns(rows: List[List[str]], residue: str) -> List[List[str]]:
    """Return a new list of rows with the residue one-hot columns prepended."""

    one_hot = ["1" if res == residue else "0" for res in RESIDUE_ORDER]
    updated_rows: List[List[str]] = []

    header = rows[0]
    updated_rows.append(list(COLUMN_NAMES) + header)

    for row in rows[1:]:
        updated_rows.append(one_hot + row)

    return updated_rows


def process_file(path: str) -> str:
    """Process a single CSV file. Returns a status string."""

    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return "empty"

    header = rows[0]
    if all(name in header for name in COLUMN_NAMES):
        return "already_processed"

    residue = infer_residue(os.path.basename(path))
    if residue is None:
        return "unknown_residue"

    updated_rows = add_one_hot_columns(rows, residue)

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(updated_rows)

    return "updated"


def main(base_dir: str = BASE_DIR) -> None:
    updated = 0
    skipped_unknown = []
    already = 0
    empty = 0

    for root, _dirs, files in os.walk(base_dir):
        for file_name in files:
            if not file_name.lower().endswith(".csv"):
                continue
            file_path = os.path.join(root, file_name)
            status = process_file(file_path)
            if status == "updated":
                updated += 1
            elif status == "unknown_residue":
                skipped_unknown.append(file_path)
            elif status == "already_processed":
                already += 1
            elif status == "empty":
                empty += 1

    print(f"Updated {updated} CSV files.")
    print(f"Skipped {already} files that already had the residue columns.")
    if empty:
        print(f"Skipped {empty} empty CSV files.")
    if skipped_unknown:
        print("Skipped files with unknown residue: ")
        for path in skipped_unknown:
            print(f"  {path}")


if __name__ == "__main__":
    main()
