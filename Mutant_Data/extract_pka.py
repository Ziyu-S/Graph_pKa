"""Utility to extract predicted pKa values from PROPKA `.pka` outputs.

The script walks a directory containing PROPKA output files, harvests the
"SUMMARY OF THIS PREDICTION" table from each file, and writes a consolidated
CSV (or stdout) with one row per residue entry.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass
class PkaRecord:
    """Structured representation of a PROPKA summary row."""

    source_file: Path
    residue_name: str
    residue_number: str
    chain_id: str
    predicted_pka: Optional[float]
    model_pka: Optional[float]
    ligand: Optional[str]
    atom_type: Optional[str]


SUMMARY_HEADER = "SUMMARY OF THIS PREDICTION"
TABLE_HEADER_KEYWORD = "Group"


def iter_pka_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield `.pka` files under *root*."""

    pattern = "**/*.pka" if recursive else "*.pka"
    yield from root.glob(pattern)


def parse_summary_block(lines: Iterable[str]) -> Iterator[str]:
    """Yield lines belonging to the summary table."""

    lines = list(lines)
    try:
        summary_idx = next(
            idx for idx, line in enumerate(lines) if SUMMARY_HEADER in line
        )
    except StopIteration as exc:
        raise ValueError("Summary header not found") from exc

    # Find the table header following the summary marker.
    try:
        header_idx = next(
            idx
            for idx in range(summary_idx + 1, len(lines))
            if TABLE_HEADER_KEYWORD in lines[idx]
        )
    except StopIteration as exc:
        raise ValueError("Table header following summary not found") from exc

    # Data begins on the next line after the header.
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if set(stripped) == {"-"}:
            break
        yield line


def parse_record(source: Path, line: str) -> PkaRecord:
    """Convert a summary line into a `PkaRecord`."""

    def to_float(value: str) -> Optional[float]:
        if not value or value in {"NA", "N/A", "nan"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    # Split the line by whitespace
    tokens = line.split()
    
    # Expected format: RESIDUE_NAME RESIDUE_NUMBER CHAIN pKa model-pKa [ligand] [atom-type]
    residue_name = tokens[0] if len(tokens) > 0 else ""
    residue_number = tokens[1] if len(tokens) > 1 else ""
    chain_id = tokens[2] if len(tokens) > 2 else ""
    pka_str = tokens[3] if len(tokens) > 3 else ""
    model_str = tokens[4] if len(tokens) > 4 else ""
    ligand = tokens[5] if len(tokens) > 5 else None
    atom_type = tokens[6] if len(tokens) > 6 else None

    predicted = to_float(pka_str)
    model = to_float(model_str)

    return PkaRecord(
        source_file=source,
        residue_name=residue_name,
        residue_number=residue_number,
        chain_id=chain_id,
        predicted_pka=predicted,
        model_pka=model,
        ligand=ligand,
        atom_type=atom_type,
    )


def extract_records(path: Path) -> list[PkaRecord]:
    """Extract summary entries from a single `.pka` file."""

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        lines = list(parse_summary_block(text))
    except ValueError:
        return []
    return [parse_record(path, line) for line in lines]


def write_records(records: Iterable[PkaRecord], output: Optional[Path]) -> None:
    """Write records to *output* (CSV file or stdout)."""

    fieldnames = [
        "source_file",
        "residue_name",
        "residue_number",
        "chain_id",
        "predicted_pka",
        "model_pka",
        "ligand",
        "atom_type",
    ]

    writer: csv.DictWriter
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", newline="", encoding="utf-8")
        close_handle = True
    else:
        handle = sys.stdout
        close_handle = False

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "source_file": str(record.source_file),
                "residue_name": record.residue_name,
                "residue_number": record.residue_number,
                "chain_id": record.chain_id,
                "predicted_pka": record.predicted_pka,
                "model_pka": record.model_pka,
                "ligand": record.ligand,
                "atom_type": record.atom_type,
            }
        )

    if close_handle:
        handle.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract predicted pKa values from PROPKA .pka files.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PROPKA .pka outputs.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level directory (no subdirectories).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the consolidated CSV (defaults to stdout).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    # Hardcoded paths
    input_dir = Path("/home/ziyu-song/Graph_pKa/Mutant_Data/PROPKA_Results/")
    output_file = Path("/home/ziyu-song/Graph_pKa/Mutant_Data/pka_results.csv")

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return 1
    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory.")
        return 1

    files = sorted(iter_pka_files(input_dir, recursive=True))
    if not files:
        print("Error: No .pka files found in the provided directory.")
        return 1

    all_records: list[PkaRecord] = []
    for path in files:
        all_records.extend(extract_records(path))

    if not all_records:
        print("Error: No summary records extracted from the discovered .pka files.")
        return 1

    write_records(all_records, output_file)
    print(f"Successfully extracted {len(all_records)} records to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
