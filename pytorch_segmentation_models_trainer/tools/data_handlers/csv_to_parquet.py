# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_csv_to_parquet(
    input_path: Path, output_path: Path = None, recursive: bool = False
):
    """Converts a CSV file or a directory of CSV files to Parquet.

    Args:
        input_path (Path): Path to CSV file or directory.
        output_path (Path, optional): Path to output file or directory.
            If None, uses the same path with .parquet suffix.
        recursive (bool, optional): Whether to search recursively in directory.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            print(f"Skipping {input_path}: not a CSV file.")
            return

        if output_path is None:
            output_path = input_path.with_suffix(".parquet")

        print(f"Converting {input_path} -> {output_path}")
        df = pd.read_csv(input_path)
        df.to_parquet(output_path, index=False)
        print("Done.")

    elif input_path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        csv_files = list(input_path.glob(pattern))

        if not csv_files:
            print(f"No CSV files found in {input_path}")
            return

        print(f"Found {len(csv_files)} CSV files. Converting...")
        for csv_file in tqdm(csv_files):
            # Resolve output path
            if output_path is None:
                target_parquet = csv_file.with_suffix(".parquet")
            else:
                rel_path = csv_file.relative_to(input_path)
                target_parquet = output_path / rel_path.with_suffix(".parquet")
                target_parquet.parent.mkdir(parents=True, exist_ok=True)

            try:
                df = pd.read_csv(csv_file)
                df.to_parquet(target_parquet, index=False)
            except Exception as e:
                print(f"Failed to convert {csv_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV datasets to Parquet format."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output path (file or directory)."
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursive search for CSV files in directory.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist.")
        return

    convert_csv_to_parquet(input_path, output_path, args.recursive)


if __name__ == "__main__":
    main()
