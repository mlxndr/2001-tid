#!/usr/bin/env python3
"""
Quick preview of parquet files.

Usage:
    python scripts/preview_parquet.py data/processed/validated_raw.parquet
    python scripts/preview_parquet.py data/processed/validated_raw.parquet --rows 20
"""

import pandas as pd
import sys
from pathlib import Path

def preview_parquet(file_path, num_rows=10):
    """Preview a parquet file."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Reading: {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024**2:.2f} MB")
    print("=" * 80)

    # Read parquet file
    df = pd.read_parquet(file_path)

    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Column info
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col} ({df[col].dtype})")

    # First rows
    print(f"\nFirst {num_rows} rows:")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(num_rows))

    # Basic statistics
    print(f"\n\nBasic Statistics:")
    print("=" * 80)
    print(df.describe())

    # Missing values
    print(f"\n\nMissing Values:")
    print("=" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
        else:
            print(f"  {col}: 0")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_parquet.py <file.parquet> [--rows N]")
        sys.exit(1)

    file_path = sys.argv[1]
    num_rows = 10

    if "--rows" in sys.argv:
        idx = sys.argv.index("--rows")
        num_rows = int(sys.argv[idx + 1])

    preview_parquet(file_path, num_rows)
