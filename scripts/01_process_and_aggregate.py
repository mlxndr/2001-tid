#!/usr/bin/env python3
"""
Process and aggregate Hansard data with thematic headings.

This script:
1. Reads hansard_2001.csv (with chunked reading for large files)
2. Reads thematic_headings.csv
3. Validates the data (missing values, tid format, summary statistics)
4. Parses tid hierarchies from both files
5. Adds proper date column
6. Saves validated raw data to parquet with compression
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HansardProcessor:
    """Process and validate Hansard data with thematic headings."""

    def __init__(self, hansard_file, thematic_file, output_file, use_sample=False):
        """
        Initialize the processor.

        Args:
            hansard_file: Path to hansard CSV file
            thematic_file: Path to thematic headings CSV file
            output_file: Path to output parquet file
            use_sample: If True, use sample_hansard_2001.csv, else hansard_2001.csv
        """
        self.hansard_file = Path(hansard_file)
        self.thematic_file = Path(thematic_file)
        self.output_file = Path(output_file)
        self.use_sample = use_sample

        # TID pattern: starts with 2 letters, then numbers/letters/colons
        self.tid_pattern = re.compile(r'^[A-Z]{2}([:\da-z]+)?$|^NULL$|^\d{2}:\d{2}$')

        self.hansard_df = None
        self.thematic_df = None
        self.hierarchy_lookup = {}

    def read_thematic_headings(self):
        """Read and parse thematic headings file."""
        logger.info(f"Reading thematic headings from {self.thematic_file}")

        # Read thematic headings
        self.thematic_df = pd.read_csv(self.thematic_file)

        logger.info(f"Loaded {len(self.thematic_df):,} thematic headings")
        logger.info(f"Columns: {list(self.thematic_df.columns)}")

        # Build hierarchy lookup
        logger.info("Building hierarchy lookup from thematic headings...")
        self._build_hierarchy_lookup()

        return self.thematic_df

    def _build_hierarchy_lookup(self):
        """Build a complete hierarchy lookup from thematic_headings."""
        for _, row in tqdm(self.thematic_df.iterrows(),
                          total=len(self.thematic_df),
                          desc="Building hierarchy lookup"):
            # Reconstruct tid from hierarchy levels
            tid_parts = []
            for level in ['s1', 's2', 's3', 's4', 's5']:
                if pd.notna(row[level]) and row[level] != '':
                    tid_parts.append(str(row[level]))

            if tid_parts:
                tid = ':'.join(tid_parts)
                self.hierarchy_lookup[tid] = {
                    's1': row['s1'] if pd.notna(row['s1']) else None,
                    's2': row['s2'] if pd.notna(row['s2']) else None,
                    's3': row['s3'] if pd.notna(row['s3']) else None,
                    's4': row['s4'] if pd.notna(row['s4']) else None,
                    's5': row['s5'] if pd.notna(row['s5']) else None,
                    'heading': row['thematicheading'] if pd.notna(row['thematicheading']) else None
                }

        logger.info(f"Built hierarchy lookup with {len(self.hierarchy_lookup):,} entries")

    def read_hansard_data(self, chunksize=100000):
        """
        Read hansard data with optional chunking for large files.

        Args:
            chunksize: Number of rows to read per chunk (None to read all at once)
        """
        logger.info(f"Reading Hansard data from {self.hansard_file}")

        # Check file size
        file_size_mb = self.hansard_file.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        # For small files or sample, read all at once
        if file_size_mb < 100 or self.use_sample:
            logger.info("Reading entire file at once...")
            self.hansard_df = pd.read_csv(self.hansard_file)
        else:
            # For large files, read in chunks
            logger.info(f"Reading file in chunks of {chunksize:,} rows...")
            chunks = []
            for chunk in tqdm(pd.read_csv(self.hansard_file, chunksize=chunksize),
                            desc="Reading chunks"):
                chunks.append(chunk)
            self.hansard_df = pd.concat(chunks, ignore_index=True)

        logger.info(f"Loaded {len(self.hansard_df):,} rows")
        logger.info(f"Columns: {list(self.hansard_df.columns)}")
        logger.info(f"Memory usage: {self.hansard_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return self.hansard_df

    def validate_data(self):
        """Validate the hansard data and print summary statistics."""
        logger.info("=" * 80)
        logger.info("DATA VALIDATION")
        logger.info("=" * 80)

        # Check for missing values
        logger.info("\nMissing Values:")
        missing = self.hansard_df.isnull().sum()
        missing_pct = (missing / len(self.hansard_df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        logger.info("\n" + missing_df.to_string(index=False))

        # Verify tid format
        logger.info("\nValidating TID format...")
        invalid_tids = self.hansard_df[~self.hansard_df['tid'].astype(str).str.match(self.tid_pattern, na=False)]

        if len(invalid_tids) > 0:
            logger.warning(f"Found {len(invalid_tids):,} rows with invalid TID format")
            logger.warning(f"Sample invalid TIDs: {invalid_tids['tid'].head(10).tolist()}")
        else:
            logger.info("✓ All TIDs match expected format")

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)

        # Date range
        logger.info("\nDate Range:")
        logger.info(f"  Years: {sorted(self.hansard_df['year'].unique())}")
        logger.info(f"  Months: {sorted(self.hansard_df['mon'].unique())}")
        logger.info(f"  Days: min={self.hansard_df['day'].min()}, max={self.hansard_df['day'].max()}")

        # TID statistics
        logger.info("\nTID Statistics:")
        unique_tids = self.hansard_df['tid'].nunique()
        logger.info(f"  Unique TIDs: {unique_tids:,}")
        logger.info(f"  NULL TIDs: {(self.hansard_df['tid'] == 'NULL').sum():,}")

        # Top TIDs by frequency
        logger.info("\nTop 10 TIDs by frequency:")
        top_tids = self.hansard_df['tid'].value_counts().head(10)
        for tid, count in top_tids.items():
            pct = (count / len(self.hansard_df) * 100)
            logger.info(f"  {tid}: {count:,} ({pct:.2f}%)")

        # Total counts
        logger.info("\nTotal Counts:")
        logger.info(f"  Total rows: {len(self.hansard_df):,}")
        logger.info(f"  Total 'tot' sum: {self.hansard_df['tot'].sum():,}")
        logger.info(f"  Mean 'tot' per row: {self.hansard_df['tot'].mean():.2f}")
        logger.info(f"  Median 'tot' per row: {self.hansard_df['tot'].median():.0f}")

        # House distribution
        logger.info("\nHouse Distribution:")
        house_counts = self.hansard_df['house'].value_counts()
        for house, count in house_counts.items():
            pct = (count / len(self.hansard_df) * 100)
            logger.info(f"  {house}: {count:,} ({pct:.2f}%)")

        logger.info("\n" + "=" * 80)

    def parse_tid_hierarchy(self):
        """Parse TID hierarchies and add s1-s5 columns to hansard data."""
        logger.info("Parsing TID hierarchies...")

        def extract_hierarchy(tid):
            """Extract s1-s5 levels from a tid string."""
            if pd.isna(tid) or tid == 'NULL':
                return [None, None, None, None, None]

            tid_str = str(tid)
            parts = tid_str.split(':')

            # Pad with None to ensure 5 levels
            hierarchy = parts + [None] * (5 - len(parts))
            return hierarchy[:5]

        # Apply hierarchy extraction
        logger.info("Extracting hierarchy levels from TIDs...")
        hierarchies = []
        for tid in tqdm(self.hansard_df['tid'], desc="Parsing TIDs"):
            hierarchies.append(extract_hierarchy(tid))

        # Convert to DataFrame and add columns
        hierarchy_df = pd.DataFrame(
            hierarchies,
            columns=['s1', 's2', 's3', 's4', 's5']
        )

        # Add hierarchy columns to main dataframe
        self.hansard_df[['s1', 's2', 's3', 's4', 's5']] = hierarchy_df

        # Log hierarchy statistics
        logger.info("\nHierarchy Level Statistics:")
        for level in ['s1', 's2', 's3', 's4', 's5']:
            non_null = self.hansard_df[level].notna().sum()
            unique = self.hansard_df[level].nunique()
            logger.info(f"  {level}: {non_null:,} non-null values, {unique:,} unique")

        # Check coverage against thematic headings
        logger.info("\nChecking TID coverage in thematic headings...")
        tids_in_data = set(self.hansard_df['tid'].unique())
        tids_in_thematic = set(self.hierarchy_lookup.keys())

        matched = tids_in_data.intersection(tids_in_thematic)
        unmatched = tids_in_data - tids_in_thematic

        logger.info(f"  TIDs in data: {len(tids_in_data):,}")
        logger.info(f"  TIDs in thematic headings: {len(tids_in_thematic):,}")
        logger.info(f"  Matched: {len(matched):,} ({len(matched)/len(tids_in_data)*100:.1f}%)")
        logger.info(f"  Unmatched: {len(unmatched):,} ({len(unmatched)/len(tids_in_data)*100:.1f}%)")

        if len(unmatched) > 0 and len(unmatched) <= 20:
            # Filter out NaN values before sorting
            valid_unmatched = [tid for tid in unmatched if pd.notna(tid)]
            logger.info(f"  Unmatched TIDs: {sorted(valid_unmatched)}")
        elif len(unmatched) > 20:
            # Filter out NaN values before sorting
            valid_unmatched = [tid for tid in unmatched if pd.notna(tid)]
            logger.info(f"  Sample unmatched TIDs: {sorted(valid_unmatched)[:20]}")

    def add_date_column(self):
        """Add proper date column by parsing year, month, day columns."""
        logger.info("Adding date column...")

        # Month mapping
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # Convert month names to numbers
        self.hansard_df['month_num'] = self.hansard_df['mon'].map(month_map)

        # Create date column
        try:
            self.hansard_df['date'] = pd.to_datetime(
                self.hansard_df[['year', 'month_num', 'day']].rename(
                    columns={'month_num': 'month'}
                )
            )
            logger.info(f"✓ Date column added successfully")
            logger.info(f"  Date range: {self.hansard_df['date'].min()} to {self.hansard_df['date'].max()}")

        except Exception as e:
            logger.error(f"Error creating date column: {e}")
            # Add date as None if there are parsing errors
            self.hansard_df['date'] = None

        # Keep month_num for reference but it's not needed in final output

    def save_to_parquet(self, compression='snappy'):
        """
        Save validated data to parquet format with compression.

        Args:
            compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')
        """
        logger.info(f"Saving to parquet with {compression} compression...")

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Select columns to save
        output_columns = [
            'id', 'filename', 'house', 'year', 'mon', 'day', 'date',
            'tid', 's1', 's2', 's3', 's4', 's5', 'tot'
        ]

        # Save to parquet
        self.hansard_df[output_columns].to_parquet(
            self.output_file,
            compression=compression,
            index=False
        )

        # Report file size
        output_size_mb = self.output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved to {self.output_file}")
        logger.info(f"  Output size: {output_size_mb:.2f} MB")
        logger.info(f"  Compression: {compression}")
        logger.info(f"  Rows: {len(self.hansard_df):,}")
        logger.info(f"  Columns: {len(output_columns)}")

    def create_hierarchical_aggregations(self, compression='snappy'):
        """
        Create hierarchical aggregations at different TID levels.

        Creates 6 parquet files:
        - daily_full_tid: Aggregated by date, house, full tid
        - daily_s1: Aggregated by date, house, s1
        - daily_s2: Aggregated by date, house, s1+s2
        - daily_s3: Aggregated by date, house, s1+s2+s3
        - daily_s4: Aggregated by date, house, s1+s2+s3+s4
        - daily_s5: Aggregated by date, house, s1+s2+s3+s4+s5

        Args:
            compression: Compression algorithm to use
        """
        logger.info("\n" + "=" * 80)
        logger.info("CREATING HIERARCHICAL AGGREGATIONS")
        logger.info("=" * 80)

        output_dir = self.output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store file sizes for reporting
        file_sizes = {}

        # 1. Daily Full TID aggregation
        logger.info("\n1. Creating daily_full_tid aggregation...")
        agg1 = self.hansard_df.groupby(
            ['date', 'house', 'tid', 's1', 's2', 's3', 's4', 's5'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg1 = agg1.rename(columns={'tot': 'total_count'})

        output1 = output_dir / 'daily_full_tid.parquet'
        agg1.to_parquet(output1, compression=compression, index=False)
        file_sizes['daily_full_tid'] = output1.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output1.name}")
        logger.info(f"   Rows: {len(agg1):,}, Size: {file_sizes['daily_full_tid']:.2f} KB")

        # 2. Daily S1 aggregation
        logger.info("\n2. Creating daily_s1 aggregation...")
        agg2 = self.hansard_df[self.hansard_df['s1'].notna()].groupby(
            ['date', 'house', 's1'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg2 = agg2.rename(columns={'tot': 'total_count'})

        output2 = output_dir / 'daily_s1.parquet'
        agg2.to_parquet(output2, compression=compression, index=False)
        file_sizes['daily_s1'] = output2.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output2.name}")
        logger.info(f"   Rows: {len(agg2):,}, Size: {file_sizes['daily_s1']:.2f} KB")

        # 3. Daily S2 aggregation
        logger.info("\n3. Creating daily_s2 aggregation...")
        agg3 = self.hansard_df[self.hansard_df['s2'].notna()].groupby(
            ['date', 'house', 's1', 's2'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg3 = agg3.rename(columns={'tot': 'total_count'})

        output3 = output_dir / 'daily_s2.parquet'
        agg3.to_parquet(output3, compression=compression, index=False)
        file_sizes['daily_s2'] = output3.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output3.name}")
        logger.info(f"   Rows: {len(agg3):,}, Size: {file_sizes['daily_s2']:.2f} KB")

        # 4. Daily S3 aggregation
        logger.info("\n4. Creating daily_s3 aggregation...")
        agg4 = self.hansard_df[self.hansard_df['s3'].notna()].groupby(
            ['date', 'house', 's1', 's2', 's3'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg4 = agg4.rename(columns={'tot': 'total_count'})

        output4 = output_dir / 'daily_s3.parquet'
        agg4.to_parquet(output4, compression=compression, index=False)
        file_sizes['daily_s3'] = output4.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output4.name}")
        logger.info(f"   Rows: {len(agg4):,}, Size: {file_sizes['daily_s3']:.2f} KB")

        # 5. Daily S4 aggregation
        logger.info("\n5. Creating daily_s4 aggregation...")
        agg5 = self.hansard_df[self.hansard_df['s4'].notna()].groupby(
            ['date', 'house', 's1', 's2', 's3', 's4'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg5 = agg5.rename(columns={'tot': 'total_count'})

        output5 = output_dir / 'daily_s4.parquet'
        agg5.to_parquet(output5, compression=compression, index=False)
        file_sizes['daily_s4'] = output5.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output5.name}")
        logger.info(f"   Rows: {len(agg5):,}, Size: {file_sizes['daily_s4']:.2f} KB")

        # 6. Daily S5 aggregation (full hierarchy)
        logger.info("\n6. Creating daily_s5 aggregation...")
        agg6 = self.hansard_df[self.hansard_df['s5'].notna()].groupby(
            ['date', 'house', 's1', 's2', 's3', 's4', 's5'],
            dropna=False
        ).agg({'tot': 'sum'}).reset_index()
        agg6 = agg6.rename(columns={'tot': 'total_count'})

        output6 = output_dir / 'daily_s5.parquet'
        agg6.to_parquet(output6, compression=compression, index=False)
        file_sizes['daily_s5'] = output6.stat().st_size / 1024
        logger.info(f"   ✓ Saved: {output6.name}")
        logger.info(f"   Rows: {len(agg6):,}, Size: {file_sizes['daily_s5']:.2f} KB")

        # Summary report
        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATION SUMMARY")
        logger.info("=" * 80)

        total_size_kb = sum(file_sizes.values())
        original_size_kb = self.output_file.stat().st_size / 1024

        logger.info(f"\nOriginal validated_raw.parquet: {original_size_kb:.2f} KB")
        logger.info(f"\nAggregated files:")
        for name, size in file_sizes.items():
            pct = (size / original_size_kb * 100) if original_size_kb > 0 else 0
            logger.info(f"  {name:20s}: {size:8.2f} KB ({pct:5.1f}% of original)")

        logger.info(f"\nTotal aggregated size: {total_size_kb:.2f} KB")
        logger.info(f"Compression ratio: {(total_size_kb / original_size_kb * 100):.1f}% of original")
        logger.info("=" * 80)

    def process(self):
        """Run the complete processing pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING HANSARD DATA PROCESSING")
        logger.info("=" * 80)
        logger.info(f"Input: {self.hansard_file}")
        logger.info(f"Thematic: {self.thematic_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info("=" * 80)

        # Step 1: Read thematic headings
        self.read_thematic_headings()

        # Step 2: Read hansard data
        self.read_hansard_data()

        # Step 3: Validate data
        self.validate_data()

        # Step 4: Parse TID hierarchies
        self.parse_tid_hierarchy()

        # Step 5: Add date column
        self.add_date_column()

        # Step 6: Save to parquet
        self.save_to_parquet(compression='snappy')

        # Step 7: Create hierarchical aggregations
        self.create_hierarchical_aggregations(compression='snappy')

        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    # Determine base directory (repository root)
    base_dir = Path(__file__).parent.parent

    # Set up file paths
    # Check if full file exists, otherwise use sample
    hansard_file = base_dir / "hansard_2001.csv"
    use_sample = False

    if not hansard_file.exists():
        hansard_file = base_dir / "sample_hansard_2001.csv"
        use_sample = True
        logger.info("Using sample data file")

    thematic_file = base_dir / "thematic_headings.csv"
    output_file = base_dir / "data" / "processed" / "validated_raw.parquet"

    # Verify input files exist
    if not hansard_file.exists():
        logger.error(f"Hansard file not found: {hansard_file}")
        sys.exit(1)

    if not thematic_file.exists():
        logger.error(f"Thematic headings file not found: {thematic_file}")
        sys.exit(1)

    # Create processor and run
    processor = HansardProcessor(
        hansard_file=hansard_file,
        thematic_file=thematic_file,
        output_file=output_file,
        use_sample=use_sample
    )

    processor.process()


if __name__ == "__main__":
    main()
