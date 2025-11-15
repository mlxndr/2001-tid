#!/usr/bin/env python3
"""
Calculate statistics and detect anomalies in Hansard TID data.

This script:
1. Reads aggregated data at each hierarchy level (s1-s5)
2. Calculates baseline statistics for each tid:
   - Mean daily count
   - Standard deviation
   - Median
   - 95th percentile
3. Calculates z-scores for each date
4. Flags anomalies where |z| > 2.0
5. Outputs anomaly data to parquet files
6. Creates summary JSON with top anomalies and insights
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in Hansard TID data using statistical methods."""

    def __init__(self, input_dir, output_dir, min_observations=10, z_threshold=2.0):
        """
        Initialize the anomaly detector.

        Args:
            input_dir: Directory containing processed parquet files
            output_dir: Directory to save anomaly results
            min_observations: Minimum number of observations required for a tid
            z_threshold: Z-score threshold for anomaly detection
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_observations = min_observations
        self.z_threshold = z_threshold

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for anomaly summaries
        self.all_anomalies = []
        self.anomaly_counts_by_tid = {}
        self.date_anomaly_counts = {}

    def process_aggregation_level(self, level_name):
        """
        Process a single aggregation level (s1-s5).

        Args:
            level_name: Name of the aggregation level (e.g., 's1', 's2', etc.)

        Returns:
            DataFrame with anomalies detected
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {level_name.upper()}")
        logger.info(f"{'='*80}")

        # Read the data
        input_file = self.input_dir / f"daily_{level_name}.parquet"
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            return None

        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df):,} rows from {input_file.name}")

        # Determine tid columns based on level
        tid_columns = self._get_tid_columns(level_name)
        logger.info(f"TID columns: {tid_columns}")

        # Create a composite tid identifier
        if len(tid_columns) > 1:
            df['tid'] = df[tid_columns].apply(
                lambda x: ':'.join([str(v) for v in x if pd.notna(v)]),
                axis=1
            )
        else:
            df['tid'] = df[tid_columns[0]].astype(str)

        # Filter out days with no activity (Parliament doesn't sit every day)
        # We'll only calculate statistics based on days that actually have data
        logger.info("Filtering to days with parliamentary activity...")
        active_dates = df.groupby('date')['total_count'].sum()
        active_dates = active_dates[active_dates > 0].index
        df = df[df['date'].isin(active_dates)]
        logger.info(f"Retained {len(df):,} rows on {len(active_dates):,} active days")

        # Calculate baseline statistics per tid and house
        logger.info("Calculating baseline statistics...")
        stats = self._calculate_baseline_statistics(df, tid_columns)

        # Filter tids with insufficient observations
        sufficient_obs = stats['observation_count'] >= self.min_observations
        logger.info(f"TIDs with sufficient observations (>={self.min_observations}): "
                   f"{sufficient_obs.sum():,} / {len(stats):,}")

        stats = stats[sufficient_obs].copy()

        if len(stats) == 0:
            logger.warning(f"No tids with sufficient observations for {level_name}")
            return None

        # Merge statistics back to original data
        merge_cols = ['house', 'tid']
        df = df.merge(stats, on=merge_cols, how='inner')

        # Calculate z-scores
        logger.info("Calculating z-scores...")
        df = self._calculate_z_scores(df)

        # Flag anomalies
        df['is_anomaly'] = df['z_score'].abs() > self.z_threshold
        anomaly_count = df['is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(df) * 100) if len(df) > 0 else 0

        logger.info(f"Detected {anomaly_count:,} anomalies ({anomaly_pct:.2f}%)")

        # Select output columns
        output_cols = (
            ['date', 'house'] +
            tid_columns +
            ['total_count', 'mean', 'std_dev', 'median', 'percentile_95',
             'z_score', 'is_anomaly']
        )

        # Rename total_count to count for clarity
        df = df.rename(columns={'total_count': 'count'})
        output_cols = [c if c != 'total_count' else 'count' for c in output_cols]

        result = df[output_cols].copy()

        # Save to parquet
        output_file = self.output_dir / f"anomalies_{level_name}.parquet"
        result.to_parquet(output_file, compression='snappy', index=False)
        output_size = output_file.stat().st_size / 1024
        logger.info(f"Saved {len(result):,} rows to {output_file.name} ({output_size:.2f} KB)")

        # Collect anomalies for summary
        self._collect_anomalies_for_summary(result, level_name, tid_columns)

        return result

    def _get_tid_columns(self, level_name):
        """Get the tid columns for a given aggregation level."""
        level_map = {
            's1': ['s1'],
            's2': ['s1', 's2'],
            's3': ['s1', 's2', 's3'],
            's4': ['s1', 's2', 's3', 's4'],
            's5': ['s1', 's2', 's3', 's4', 's5']
        }
        return level_map.get(level_name, [level_name])

    def _calculate_baseline_statistics(self, df, tid_columns):
        """
        Calculate baseline statistics for each tid.

        Args:
            df: DataFrame with date, house, tid columns
            tid_columns: List of tid column names

        Returns:
            DataFrame with statistics per tid
        """
        # Group by house and tid
        grouped = df.groupby(['house', 'tid'])['total_count']

        # Calculate statistics
        stats = pd.DataFrame({
            'mean': grouped.mean(),
            'std_dev': grouped.std(),
            'median': grouped.median(),
            'percentile_95': grouped.quantile(0.95),
            'observation_count': grouped.count()
        }).reset_index()

        # Handle cases where std_dev is 0 or NaN
        # This can happen when all values are the same
        stats['std_dev'] = stats['std_dev'].fillna(1.0)
        stats.loc[stats['std_dev'] == 0, 'std_dev'] = 1.0

        return stats

    def _calculate_z_scores(self, df):
        """
        Calculate z-scores for each observation.

        Args:
            df: DataFrame with count, mean, and std_dev columns

        Returns:
            DataFrame with z_score column added
        """
        df['z_score'] = (df['total_count'] - df['mean']) / df['std_dev']

        # Handle any NaN or inf values
        df['z_score'] = df['z_score'].fillna(0.0)
        df['z_score'] = df['z_score'].replace([np.inf, -np.inf], 0.0)

        return df

    def _collect_anomalies_for_summary(self, df, level_name, tid_columns):
        """Collect anomalies for summary statistics."""
        anomalies = df[df['is_anomaly']].copy()

        if len(anomalies) == 0:
            return

        # Add level information
        anomalies['level'] = level_name

        # Create tid string for summary
        if 'tid' not in anomalies.columns:
            if len(tid_columns) > 1:
                anomalies['tid'] = anomalies[tid_columns].apply(
                    lambda x: ':'.join([str(v) for v in x if pd.notna(v)]),
                    axis=1
                )
            else:
                anomalies['tid'] = anomalies[tid_columns[0]].astype(str)

        # Collect for overall summary
        for _, row in anomalies.iterrows():
            self.all_anomalies.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'house': row['house'],
                'level': level_name,
                'tid': row['tid'],
                'count': int(row['count']),
                'mean': float(row['mean']),
                'z_score': float(row['z_score'])
            })

            # Count anomalies by tid
            tid_key = f"{level_name}:{row['tid']}"
            self.anomaly_counts_by_tid[tid_key] = \
                self.anomaly_counts_by_tid.get(tid_key, 0) + 1

            # Count anomalies by date
            date_key = row['date'].strftime('%Y-%m-%d')
            self.date_anomaly_counts[date_key] = \
                self.date_anomaly_counts.get(date_key, 0) + 1

    def create_summary(self):
        """Create summary JSON file with anomaly insights."""
        logger.info(f"\n{'='*80}")
        logger.info("CREATING ANOMALY SUMMARY")
        logger.info(f"{'='*80}")

        # Sort all anomalies by absolute z-score
        all_anomalies_sorted = sorted(
            self.all_anomalies,
            key=lambda x: abs(x['z_score']),
            reverse=True
        )

        # Top 50 most anomalous days
        top_50_anomalies = all_anomalies_sorted[:50]

        # Top tids by anomaly count
        top_tids = sorted(
            self.anomaly_counts_by_tid.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]

        # Dates with most anomalies
        top_dates = sorted(
            self.date_anomaly_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]

        # Calculate date ranges with high anomaly density
        date_ranges = self._find_anomaly_clusters()

        # Create summary dictionary
        summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'z_threshold': self.z_threshold,
                'min_observations': self.min_observations,
                'total_anomalies': len(self.all_anomalies),
                'unique_tids_with_anomalies': len(self.anomaly_counts_by_tid),
                'unique_dates_with_anomalies': len(self.date_anomaly_counts)
            },
            'top_50_most_anomalous_days': [
                {
                    **anomaly,
                    'abs_z_score': abs(anomaly['z_score'])
                }
                for anomaly in top_50_anomalies
            ],
            'top_50_tids_by_anomaly_count': [
                {
                    'tid': tid,
                    'anomaly_count': count
                }
                for tid, count in top_tids
            ],
            'top_50_dates_by_anomaly_count': [
                {
                    'date': date,
                    'anomaly_count': count
                }
                for date, count in top_dates
            ],
            'date_ranges_with_most_anomalies': date_ranges
        }

        # Save to JSON
        output_file = self.output_dir / 'anomaly_summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved anomaly summary to {output_file.name}")
        logger.info(f"Total anomalies detected: {len(self.all_anomalies):,}")
        logger.info(f"Unique TIDs with anomalies: {len(self.anomaly_counts_by_tid):,}")
        logger.info(f"Unique dates with anomalies: {len(self.date_anomaly_counts):,}")

        # Display top 10 most anomalous days
        logger.info("\nTop 10 Most Anomalous Days:")
        for i, anomaly in enumerate(top_50_anomalies[:10], 1):
            logger.info(
                f"  {i}. {anomaly['date']} - {anomaly['house']} - "
                f"{anomaly['level']}:{anomaly['tid']} - "
                f"z={anomaly['z_score']:.2f}, count={anomaly['count']}, mean={anomaly['mean']:.1f}"
            )

        # Display top 10 TIDs by anomaly count
        logger.info("\nTop 10 TIDs by Anomaly Count:")
        for i, (tid, count) in enumerate(top_tids[:10], 1):
            logger.info(f"  {i}. {tid}: {count:,} anomalies")

        return summary

    def _find_anomaly_clusters(self):
        """Find date ranges with high anomaly density."""
        if not self.date_anomaly_counts:
            return []

        # Convert to DataFrame for easier manipulation
        date_counts = pd.DataFrame([
            {'date': pd.to_datetime(date), 'count': count}
            for date, count in self.date_anomaly_counts.items()
        ]).sort_values('date')

        # Calculate 7-day rolling sum
        date_counts = date_counts.set_index('date')
        date_counts['rolling_7day'] = date_counts['count'].rolling(7, min_periods=1).sum()

        # Find top 10 7-day windows
        top_windows = date_counts.nlargest(10, 'rolling_7day')

        return [
            {
                'center_date': date.strftime('%Y-%m-%d'),
                'anomaly_count_7day_window': int(row['rolling_7day']),
                'anomaly_count_single_day': int(row['count'])
            }
            for date, row in top_windows.iterrows()
        ]

    def process_all_levels(self):
        """Process all aggregation levels and create summary."""
        logger.info("="*80)
        logger.info("STARTING ANOMALY DETECTION")
        logger.info("="*80)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Minimum observations: {self.min_observations}")
        logger.info(f"Z-score threshold: {self.z_threshold}")
        logger.info("="*80)

        # Process each level
        levels = ['s1', 's2', 's3', 's4', 's5']
        results = {}

        for level in levels:
            result = self.process_aggregation_level(level)
            if result is not None:
                results[level] = result

        # Create summary
        if self.all_anomalies:
            self.create_summary()
        else:
            logger.warning("No anomalies detected across any level")

        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION COMPLETE")
        logger.info("="*80)

        return results


def main():
    """Main entry point."""
    # Determine base directory (repository root)
    base_dir = Path(__file__).parent.parent

    # Set up paths
    input_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "data" / "processed"

    # Verify input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Create anomaly detector
    detector = AnomalyDetector(
        input_dir=input_dir,
        output_dir=output_dir,
        min_observations=10,  # Minimum 10 observations to calculate statistics
        z_threshold=2.0       # Flag anomalies where |z| > 2.0
    )

    # Process all levels
    detector.process_all_levels()


if __name__ == "__main__":
    main()
