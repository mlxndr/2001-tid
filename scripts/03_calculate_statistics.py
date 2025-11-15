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
            data_dir: Directory containing aggregated parquet files
            min_occurrences: Minimum number of occurrences for a TID to be analyzed
            z_threshold: Z-score threshold for flagging anomalies
        """
        self.data_dir = Path(data_dir)
        self.min_occurrences = min_occurrences
        self.z_threshold = z_threshold

        self.aggregation_levels = ['s1', 's2', 's3', 's4', 's5']
        self.all_anomalies = []
        self.summary_stats = {}

    def calculate_baseline_statistics(self, df, tid_cols):
        """
        Calculate baseline statistics for each TID.

        Args:
            df: DataFrame with date, house, tid columns, and total_count
            tid_cols: List of column names that form the TID identifier

        Returns:
            DataFrame with baseline statistics
        """
        # Group by TID and calculate statistics
        stats = df.groupby(tid_cols).agg({
            'total_count': ['count', 'mean', 'std', 'median', lambda x: np.percentile(x, 95)]
        }).reset_index()

        # Flatten column names
        stats.columns = tid_cols + ['num_occurrences', 'mean_count', 'std_count',
                                     'median_count', 'p95_count']

        # Filter out low-frequency TIDs
        stats = stats[stats['num_occurrences'] >= self.min_occurrences].copy()

        # Handle zero or NaN standard deviation (constant values)
        stats['std_count'] = stats['std_count'].fillna(0)
        stats.loc[stats['std_count'] == 0, 'std_count'] = 1.0  # Avoid division by zero

        logger.info(f"  Calculated stats for {len(stats):,} TIDs (min {self.min_occurrences} occurrences)")
        logger.info(f"  Mean occurrences per TID: {stats['num_occurrences'].mean():.1f}")

        return stats

    def calculate_z_scores(self, df, stats, tid_cols):
        """
        Calculate z-scores for each observation.

        Args:
            df: DataFrame with observations
            stats: DataFrame with baseline statistics
            tid_cols: List of column names that form the TID identifier

        Returns:
            DataFrame with z-scores and anomaly flags
        """
        # Merge observations with statistics
        df_with_stats = df.merge(stats, on=tid_cols, how='inner')

        # Calculate z-scores
        df_with_stats['z_score'] = (
            (df_with_stats['total_count'] - df_with_stats['mean_count']) /
            df_with_stats['std_count']
        )

        # Flag anomalies
        df_with_stats['is_anomaly'] = (
            np.abs(df_with_stats['z_score']) > self.z_threshold
        )

        # Select and rename columns for output
        output_cols = ['date', 'house'] + tid_cols + [
            'total_count', 'mean_count', 'std_count', 'median_count',
            'p95_count', 'z_score', 'is_anomaly'
        ]

        result = df_with_stats[output_cols].copy()
        result = result.rename(columns={'total_count': 'count'})

        logger.info(f"  Calculated z-scores for {len(result):,} observations")
        logger.info(f"  Found {result['is_anomaly'].sum():,} anomalies ({result['is_anomaly'].mean()*100:.1f}%)")

        return result

    def process_aggregation_level(self, level):
        """
        Process one aggregation level.

        Args:
            level: Aggregation level ('s1', 's2', 's3', 's4', or 's5')

        Returns:
            DataFrame with anomalies for this level
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing aggregation level: {level}")
        logger.info(f"{'='*80}")

        # Read aggregation file
        input_file = self.data_dir / f'daily_{level}.parquet'

        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            return None

        logger.info(f"Reading {input_file.name}...")
        df = pd.read_parquet(input_file)
        logger.info(f"  Loaded {len(df):,} rows")

        # Determine TID columns based on level
        if level == 's1':
            tid_cols = ['s1']
        elif level == 's2':
            tid_cols = ['s1', 's2']
        elif level == 's3':
            tid_cols = ['s1', 's2', 's3']
        elif level == 's4':
            tid_cols = ['s1', 's2', 's3', 's4']
        else:  # s5
            tid_cols = ['s1', 's2', 's3', 's4', 's5']

        # Calculate baseline statistics
        logger.info("Calculating baseline statistics...")
        stats = self.calculate_baseline_statistics(df, tid_cols)

        # Calculate z-scores
        logger.info("Calculating z-scores and detecting anomalies...")
        anomalies = self.calculate_z_scores(df, stats, tid_cols)

        # Add day of week information
        anomalies['day_of_week'] = pd.to_datetime(anomalies['date']).dt.day_name()

        # Save to parquet
        output_file = self.data_dir / f'anomalies_{level}.parquet'
        anomalies.to_parquet(output_file, compression='snappy', index=False)

        output_size_kb = output_file.stat().st_size / 1024
        logger.info(f"✓ Saved to {output_file.name} ({output_size_kb:.2f} KB)")

        # Store for summary
        self.all_anomalies.append({
            'level': level,
            'data': anomalies,
            'stats': stats
        })

        return anomalies

    def generate_summary(self):
        """Generate comprehensive summary of anomalies."""
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING ANOMALY SUMMARY")
        logger.info(f"{'='*80}")

        summary = {
            'generated_at': datetime.now().isoformat(),
            'parameters': {
                'min_occurrences': self.min_occurrences,
                'z_threshold': self.z_threshold
            },
            'by_level': {},
            'top_anomalous_days': [],
            'anomaly_counts_by_tid': {},
            'date_ranges_with_most_anomalies': []
        }

        # Collect all anomalies across levels
        all_anomaly_records = []

        for item in self.all_anomalies:
            level = item['level']
            data = item['data']
            stats = item['stats']

            # Level-specific summary
            anomaly_data = data[data['is_anomaly']]

            summary['by_level'][level] = {
                'total_observations': len(data),
                'total_anomalies': len(anomaly_data),
                'anomaly_rate': f"{len(anomaly_data)/len(data)*100:.2f}%" if len(data) > 0 else "0.00%",
                'unique_tids': int(stats.shape[0]),
                'avg_z_score_absolute': float(np.abs(anomaly_data['z_score']).mean()) if len(anomaly_data) > 0 else 0,
                'max_z_score_absolute': float(np.abs(anomaly_data['z_score']).max()) if len(anomaly_data) > 0 else 0
            }

            # Collect anomaly records for cross-level analysis
            if len(anomaly_data) > 0:
                # Create TID string for grouping
                if level == 's1':
                    tid_str = anomaly_data['s1'].astype(str)
                elif level == 's2':
                    tid_str = anomaly_data['s1'].astype(str) + ':' + anomaly_data['s2'].astype(str)
                elif level == 's3':
                    tid_str = (anomaly_data['s1'].astype(str) + ':' +
                              anomaly_data['s2'].astype(str) + ':' +
                              anomaly_data['s3'].astype(str))
                elif level == 's4':
                    tid_str = (anomaly_data['s1'].astype(str) + ':' +
                              anomaly_data['s2'].astype(str) + ':' +
                              anomaly_data['s3'].astype(str) + ':' +
                              anomaly_data['s4'].astype(str))
                else:  # s5
                    tid_str = (anomaly_data['s1'].astype(str) + ':' +
                              anomaly_data['s2'].astype(str) + ':' +
                              anomaly_data['s3'].astype(str) + ':' +
                              anomaly_data['s4'].astype(str) + ':' +
                              anomaly_data['s5'].astype(str))

                anomaly_records = anomaly_data.copy()
                anomaly_records['tid_string'] = tid_str
                anomaly_records['level'] = level
                all_anomaly_records.append(anomaly_records)

        # Combine all anomalies
        if all_anomaly_records:
            combined_anomalies = pd.concat(all_anomaly_records, ignore_index=True)

            # Top 50 most anomalous days (by absolute z-score)
            top_anomalies = combined_anomalies.nlargest(50, 'z_score', keep='all')[
                ['date', 'house', 'level', 'tid_string', 'count', 'mean_count', 'z_score']
            ]

            summary['top_anomalous_days'] = [
                {
                    'date': str(row['date']),
                    'house': row['house'],
                    'level': row['level'],
                    'tid': row['tid_string'],
                    'count': int(row['count']),
                    'mean_count': float(row['mean_count']),
                    'z_score': float(row['z_score'])
                }
                for _, row in top_anomalies.iterrows()
            ]

            # Count of anomalies by TID (top 30)
            tid_anomaly_counts = combined_anomalies.groupby(['level', 'tid_string']).size().reset_index(name='count')
            tid_anomaly_counts = tid_anomaly_counts.nlargest(30, 'count')

            summary['anomaly_counts_by_tid'] = [
                {
                    'level': row['level'],
                    'tid': row['tid_string'],
                    'anomaly_count': int(row['count'])
                }
                for _, row in tid_anomaly_counts.iterrows()
            ]

            # Date ranges with most anomalies (by week)
            combined_anomalies['week'] = pd.to_datetime(combined_anomalies['date']).dt.to_period('W')
            weekly_anomalies = combined_anomalies.groupby('week').size().reset_index(name='count')
            weekly_anomalies = weekly_anomalies.nlargest(10, 'count')
            weekly_anomalies['week_start'] = weekly_anomalies['week'].apply(lambda x: str(x.start_time.date()))
            weekly_anomalies['week_end'] = weekly_anomalies['week'].apply(lambda x: str(x.end_time.date()))

            summary['date_ranges_with_most_anomalies'] = [
                {
                    'week_start': row['week_start'],
                    'week_end': row['week_end'],
                    'anomaly_count': int(row['count'])
                }
                for _, row in weekly_anomalies.iterrows()
            ]

        # Save summary to JSON
        output_file = self.data_dir / 'anomaly_summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Saved summary to {output_file.name}")

        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)

        for level, stats in summary['by_level'].items():
            logger.info(f"\n{level.upper()}:")
            logger.info(f"  Total observations: {stats['total_observations']:,}")
            logger.info(f"  Total anomalies: {stats['total_anomalies']:,}")
            logger.info(f"  Anomaly rate: {stats['anomaly_rate']}")
            logger.info(f"  Unique TIDs: {stats['unique_tids']:,}")
            logger.info(f"  Max |z-score|: {stats['max_z_score_absolute']:.2f}")

        if summary['top_anomalous_days']:
            logger.info("\nTop 10 Most Anomalous Days:")
            for i, anomaly in enumerate(summary['top_anomalous_days'][:10], 1):
                logger.info(f"  {i}. {anomaly['date']} - {anomaly['tid']} "
                          f"(z={anomaly['z_score']:.2f}, count={anomaly['count']}, mean={anomaly['mean_count']:.1f})")

        logger.info("\n" + "="*80)

    def process(self):
        """Run the complete anomaly detection pipeline."""
        logger.info("="*80)
        logger.info("STARTING ANOMALY DETECTION")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Minimum occurrences: {self.min_occurrences}")
        logger.info(f"Z-score threshold: {self.z_threshold}")
        logger.info("="*80)

        # Process each aggregation level
        for level in self.aggregation_levels:
            self.process_aggregation_level(level)

        # Generate summary
        self.generate_summary()
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
    data_dir = base_dir / "data" / "processed"

    # Verify data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please run 01_process_and_aggregate.py first")
        sys.exit(1)

    # Create detector and run
    # Note: For full dataset, use min_occurrences=5 or higher
    # For sample data (only 3 dates), use min_occurrences=2
    detector = AnomalyDetector(
        data_dir=data_dir,
        min_occurrences=2,  # Analyze TIDs that appear at least 2 times (use 5+ for full data)
        z_threshold=2.0     # Flag anomalies with |z-score| > 2.0
    )

    detector.process()

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
