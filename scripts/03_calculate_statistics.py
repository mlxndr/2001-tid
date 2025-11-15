#!/usr/bin/env python3
"""
Calculate statistics and detect anomalies in Hansard data aggregations.

For each aggregation level (s1-s5), this script:
1. Calculates baseline statistics for each TID
2. Computes z-scores for each date
3. Flags anomalies where |z| > 2.0
4. Outputs anomaly data to parquet files
5. Creates a comprehensive summary JSON file

Considers day-of-week effects and filters low-frequency TIDs.
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
    """Detect anomalies in Hansard TID usage patterns."""

    def __init__(self, data_dir, min_occurrences=5, z_threshold=2.0):
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

        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION COMPLETE")
        logger.info("="*80)


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


if __name__ == "__main__":
    main()
