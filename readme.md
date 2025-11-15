# Hansard Thematic Explorer 2001

Interactive visualisation of semantic themes in UK Parliamentary debates from 2001.

## Data Sources

- Hansard debate transcripts (2001)
- Historical Thesaurus semantic tagging scheme

## Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your data files in:
# - data/raw/hansard_2001.csv
# - data/raw/thematic_headings.csv

# Run processing pipeline
python scripts/01_process_and_aggregate.py
```

## Project Structure

See `docs/` for methodology and data structure documentation.

## License

[Choose appropriate license]
```

**4. requirements.txt** (initial version):
```
pandas>=2.0.0
pyarrow>=12.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

**5. sample_hansard_2001.csv** (extract ~50 rows from your real file)

---

## Claude Code Prompts (Use in Order)

### Prompt 1: Initial Setup and Data Validation
```
Create a Python script at scripts/01_process_and_aggregate.py that:

1. Reads data/raw/hansard_2001.csv (6 million rows, ~400MB)
2. Reads data/raw/thematic_headings.csv 
3. Validates the data:
   - Check for missing values
   - Verify tid format matches thematic hierarchy (2 letters, then numbers/letters)
   - Print summary statistics (date range, unique tids, total counts)
4. Parse the tid hierarchies from both files:
   - Extract s1, s2, s3, s4, s5 levels from each tid
   - Build a complete hierarchy lookup from thematic_headings.csv
5. Add a proper date column by parsing year, mon, day columns
6. Save validated raw data to data/processed/validated_raw.parquet with appropriate compression

Use pandas with chunked reading if needed for memory efficiency. Include progress bars using tqdm. Add clear logging output.
```

### Prompt 2: Hierarchical Aggregation
```
Extend scripts/01_process_and_aggregate.py to create hierarchical aggregations:

From the validated data, create these aggregated parquet files:

1. data/processed/daily_full_tid.parquet
   - Columns: date, house, tid, s1, s2, s3, s4, s5, total_count
   - Aggregated by: date, house, full tid

2. data/processed/daily_s1.parquet
   - Columns: date, house, s1, total_count
   - Aggregated by: date, house, s1 only

3. data/processed/daily_s2.parquet
   - Columns: date, house, s1, s2, total_count
   - Aggregated by: date, house, s1+s2

4. data/processed/daily_s3.parquet
   - Similar pattern for s1+s2+s3

5. data/processed/daily_s4.parquet
   - Similar pattern for s1+s2+s3+s4

6. data/processed/daily_s5.parquet
   - Full hierarchy (essentially same as daily_full_tid but restructured)

Use efficient aggregation and compression (snappy or gzip). Each file should be significantly smaller than the raw data. Include file size reporting at the end.
```

### Prompt 3: Hierarchy Metadata
```
Create scripts/02_build_hierarchy_metadata.py that:

1. Reads data/raw/thematic_headings.csv
2. Builds a complete JSON file at data/processed/hierarchy.json with this structure:

{
  "AA": {
    "label": "The world",
    "children": {
      "01": {
        "label": "The earth",
        "children": { ... }
      },
      "02": {
        "label": "Geographical areas/references",
        "children": {
          "a": {
            "label": "Cardinal points",
            "children": {}
          }
        }
      }
    }
  }
}

3. Also create a flat lookup at data/processed/tid_labels.json:

{
  "AA": "The world",
  "AA01": "The earth",
  "AA02": "Geographical areas/references",
  "AA02a": "Cardinal points",
  ...
}

This will be used by the web interface for navigation and labels.
```

### Prompt 4: Statistical Analysis
```
Create scripts/03_calculate_statistics.py that:

For each aggregation level (s1, s2, s3, s4, s5):

1. Calculate baseline statistics for each tid:
   - Mean daily count
   - Standard deviation
   - Median
   - 95th percentile

2. For each date, calculate z-scores:
   - z = (daily_count - mean) / std_dev
   
3. Flag anomalies where |z| > 2.0

4. Output to data/processed/anomalies_s[N].parquet with columns:
   - date, house, tid (at appropriate hierarchy level), count, mean, std_dev, z_score, is_anomaly

5. Create summary file data/processed/anomaly_summary.json:
   - Top 50 most anomalous days across all tids
   - Count of anomalies by tid
   - Date ranges with most anomalies

Consider day-of-week effects (Parliament doesn't sit every day). Handle tids with low frequency appropriately (maybe minimum occurrence threshold).
```

### Prompt 5: Data Export Summary
```
Create scripts/04_export_summary.py that:

1. Generates data/processed/data_manifest.json containing:
   - File sizes of all generated parquet files
   - Row counts for each aggregation level
   - Date range covered
   - Number of unique tids at each level
   - Top 10 most frequent s1, s2 categories
   - Memory requirements estimate
   
2. Prints a summary report to console

3. Validates all expected output files exist

This will help document what data is available for the web interface.