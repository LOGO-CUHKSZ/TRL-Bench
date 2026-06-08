# TabSketchFM Data Downloader Usage

This document describes how to download the pretraining data for TabSketchFM.

## Overview

TabSketchFM is pretrained on ~127,000 tables from various open data sources. The URLs are listed in `pretraining_tables.txt`. Due to the age of some URLs (2010-2018), expect a 40-60% success rate for downloads.

## Quick Start

```bash
cd /path/to/TRL-Bench
source load_env

cd models/tabsketchfm/data_downloader

# Test with a small sample first
python download_manager.py --start 599 --limit 10

# Check results
ls -la /path/to/TRL-Bench/datasets/tabsketchfm/opendata/
```

## Full Download

```bash
# Download all URLs (will take several days)
python download_manager.py

# Or download in batches
python download_manager.py --start 0 --limit 10000
python download_manager.py --start 10000 --limit 10000
# ... continue for remaining URLs
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Output locations
output_dir: "/path/to/TRL-Bench/datasets/tabsketchfm/opendata"
metadata_dir: "/path/to/TRL-Bench/datasets/tabsketchfm/opendata_metadata"

# Parallel workers (increase for faster downloads)
max_workers: 20

# Timeout per request
timeout: 60
```

## Output Structure

Downloaded data is organized by domain:

```
datasets/tabsketchfm/
├── opendata/
│   ├── data_gov_au/
│   │   ├── file1.csv
│   │   └── file2.csv
│   ├── socrata_com/
│   │   └── ...
│   └── [other_domains]/
│
└── opendata_metadata/
    ├── data_gov_au/
    │   ├── file1.csv.meta
    │   └── file2.csv.meta
    └── ...
```

## Command-Line Options

```bash
python download_manager.py [OPTIONS]

Options:
  --limit N        Download only N URLs
  --start N        Start from URL index N (0-indexed)
  --config FILE    Use custom config file
  --reset-failed   Retry previously failed URLs
```

## Progress Tracking

Progress is saved automatically to `download_progress.json`. The downloader will resume from the last checkpoint if interrupted.

```bash
# Check progress
cat download_progress.json

# View failed URLs
cat failed_urls.txt
```

## After Downloading

Once data is downloaded, preprocess it for pretraining:

```bash
cd /path/to/TRL-Bench
source load_env

# Preprocess OpenData
bash models/tabsketchfm/scripts/preprocessing/preprocess_opendata.sh \
    --input_dir datasets/tabsketchfm/opendata \
    --metadata_dir datasets/tabsketchfm/opendata_metadata \
    --output_dir datasets/tabsketchfm/opendata_processed

# Create train/val/test splits
python models/tabsketchfm/scripts/data_utils/create_data_splits.py \
    --opendata_dir datasets/tabsketchfm/opendata \
    --metadata_dir datasets/tabsketchfm/opendata_metadata \
    --processed_dir datasets/tabsketchfm/opendata_processed \
    --output datasets/tabsketchfm/data_splits.json.bz2
```

## Expected Results

- **Total URLs**: 127,933
- **Expected Success Rate**: 40-60%
- **Data Size**: Varies (100GB - 1TB depending on success rate)
- **Download Time**: 2-5 days with 20 workers

## Troubleshooting

### Many URLs failing
This is expected for old URLs. Try URLs from more reliable sources:
```bash
# data.gov.au URLs start around index 599
python download_manager.py --start 599 --limit 1000
```

### Connection timeouts
Increase timeout in config.yaml or reduce worker count.

### SSL errors
SSL verification is disabled by default for compatibility with old servers.

## Notes

- The original pretraining data (231GB) used by the paper authors is available separately
- For reproduction, you can use the preprocessed data if available, or download fresh data
- Many URLs from 2010-2016 are no longer accessible
